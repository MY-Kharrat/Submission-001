import asyncio
import hashlib
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from research.audit import AuditLogger
from research.extractor import Extractor
from research.runner import GAP_QUERIES, Runner
from research.search import SearchResult, SearchTool, clean_text
from shared.schemas import ExtractorOutput, Tender


def make_tender():
    return Tender(
        id="t1",
        title="Metro Transit Upgrade",
        issuer="MetroDOT",
        sector="unknown",
        requirements=["Req A"],
        deadline=date(2026, 12, 1),
        raw_text="text",
        source="manual",
        detected_at=datetime.now(timezone.utc),
        status="new",
    )


class FakeAuditLogger:
    def __init__(self):
        self.entries = []

    def log(self, tool_name, params={}, raw_content=""):
        entry = SimpleNamespace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            params=params,
            content=raw_content,
            hash=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
        )
        self.entries.append(entry)
        return entry


def make_runner(tender=None, **kwargs):
    tender = tender or make_tender()
    kwargs.setdefault("audit", FakeAuditLogger())
    kwargs.setdefault("fetch_tender", lambda tender_id: tender)
    return Runner(**kwargs)


def test_audit_logs_before_extraction_with_hash(tmp_path):
    logger_name = f"audit-test-{uuid.uuid4().hex}"
    log_file = tmp_path / "research.log"
    audit = AuditLogger(log_path=str(log_file), logger_name=logger_name)
    try:
        result = audit.log("search", {"query": "MetroDOT revenue"}, "raw content here")
        assert result is None
        assert AuditLogger._sha256("raw content here") == hashlib.sha256(b"raw content here").hexdigest()
        assert len(AuditLogger._sha256("raw content here")) == 64
        content = log_file.read_text(encoding="utf-8")
        assert "search" in content
        assert "MetroDOT revenue" in content
        assert hashlib.sha256(b"raw content here").hexdigest() not in content
        assert "raw content here" in content
    finally:
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()


def test_audit_log_format_has_timestamp_tool_params_content(tmp_path):
    logger_name = f"audit-test-{uuid.uuid4().hex}"
    log_file = tmp_path / "research.log"
    audit = AuditLogger(log_path=str(log_file), logger_name=logger_name)
    try:
        audit.log("search", {"query": "q"}, "hello")
        line = log_file.read_text(encoding="utf-8").strip()
        assert "INFO" in line
        assert "search" in line
        assert "q" in line
        assert "hello" in line
    finally:
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()


def test_search_returns_max_3_cleaned():
    assert clean_text("<p>Hello   <b>world</b></p>") == "Hello world"
    tool = SearchTool(max_results=10)
    assert tool.max_results == 3


@pytest.mark.asyncio
async def test_search_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert await SearchTool().search("MetroDOT revenue") == []


@pytest.mark.asyncio
async def test_extractor_outputs_strict_schema_or_null():
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(
            return_value='{"fact": "$500M budget", "relevance": "estimated_revenue", "source_url": "http://x"}'
        ),
    ):
        out = await Extractor().extract("some text", "http://x")
        assert isinstance(out, ExtractorOutput)
        assert out.relevance == "estimated_revenue"
    with patch("research.extractor.llm_call", new=AsyncMock(return_value="null")):
        assert await Extractor().extract("text", "http://x") is None


@pytest.mark.asyncio
async def test_extractor_rejects_invalid_json_and_schema():
    with patch("research.extractor.llm_call", new=AsyncMock(return_value="not json")):
        assert await Extractor().extract("text", "http://x") is None
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(return_value='{"fact": "x", "relevance": "nope", "source_url": "http://x"}'),
    ):
        assert await Extractor().extract("text", "http://x") is None
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(return_value='```json\n{"fact": "$1M", "relevance": "sector", "source_url": "http://x"}\n```'),
    ):
        out = await Extractor().extract("text", "http://x")
        assert isinstance(out, ExtractorOutput)
        assert out.relevance == "sector"


@pytest.mark.asyncio
async def test_extractor_has_no_search_or_runner_access():
    import research.extractor as mod

    src = open(mod.__file__).read()
    assert "SearchTool" not in src
    assert "Runner" not in src


@pytest.mark.asyncio
async def test_runner_logs_before_extraction_with_hash():
    tender = make_tender()
    order = []

    async def fake_search(query):
        assert "MetroDOT" in query
        return [SearchResult(url="http://a", content="CONTENT-A")]

    async def fake_extract(content, url):
        order.append("extract")
        return ExtractorOutput(fact="Partnered with BuildCo", relevance="key_partners", source_url=url)

    fake_audit = FakeAuditLogger()
    orig_log = fake_audit.log

    def recording_log(tool_name, params={}, raw_content=""):
        order.append("audit")
        return orig_log(tool_name, params, raw_content)

    fake_audit.log = recording_log
    runner = make_runner(tender, audit=fake_audit, iteration_cap=1, timeout_seconds=5)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert result.key_partners == ["Partnered with BuildCo"]
    assert result.sector is None
    assert result.confidence == "low"
    assert order[0] == "audit"
    assert order[1] == "extract"
    assert fake_audit.entries[0].tool_name == "search"
    assert len(fake_audit.entries[0].hash) == 64


@pytest.mark.asyncio
async def test_runner_loop_passes_structured_facts_only():
    tender = make_tender()
    seen_raw = []

    async def fake_search(query):
        assert "MetroDOT" in query
        return [SearchResult(url="http://a", content="CONTENT-A")]

    async def fake_extract(content, url)-> ExtractorOutput:
        seen_raw.append(content)
        return ExtractorOutput(fact="Partnered with BuildCo", relevance="key_partners", source_url=url)

    runner = make_runner(tender, iteration_cap=1, timeout_seconds=5)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert result.key_partners == ["Partnered with BuildCo"]
    assert result.sector is None
    assert result.confidence == "low"
    assert runner.audit.entries[0].tool_name == "search"
    assert seen_raw == ["CONTENT-A"]


@pytest.mark.asyncio
async def test_runner_respects_iteration_cap():
    tender = make_tender()
    calls = []

    async def fake_search(query):
        calls.append(query)
        return [SearchResult(url="http://a", content="x")]

    async def fake_extract(content, url):
        return None

    runner = make_runner(tender, iteration_cap=2, timeout_seconds=30)
    await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_runner_respects_timeout():
    tender = make_tender()

    async def slow_search(query):
        await asyncio.sleep(0.05)
        return []

    runner = make_runner(tender, iteration_cap=4, timeout_seconds=0.01)
    result = await runner.run(tender.id, search_fn=slow_search)
    assert result.confidence == "low"


@pytest.mark.asyncio
async def test_runner_truncates_results_to_3():
    tender = make_tender()
    seen = []

    async def fake_search(query):
        return [SearchResult(url=f"http://{i}", content=f"C-{i}") for i in range(5)]

    async def fake_extract(content, url):
        seen.append(content)
        return None

    runner = make_runner(tender, iteration_cap=1, timeout_seconds=30)
    await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert seen == ["C-0", "C-1", "C-2"]


@pytest.mark.asyncio
async def test_runner_search_exception_continues():
    tender = make_tender()
    calls = []

    async def failing_search(query):
        calls.append(query)
        raise RuntimeError("boom")

    runner = make_runner(tender, iteration_cap=2, timeout_seconds=30)
    result = await runner.run(tender.id, search_fn=failing_search)
    assert len(calls) == 2
    assert result.confidence == "low"


@pytest.mark.asyncio
async def test_runner_missing_tender_returns_none():
    runner = make_runner(make_tender(), fetch_tender=lambda tender_id: None)
    assert await runner.run("does-not-exist") is None


@pytest.mark.asyncio
async def test_runner_uses_fetch_tender_injection():
    tender = make_tender()
    fetched_ids = []
    runner = make_runner(
        tender,
        fetch_tender=lambda tender_id: (fetched_ids.append(tender_id), tender)[1],
        iteration_cap=1,
        timeout_seconds=5,
    )

    async def fake_search(query):
        return []

    await runner.run("t1", search_fn=fake_search)
    assert fetched_ids == ["t1"]


def test_finalize_confidence_levels():
    r = make_runner()
    assert r.finalize().confidence == "low"
    r.accumulated_facts = {
        "sector": [ExtractorOutput(fact="s", relevance="sector", source_url="u")],
        "estimated_revenue": [ExtractorOutput(fact="r", relevance="estimated_revenue", source_url="u")],
    }
    assert r.finalize().confidence == "medium"
    r.accumulated_facts["past_projects"] = [
        ExtractorOutput(fact="p", relevance="past_projects", source_url="u")
    ]
    r.accumulated_facts["key_partners"] = [
        ExtractorOutput(fact="k", relevance="key_partners", source_url="u2")
    ]
    r.sources = {"u", "u2"}
    done = r.finalize()
    assert done.confidence == "high"
    assert sorted(done.sources) == ["u", "u2"]
    assert done.past_projects == ["p"]


def test_runner_build_query_and_missing_fields():
    r = make_runner()
    assert r._build_query("MetroDOT", "estimated_revenue") == "MetroDOT estimated revenue annual budget"
    assert set(r._missing_fields()) == set(GAP_QUERIES.keys())
    r.accumulated_facts = {"sector": [ExtractorOutput(fact="s", relevance="sector", source_url="u")]}
    assert "sector" not in r._missing_fields()


def test_runner_default_caps():
    r = make_runner(iteration_cap=4, timeout_seconds=30)
    assert r.iteration_cap == 4
    assert r.timeout_seconds == 30
