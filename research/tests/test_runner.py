import asyncio
import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from research.audit import AuditLogger
from research.extractor import Extractor
from research.runner import GAP_QUERIES, Runner, TenderNotFoundError, compute_confidence
from research.search import SearchResult, SearchTool, clean_text
from shared.schemas import ExtractorOutput, Fact, Tender


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


def make_fact(category, value="v", confidence="medium"):
    return Fact(category=category, value=value, confidence=confidence)


def make_output(*facts):
    return ExtractorOutput(facts=list(facts))


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
    # Unit tests never touch the real DB: record persists in-memory instead.
    saved = []
    kwargs.setdefault("save_prospect", lambda tid, pr: saved.append((tid, pr)))
    runner = Runner(**kwargs)
    runner.saved = saved
    return runner


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


def test_audit_entries_share_one_run_id_and_it_is_hashed(tmp_path):
    """Every call in a run must be reconstructable from the log alone.

    Agent-loop calls all land within the same second, so timestamp proximity
    cannot separate them. The correlation ID is what makes "what did this run
    touch?" answerable, and it must be inside the hashed entry so it cannot be
    edited after the fact to re-attribute a call to another run.
    """
    logger_name = f"audit-test-{uuid.uuid4().hex}"
    log_file = tmp_path / "research.log"
    audit = AuditLogger(log_path=str(log_file), logger_name=logger_name)
    try:
        audit.log("search", {"query": "a"}, "page one")
        audit.log("search", {"query": "b"}, "page two")
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        run_ids = {json.loads(line.split(" | ")[3])["run_id"] for line in lines}
        assert run_ids == {audit.run_id}, f"run_id not shared across entries: {run_ids}"

        # And it is inside the hash, not just bolted onto the line.
        _ts, _lvl, _tool, params_json, entry_hash, _content = lines[0].split(" | ", 5)
        canonical = json.dumps({
            "timestamp": _ts, "tool_name": "search",
            "params": json.loads(params_json), "content": "page one",
        }, sort_keys=True, ensure_ascii=False)
        assert entry_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()


def test_separate_runs_get_separate_ids(tmp_path):
    """Two concurrent runs must not share a correlation ID."""
    ids = set()
    for name in ("a", "b"):
        audit = AuditLogger(
            log_path=str(tmp_path / f"{name}.log"),
            logger_name=f"audit-test-{uuid.uuid4().hex}",
        )
        ids.add(audit.run_id)
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()
    assert len(ids) == 2, "run_id must be per-instance"


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


def test_untrusted_content_cannot_forge_an_extra_audit_line(tmp_path):
    """One log() call must always produce exactly one line.

    Logged content is attacker-controlled web text. If a newline survived into
    the log file, a malicious page could append a fully-formed forged entry —
    its own timestamp, tool_name, params and hash — into a log whose purpose
    is tamper-evident forensics.
    """
    logger_name = f"audit-test-{uuid.uuid4().hex}"
    log_file = tmp_path / "research.log"
    audit = AuditLogger(log_path=str(log_file), logger_name=logger_name)
    forged = (
        "harmless looking text\n"
        '2020-01-01 | INFO | search | {"query": "innocent"} | deadbeef | FORGED'
    )
    try:
        audit.log("search", {"query": "q"}, forged)
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1, f"one entry produced {len(lines)} lines: {lines}"
        assert "FORGED" in lines[0], "content should be preserved, just escaped"
        assert "\\n" in lines[0], "the newline should be escaped, not dropped"
        # The forged fragment must not be parseable as a line of its own.
        assert not any(line.startswith("2020-01-01") for line in lines)
    finally:
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()


def test_audit_hash_still_verifies_against_stored_content(tmp_path):
    """Escaping happens before hashing, so the hash covers the stored bytes.

    If the newline were escaped after hashing, a reader could not recompute
    the hash from the log line, and the tamper-evidence would be decorative.
    """
    import json as _json

    logger_name = f"audit-test-{uuid.uuid4().hex}"
    log_file = tmp_path / "research.log"
    audit = AuditLogger(log_path=str(log_file), logger_name=logger_name)
    payload = "line one\nline two\r\nline three"
    params = {"query": "MetroDOT revenue"}
    try:
        audit.log("search", params, payload)
        line = log_file.read_text(encoding="utf-8").strip()
        _ts, _level, tool, params_json, entry_hash, content = line.split(" | ", 5)
        assert tool == "search"
        # Recompute the canonical hash from exactly what the line contains.
        canonical = _json.dumps(
            {
                "timestamp": _ts,
                "tool_name": tool,
                "params": _json.loads(params_json),
                "content": content,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        assert entry_hash == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    finally:
        for h in audit.logger.handlers:
            h.close()
        audit.logger.handlers.clear()


def test_audit_line_ending_escape_is_injective():
    """A real newline and a literal backslash-n must not collapse together.

    Otherwise two different snippets would produce the same stored form, and
    the hash could no longer tell them apart.
    """
    from research.audit import _one_line

    assert _one_line("a\nb") != _one_line("a\\nb")
    assert _one_line("a\r\nb") == "a\\r\\nb"
    assert _one_line("plain text") == "plain text"


def test_search_returns_max_3_cleaned():
    assert clean_text("<p>Hello   <b>world</b></p>") == "Hello world"
    tool = SearchTool(max_results=10)
    assert tool.max_results == 3


@pytest.mark.asyncio
async def test_search_no_api_key_returns_empty(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert await SearchTool().search("MetroDOT revenue") == []


@pytest.mark.asyncio
async def test_extractor_outputs_strict_facts_schema():
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(
            return_value='{"facts": [{"category": "estimated_revenue", "value": "$500M budget", "confidence": "high"}]}'
        ),
    ):
        out = await Extractor().extract("some text", "http://x")
        assert isinstance(out, ExtractorOutput)
        assert len(out.facts) == 1
        assert out.facts[0].category == "estimated_revenue"
        assert out.facts[0].value == "$500M budget"
        assert out.facts[0].confidence == "high"
    # "no fact found" is facts=[], never null.
    with patch("research.extractor.llm_call", new=AsyncMock(return_value="null")):
        out = await Extractor().extract("text", "http://x")
        assert out.facts == []
    with patch("research.extractor.llm_call", new=AsyncMock(return_value="")):
        out = await Extractor().extract("text", "http://x")
        assert out.facts == []


@pytest.mark.asyncio
async def test_extractor_rejects_invalid_json_and_schema():
    with patch("research.extractor.llm_call", new=AsyncMock(return_value="not json")):
        out = await Extractor().extract("text", "http://x")
        assert out.facts == []
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(
            return_value='{"facts": [{"category": "nope", "value": "x", "confidence": "high"}]}'
        ),
    ):
        out = await Extractor().extract("text", "http://x")
        assert out.facts == []
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(
            return_value='```json\n{"facts": [{"category": "sector", "value": "$1M", "confidence": "low"}]}\n```'
        ),
    ):
        out = await Extractor().extract("text", "http://x")
        assert out.facts[0].category == "sector"
    # A bare list is not the {"facts": [...]} object: rejected.
    with patch(
        "research.extractor.llm_call",
        new=AsyncMock(
            return_value='[{"category": "sector", "value": "x", "confidence": "high"}]'
        ),
    ):
        out = await Extractor().extract("text", "http://x")
        assert out.facts == []


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
        return make_output(
            Fact(category="key_partners", value="Partnered with BuildCo", confidence="medium")
        )

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
    # One solid single-source fact -> medium (evidence-based, not field count).
    assert result.confidence == "medium"
    assert result.tender_id == "t1"
    assert result.issuer == "MetroDOT"
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

    async def fake_extract(content, url):
        seen_raw.append(content)
        return make_output(
            Fact(category="key_partners", value="Partnered with BuildCo", confidence="medium")
        )

    runner = make_runner(tender, iteration_cap=1, timeout_seconds=5)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert result.key_partners == ["Partnered with BuildCo"]
    assert result.sector is None
    assert result.confidence == "medium"
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
        return make_output()

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
        return make_output()

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
async def test_runner_missing_tender_raises():
    runner = make_runner(make_tender(), fetch_tender=lambda tender_id: None)
    with pytest.raises(TenderNotFoundError):
        await runner.run("does-not-exist")


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
    r._current_tender_id = "t1"
    r._current_issuer = "MetroDOT"
    # No evidence -> low.
    done = r.finalize()
    assert done.confidence == "low"
    assert done.tender_id == "t1"
    assert done.issuer == "MetroDOT"
    # One solid single-source fact -> medium.
    r.accumulated_facts = {
        "sector": [make_fact("sector", "s", "medium")],
    }
    r.sources = {"u"}
    assert r.finalize().confidence == "medium"
    # Solid facts across >=2 distinct sources -> high.
    r.accumulated_facts["past_projects"] = [make_fact("past_projects", "p", "high")]
    r.sources = {"u", "u2"}
    done = r.finalize()
    assert done.confidence == "high"
    assert sorted(done.sources) == ["u", "u2"]
    assert done.past_projects == ["p"]
    # Thin evidence (only low-confidence facts) is never high.
    r.accumulated_facts = {
        "sector": [make_fact("sector", "s", "low")],
        "estimated_revenue": [make_fact("estimated_revenue", "r", "low")],
        "past_projects": [make_fact("past_projects", "p", "low")],
        "key_partners": [make_fact("key_partners", "k", "low")],
    }
    r.sources = {"u"}
    assert r.finalize().confidence == "low"


def test_finalize_records_sector_disagreement_in_notes():
    r = make_runner()
    r._current_tender_id = "t1"
    r._current_issuer = "MetroDOT"
    r.accumulated_facts = {
        "sector": [
            make_fact("sector", "Alpha", "high"),
            make_fact("sector", "Beta", "high"),
        ],
    }
    r.sources = {"http://a", "http://b"}
    done = r.finalize()
    assert "Alpha" in done.notes and "Beta" in done.notes
    assert done.sector == "Alpha"


def test_runner_build_query():
    r = make_runner()
    assert r._build_query("MetroDOT", "estimated_revenue") == "MetroDOT estimated revenue annual budget"


def test_runner_missing_fields_lists_each_gap():
    r = make_runner()
    assert set(r._missing_fields()) == set(GAP_QUERIES.keys())
    r.accumulated_facts = {"sector": [make_fact("sector", "s")]}
    assert "sector" not in r._missing_fields()


@pytest.mark.asyncio
async def test_runner_round_robins_queries_across_gaps():
    """Each iteration must search a real, distinct per-gap query.

    Regression for the load-bearing bug: _missing_fields() used to return a
    comma-joined string, so GAP_QUERIES.get(gap) was None and every query
    degenerated to "<issuer> None" — searches "succeeded" while researching
    nothing. The round-robin must also keep rotating so an un-fillable first
    gap cannot starve the remaining three.
    """
    tender = make_tender()
    queries = []

    async def fake_search(query):
        queries.append(query)
        return []  # nothing extractable: every gap stays missing

    runner = make_runner(tender, iteration_cap=4, timeout_seconds=30)
    await runner.run(tender.id, search_fn=fake_search)

    assert len(queries) == 4
    # No garbage queries, and every query names a real GAP_QUERIES label.
    assert not any(q.endswith("None") for q in queries)
    assert all(q.startswith("MetroDOT ") for q in queries)
    labels = [q[len("MetroDOT "):] for q in queries]
    assert set(labels) == set(GAP_QUERIES.values())
    # One gap per iteration, no repeats while every gap is still missing.
    assert len(set(labels)) == 4


@pytest.mark.asyncio
async def test_repeated_facts_are_deduplicated_across_iterations():
    """The same fact re-read on a later iteration is stored once.

    Consecutive gap queries hit overlapping pages, so the identical fact comes
    back again. Listing it twice is noise, and for an attacker-controlled page
    it means the payload is restated once per iteration.
    """
    tender = make_tender()

    async def fake_search(query):
        return [SearchResult(url="http://a", content="same page every time")]

    async def fake_extract(content, url):
        return make_output(
            make_fact("sector", "Transport", "medium"),
            make_fact("key_partners", "Stellar Civil", "medium"),
        )

    runner = make_runner(tender, iteration_cap=3, timeout_seconds=30)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)

    assert result.sector == "Transport"
    assert result.key_partners == ["Stellar Civil"]
    assert result.sources == ["http://a"]


@pytest.mark.asyncio
async def test_dedupe_ignores_case_and_surrounding_whitespace():
    """Same fact, different surface form -> still one entry."""
    tender = make_tender()
    seen = {"n": 0}

    async def fake_search(query):
        seen["n"] += 1
        return [SearchResult(url="http://a", content="c")]

    async def fake_extract(content, url):
        value = "Stellar Civil" if seen["n"] == 1 else "  stellar civil  "
        return make_output(make_fact("key_partners", value, "medium"))

    runner = make_runner(tender, iteration_cap=3, timeout_seconds=30)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)

    assert result.key_partners == ["Stellar Civil"]


@pytest.mark.asyncio
async def test_source_that_only_echoes_known_facts_is_not_credited_as_corroboration():
    """A source adding zero new information does not raise confidence.

    compute_confidence() awards "high" for >= 2 distinct source_urls. Without
    the runner's dedupe, an attacker controlling two domains could post the
    same fabricated value on both and manufacture "high" confidence out of one
    invention — the same inflation the function already refuses for four facts
    from a single URL, just spread across URLs. A source is therefore credited
    only when it contributes a fact that is actually new.

    The deliberate cost: two outlets echoing identical text stays "medium".
    """
    tender = make_tender()
    seen = {"n": 0}

    async def fake_search(query):
        seen["n"] += 1
        return [SearchResult(url=f"http://src{seen['n']}", content="c")]

    async def fake_extract(content, url):
        return make_output(make_fact("sector", "Transport", "medium"))

    runner = make_runner(tender, iteration_cap=4, timeout_seconds=30)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)

    assert result.sources == ["http://src1"]
    assert result.confidence == "medium", (
        "an echo-only second source must not be able to manufacture 'high'"
    )


@pytest.mark.asyncio
async def test_second_source_contributing_one_new_fact_is_credited():
    """The complement of the echo case: new information *is* corroborated."""
    tender = make_tender()
    seen = {"n": 0}

    async def fake_search(query):
        seen["n"] += 1
        return [SearchResult(url=f"http://src{seen['n']}", content="c")]

    async def fake_extract(content, url):
        if url == "http://src1":
            return make_output(make_fact("sector", "Transport", "medium"))
        return make_output(
            make_fact("sector", "Transport", "medium"),
            make_fact("estimated_revenue", "$1.2B", "medium"),
        )

    runner = make_runner(tender, iteration_cap=4, timeout_seconds=30)
    result = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)

    assert sorted(result.sources) == ["http://src1", "http://src2"]
    assert result.confidence == "high"
    assert result.estimated_revenue == "$1.2B"


@pytest.mark.asyncio
async def test_runner_skips_gaps_already_filled():
    """A gap backed by a fact is never re-searched."""
    tender = make_tender()
    queries = []

    async def fake_search(query):
        queries.append(query)
        return [SearchResult(url="http://a", content="c")]

    async def fake_extract(content, url):
        # Only ever fills "sector".
        return make_output(make_fact("sector", "Transport", "medium"))

    runner = make_runner(tender, iteration_cap=4, timeout_seconds=30)
    await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)

    labels = [q[len("MetroDOT "):] for q in queries]
    # "sector" is searched once, on iteration 0 when it was genuinely missing,
    # and never re-searched once the fact is in hand. The other three gaps,
    # which stay un-fillable here, each get a turn.
    assert labels.count(GAP_QUERIES["sector"]) == 1
    assert labels[0] == GAP_QUERIES["sector"]
    assert set(labels[1:]) == {
        GAP_QUERIES["estimated_revenue"],
        GAP_QUERIES["past_projects"],
        GAP_QUERIES["key_partners"],
    }


@pytest.mark.asyncio
async def test_issuer_cache_restamps_tender_id():
    from shared.schemas import ProspectResearch

    tender2 = make_tender()
    tender2.id = "t2"  # same issuer MetroDOT, different tender
    cached = ProspectResearch(
        tender_id="t1",
        issuer="MetroDOT",
        sector="s",
        estimated_revenue=None,
        past_projects=[],
        key_partners=[],
        notes="",
        confidence="high",
        sources=["http://c"],
    )

    async def boom_search(query):
        raise AssertionError("cache hit must not search")

    runner = make_runner(tender2, iteration_cap=1, timeout_seconds=5)
    runner.fetch_cached = lambda issuer: cached
    result = await runner.run("t2", search_fn=boom_search)
    assert result.tender_id == "t2"
    assert result.issuer == "MetroDOT"


def test_runner_default_caps():
    r = make_runner(iteration_cap=4, timeout_seconds=30)
    assert r.iteration_cap == 4
    assert r.timeout_seconds == 30


@pytest.mark.asyncio
async def test_runner_state_reset_between_runs():
    async def fake_search(query):
        return [SearchResult(url="http://a", content="x")]

    async def fake_extract(content, url):
        return make_output(make_fact("key_partners", "BuildCo"))

    tender = make_tender()
    runner = make_runner(tender, iteration_cap=1, timeout_seconds=5)
    first = await runner.run(tender.id, search_fn=fake_search, extract_fn=fake_extract)
    assert first.key_partners == ["BuildCo"]
    # A second run with empty extraction must not leak the first run's facts.
    async def empty_extract(content, url):
        return make_output()

    second = await runner.run(tender.id, search_fn=fake_search, extract_fn=empty_extract)
    assert second.key_partners == []
    assert second.confidence == "low"
