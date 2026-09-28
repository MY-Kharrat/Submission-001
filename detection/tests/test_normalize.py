import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


os.environ["INTERNAL_SERVICE_TOKEN"] = "test-token"

from shared import store
from detection.main import app
from detection.normalize import (
    RAW_TEXT_MAX_CHARS,
    SECTOR_KEYPHRASES,
    ValidationError,
    extract_json_payload,
    normalize_tender,
    tag_sector,
    validate_fields,
)
from shared.schemas import CAPABILITY_TAXONOMY

VALID_DATA = {
    "title": "MuleSoft Integration Across HR Systems",
    "issuer": "ACME Group",
    "deadline": "2026-12-01",
    "raw_text": "Deliver a MuleSoft integration layer unifying our HR and payroll systems.",
    "requirements": ["MuleSoft integration layer", "Versioned API handover"],
    "source": "simulated_feed",
}
TOKEN_HEADER = {"X-Internal-Token": "test-token"}


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Provides isolated SQLite database per test function."""
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)
    store.init_db()
    yield


# --- Upfront Field & Date Validation Tests ---

def test_missing_required_field():
    for field in ("title", "issuer", "deadline", "raw_text"):
        data = {**VALID_DATA, field: ""}
        with pytest.raises(ValidationError, match=f"Missing required field: {field}"):
            validate_fields(data)


def test_whitespace_required_field():
    for field in ("title", "issuer", "deadline", "raw_text"):
        data = {**VALID_DATA, field: "   "}
        with pytest.raises(ValidationError, match=f"Missing required field: {field}"):
            validate_fields(data)


def test_invalid_deadline_format():
    data = {**VALID_DATA, "deadline": "not-a-valid-date"}
    with pytest.raises(ValidationError, match="Invalid deadline date format"):
        validate_fields(data)


@pytest.mark.asyncio
async def test_invalid_date_rejects_upfront_without_llm():
    """Verifies malformed date rejects before executing any paid LLM calls."""
    data = {**VALID_DATA, "deadline": "invalid-date", "requirements": []}
    with patch("detection.normalize.llm_call", new_callable=AsyncMock) as mock_llm:
        with pytest.raises(ValidationError, match="Invalid deadline date format"):
            await normalize_tender(data)
        assert mock_llm.call_count == 0


def test_raw_text_over_cap():
    data = {**VALID_DATA, "raw_text": "x" * (RAW_TEXT_MAX_CHARS + 1)}
    with pytest.raises(ValidationError, match="exceeds"):
        validate_fields(data)


def test_raw_text_at_cap_accepted():
    data = {**VALID_DATA, "raw_text": "x" * RAW_TEXT_MAX_CHARS}
    validate_fields(data)


# --- Idempotent Deduplication Tests ---

@pytest.mark.asyncio
async def test_duplicate_returns_existing_without_re_llm():
    """Verifies duplicate ingest returns existing record idempotently without re-spending LLM tokens."""
    with patch("detection.normalize.llm_call", new_callable=AsyncMock) as mock_llm:
        tender1, is_existing1 = await normalize_tender(VALID_DATA.copy())
        assert not is_existing1

        tender2, is_existing2 = await normalize_tender(VALID_DATA.copy())
        assert is_existing2
        assert tender2.id == tender1.id
        assert mock_llm.call_count == 0


# --- Sector Tagging Keyphrase & LLM Fallback Tests ---

def test_keyphrase_sector_match():
    assert (
        tag_sector("mulesoft integration with sap successfactors and payroll", "Global HRIS Integration")
        == "Data Integration"
    )
    assert (
        tag_sector("demand forecasting using machine learning on order history", "Forecasting")
        == "AI Development"
    )
    assert (
        tag_sector("power bi reporting layer with a semantic model and kpi definitions", "Analytics")
        == "BI & Dashboarding"
    )
    assert (
        tag_sector("salesforce service cloud case management rollout", "CRM Overhaul")
        == "Salesforce Ecosystem"
    )
    assert (
        tag_sector("kafka streaming into a databricks lakehouse", "Streaming")
        == "Data Platform"
    )


def test_keyphrase_sector_n8n_keyed_to_data_integration():
    """n8n is OliveSoft's certified-partner automation tool, so it must land on Data Integration."""
    assert (
        tag_sector("n8n workflow automation connecting erp to crm", "Automation")
        == "Data Integration"
    )


def test_keyphrase_sector_tie_break_follows_taxonomy_order():
    """Equal scores resolve to the earlier category in CAPABILITY_TAXONOMY, deterministically."""
    # One keyphrase from each of the two earliest lines, so the scores tie.
    tied_text = "mulesoft and machine learning"
    assert (
        tag_sector(tied_text, "Notice")
        == CAPABILITY_TAXONOMY[0]
    )
    # Repeat calls agree; the winner is a function of taxonomy order, not dict iteration luck.
    assert len({tag_sector(tied_text, "Notice") for _ in range(10)}) == 1


def test_sector_keyphrases_cover_exact_taxonomy():
    """Keyphrase map and taxonomy must not drift: the module-level guard enforces it at import."""
    assert set(SECTOR_KEYPHRASES) == set(CAPABILITY_TAXONOMY)
    assert CAPABILITY_TAXONOMY == [
        "Data Integration",
        "AI Development",
        "BI & Dashboarding",
        "Salesforce Ecosystem",
        "Data Platform",
    ]


def test_keyphrase_sector_no_match_returns_unknown():
    assert tag_sector("generic ambiguous text without keyphrases", "Notice") == "unknown"


# --- Markdown Fence Proof JSON Extraction Tests ---

def test_extract_json_payload_markdown_fences():
    fenced_input = '```json\n["Req A", "Req B"]\n```'
    assert extract_json_payload(fenced_input) == '["Req A", "Req B"]'

    preamble_input = 'Here are the requirements:\n```json\n["Req 1"]\n```'
    assert extract_json_payload(preamble_input) == '["Req 1"]'


# --- Authentication Security Tests ---

@pytest.mark.asyncio
async def test_missing_token_returns_401():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/tenders")
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid or missing X-Internal-Token"


@pytest.mark.asyncio
async def test_wrong_token_returns_401():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/tenders", headers={"X-Internal-Token": "wrong-secret"})
        assert resp.status_code == 401


@pytest.mark.asyncio
async def test_valid_token_works():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/tenders", headers=TOKEN_HEADER)
        assert resp.status_code == 200


# --- API Integration & Seed Tests ---

@pytest.mark.asyncio
async def test_ingest_endpoint():
    with patch("detection.normalize.llm_call", new_callable=AsyncMock):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/tenders/ingest", json=VALID_DATA, headers=TOKEN_HEADER)
            assert resp.status_code == 200
            body = resp.json()
            assert body["title"] == "MuleSoft Integration Across HR Systems"
            assert body["issuer"] == "ACME Group"
            assert body["sector"] == "Data Integration"


@pytest.mark.asyncio
async def test_ingest_invalid_date_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad_data = {**VALID_DATA, "deadline": "2026-99-99"}
        resp = await client.post("/tenders/ingest", json=bad_data, headers=TOKEN_HEADER)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_seed_tenders_endpoint():
    with patch(
        "detection.normalize.llm_call",
        new_callable=AsyncMock,
        return_value="Data Integration",
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            assert resp.status_code == 200
            body = resp.json()
            assert body["seeded"] >= 15
            assert all(t["status"] == "success" for t in body["tenders"])


@pytest.mark.asyncio
async def test_seed_feed_reports_dedupe_on_repost():
    """The feed ships a re-announced tender; seeding twice must not create a second row."""
    with patch(
        "detection.normalize.llm_call",
        new_callable=AsyncMock,
        return_value="Data Integration",
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            first = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            second = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            assert first.status_code == second.status_code == 200

            before = len((await client.get("/tenders", headers=TOKEN_HEADER)).json())
            after = len((await client.get("/tenders", headers=TOKEN_HEADER)).json())
            assert before == after

            repost = next(t for t in second.json()["tenders"] if t["file"] == "tender_015.json")
            assert repost["existing"] is True


@pytest.mark.asyncio
async def test_seed_feed_exercises_requirement_extraction_fallback():
    """tender_013 ships no requirements[] key, so the LLM extraction path must fill it."""
    with patch(
        "detection.normalize.llm_call",
        new_callable=AsyncMock,
        return_value='["Integrate telematics with maintenance and fuel systems"]',
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            assert resp.status_code == 200
            extracted = next(
                t for t in resp.json()["tenders"] if t["file"] == "tender_013.json"
            )
            assert extracted["status"] == "success"

            tender = (await client.get(
                f"/tenders/{extracted['id']}", headers=TOKEN_HEADER
            )).json()
            assert tender["requirements"] == [
                "Integrate telematics with maintenance and fuel systems"
            ]


@pytest.mark.asyncio
async def test_seed_feed_sector_fallback_only_for_unclassifiable_tender():
    """Only tender_014 should miss the keyphrase pass and reach the LLM classifier."""
    with patch(
        "detection.normalize.llm_call",
        new_callable=AsyncMock,
        return_value="Data Integration",
    ) as mock_llm:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            assert resp.status_code == 200

            unclassifiable = next(
                t for t in resp.json()["tenders"] if t["file"] == "tender_014.json"
            )
            assert unclassifiable["status"] == "success"
            # Keyphrase tagging returns "unknown" for this one, so the payload
            # carried no sector and only the LLM fallback can fill it in.
            classified = (await client.get(
                f"/tenders/{unclassifiable['id']}", headers=TOKEN_HEADER
            )).json()
            assert classified["sector"] == "Data Integration"

            # Exactly two paid calls for the whole feed: the classifier for
            # tender_014, and requirement extraction for requirements-less
            # tender_013. Every other file is handled deterministically.
            assert mock_llm.call_count == 2
