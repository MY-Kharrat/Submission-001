import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


os.environ["INTERNAL_SERVICE_TOKEN"] = "test-token"

from shared import store
from detection.main import app
from detection.normalize import (
    RAW_TEXT_MAX_CHARS,
    ValidationError,
    extract_json_payload,
    normalize_tender,
    tag_sector,
    validate_fields,
)

VALID_DATA = {
    "title": "Penetration Testing Engagement",
    "issuer": "ACME Security",
    "deadline": "2026-12-01",
    "raw_text": "Perform a thorough penetration testing engagement on primary web infrastructure.",
    "requirements": ["OWASP top 10 testing", "Executive summary report"],
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
        tag_sector("penetration testing engagement required", "Security Audit")
        == "Security Assessment & Penetration Testing"
    )
    assert (
        tag_sector("building an artificial intelligence machine learning model", "AI Project")
        == "Data/AI Integration Consulting"
    )


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
            assert body["title"] == "Penetration Testing Engagement"
            assert body["issuer"] == "ACME Security"


@pytest.mark.asyncio
async def test_ingest_invalid_date_returns_422():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        bad_data = {**VALID_DATA, "deadline": "2026-99-99"}
        resp = await client.post("/tenders/ingest", json=bad_data, headers=TOKEN_HEADER)
        assert resp.status_code == 422


@pytest.mark.asyncio
async def test_seed_tenders_endpoint():
    with patch("detection.normalize.llm_call", new_callable=AsyncMock, return_value="Security Assessment & Penetration Testing"):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/tenders/seed", headers=TOKEN_HEADER)
            assert resp.status_code == 200
            body = resp.json()
            assert body["seeded"] >= 15
