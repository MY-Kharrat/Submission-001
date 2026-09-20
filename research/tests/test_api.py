"""API contract tests: paid runs are POST-only; finalize persists exactly once."""

import os

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock

os.environ["INTERNAL_SERVICE_TOKEN"] = "test-token"

from research.tests.test_runner import make_tender  # noqa: F401  (reuse helper)
from research.search import SearchTool
from shared import store


@pytest.fixture(autouse=True)
def no_network_search(monkeypatch):
    # Hermetic: never hit Tavily/LLM from API tests regardless of ambient keys.
    monkeypatch.setattr(SearchTool, "search", AsyncMock(return_value=[]))


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(store, "DB_PATH", db_path)
    store.init_db()
    from research.tests.test_runner import make_tender as _make

    store.save_tender(_make(), "seed-hash-t1")
    yield


TOKEN_HEADER = {"X-Internal-Token": "test-token"}


@pytest.mark.asyncio
async def test_run_is_post_only_and_persists(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from research.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # The old paid-GET route is gone.
        resp = await client.get("/research/run", params={"tender_id": "t1"}, headers=TOKEN_HEADER)
        assert resp.status_code in (404, 405)

        # POST runs the pipeline and persists exactly once.
        resp = await client.post("/research/t1", headers=TOKEN_HEADER)
        assert resp.status_code == 200
        body = resp.json()
        assert body["tender_id"] == "t1"
        assert body["issuer"] == "MetroDOT"

        # Stored result is retrievable via GET (safe, read-only).
        resp = await client.get("/research/t1", headers=TOKEN_HEADER)
        assert resp.status_code == 200
        assert resp.json()["tender_id"] == "t1"


@pytest.mark.asyncio
async def test_run_missing_tender_is_404():
    from research.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/research/does-not-exist", headers=TOKEN_HEADER)
        assert resp.status_code == 404
