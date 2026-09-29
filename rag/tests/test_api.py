import os
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

os.environ["INTERNAL_SERVICE_TOKEN"] = "test-token"

from rag.main import app


RESULT = {
    "id": "row-1",
    "content": "Amina knows FastAPI",
    "doc_type": "cv",
    "source_file": "fake_cvs.json",
    "metadata": {"record_id": "CV-1", "skills": ["FastAPI"]},
    "similarity": 0.91,
    "ranking_score": 0.91,
}


@pytest.mark.asyncio
async def test_health_is_public_and_does_not_initialize_providers():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_query_requires_internal_token():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/query", json={"text": "FastAPI engineer"})
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_returns_record_metadata():
    with patch("rag.main.query_knowledge_base", return_value=[RESULT]):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/query",
                json={"text": "FastAPI engineer", "doc_type": "cv"},
                headers={"X-Internal-Token": "test-token"},
            )
    assert response.status_code == 200
    assert response.json()[0]["metadata"]["record_id"] == "CV-1"


@pytest.mark.asyncio
async def test_query_rejects_bad_limits_and_extra_fields():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/query",
            json={"text": "valid query", "top_k": 100, "unexpected": True},
            headers={"X-Internal-Token": "test-token"},
        )
    assert response.status_code == 422
