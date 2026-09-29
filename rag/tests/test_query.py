from types import SimpleNamespace
from unittest.mock import patch

import pytest

from rag.query import RetrievalError, query_knowledge_base


VALID_ROW = {
    "id": "row-1",
    "content": "Amina knows FastAPI",
    "doc_type": "cv",
    "source_file": "fake_cvs.json",
    "metadata": {"record_id": "CV-1"},
    "similarity": 0.91,
}


class FakeClient:
    def __init__(self, data):
        self.data = data
        self.call = None

    def rpc(self, name, params):
        self.call = (name, params)
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=self.data))


def test_query_passes_bounded_contract_and_keeps_metadata():
    client = FakeClient([VALID_ROW])
    with patch("rag.query.embed_text", return_value=[0.0] * 384):
        result = query_knowledge_base(
            "FastAPI backend",
            top_k=3,
            doc_type="cv",
            similarity_threshold=0.4,
            client=client,
        )
    assert result[0]["metadata"]["record_id"] == "CV-1"
    assert client.call[0] == "match_knowledge_chunks"
    assert client.call[1]["match_count"] == 24
    assert client.call[1]["match_threshold"] == 0.4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"text": "valid", "top_k": 0},
        {"text": "valid", "top_k": 21},
        {"text": "valid", "doc_type": "secret"},
        {"text": "valid", "similarity_threshold": 1.1},
    ],
)
def test_query_rejects_invalid_input(kwargs):
    with pytest.raises(ValueError):
        query_knowledge_base(client=FakeClient([]), **kwargs)


def test_query_rejects_malformed_database_rows():
    client = FakeClient([{"id": "missing-fields"}])
    with patch("rag.query.embed_text", return_value=[0.0] * 384):
        with pytest.raises(RetrievalError, match="malformed"):
            query_knowledge_base("valid query", client=client)


def test_query_rejects_invalid_metadata():
    client = FakeClient([{**VALID_ROW, "metadata": "not-an-object"}])
    with patch("rag.query.embed_text", return_value=[0.0] * 384):
        with pytest.raises(RetrievalError, match="metadata"):
            query_knowledge_base("valid query", client=client)


def test_query_deduplicates_chunks_from_the_same_record():
    duplicate = {**VALID_ROW, "id": "row-2", "content": "another CV-1 passage"}
    distinct = {
        **VALID_ROW,
        "id": "row-3",
        "metadata": {"record_id": "CV-2"},
        "similarity": 0.8,
    }
    client = FakeClient([VALID_ROW, duplicate, distinct])
    with patch("rag.query.embed_text", return_value=[0.0] * 384):
        result = query_knowledge_base("valid query", top_k=2, client=client)
    assert [row["metadata"]["record_id"] for row in result] == ["CV-1", "CV-2"]


def test_query_gives_relevant_olivesoft_projects_a_bounded_priority():
    external = {
        **VALID_ROW,
        "id": "project-1",
        "doc_type": "project",
        "metadata": {"record_id": "PRJ-1"},
        "similarity": 0.80,
    }
    internal = {
        **external,
        "id": "project-2",
        "metadata": {
            "record_id": "olivesoft-1",
            "project_origin": "olivesoft",
            "matching_priority_boost": 0.08,
        },
        "similarity": 0.75,
    }
    with patch("rag.query.embed_text", return_value=[0.0] * 384):
        result = query_knowledge_base(
            "valid query", top_k=2, doc_type="project", client=FakeClient([external, internal])
        )
    assert [row["metadata"]["record_id"] for row in result] == ["olivesoft-1", "PRJ-1"]
    assert result[0]["ranking_score"] == pytest.approx(0.83)
