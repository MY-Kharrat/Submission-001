import json

import pytest

from rag.evaluate_local import LocalRetriever, load_corpus


def test_local_retriever_filters_and_deduplicates_records():
    documents = [
        {"content": "a", "doc_type": "cv", "metadata": {"record_id": "CV-1"}},
        {"content": "b", "doc_type": "cv", "metadata": {"record_id": "CV-1"}},
        {"content": "c", "doc_type": "project", "metadata": {"record_id": "PRJ-1"}},
    ]
    retriever = LocalRetriever(
        documents,
        [[1.0, 0.0], [0.7, 0.3], [0.9, 0.1]],
        query_embedder=lambda _text: [1.0, 0.0],
    )
    results = retriever("backend", top_k=5, doc_type="cv")
    assert [item["metadata"]["record_id"] for item in results] == ["CV-1"]
    assert results[0]["similarity"] == pytest.approx(1.0)


def test_local_retriever_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        LocalRetriever([], [[1.0]], query_embedder=lambda _text: [1.0])


def test_load_corpus_rejects_non_list(tmp_path):
    cvs = tmp_path / "cvs.json"
    projects = tmp_path / "projects.json"
    cvs.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    projects.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON list"):
        load_corpus(cvs, projects)
