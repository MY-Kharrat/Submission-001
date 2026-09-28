import pytest

from rag.evaluation import evaluate_cases, precision_at_k, recall_at_k, reciprocal_rank


def test_metrics():
    retrieved = ["A", "B", "C"]
    relevant = {"B", "D"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert reciprocal_rank(retrieved, relevant) == 0.5


def test_recall_rejects_empty_labels():
    with pytest.raises(ValueError):
        recall_at_k(["A"], set(), 1)


def test_evaluate_cases_uses_record_ids():
    def fake_retriever(query, top_k, doc_type):
        assert query == "FastAPI"
        assert top_k == 2
        assert doc_type == "cv"
        return [
            {"metadata": {"record_id": "CV-1"}},
            {"metadata": {"record_id": "CV-2"}},
        ]

    report = evaluate_cases(
        [{"id": "q1", "query": "FastAPI", "doc_type": "cv", "relevant_ids": ["CV-1"]}],
        k=2,
        retriever=fake_retriever,
    )
    assert report["recall_at_k"] == 1.0
    assert report["precision_at_k"] == 0.5
    assert report["mean_reciprocal_rank"] == 1.0


def test_sql_and_model_dimensions_stay_aligned():
    from pathlib import Path

    sql = (Path(__file__).parents[2] / "sql" / "schema.sql").read_text(encoding="utf-8")
    assert "vector(384)" in sql
    assert "vector(1536)" not in sql
