"""Labelled retrieval benchmark and standard information-retrieval metrics."""

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Callable

from rag.query import query_knowledge_base


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    return len(set(retrieved[:k]) & relevant) / k


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if not relevant:
        raise ValueError("relevant set must not be empty")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, record_id in enumerate(retrieved, start=1):
        if record_id in relevant:
            return 1.0 / rank
    return 0.0


def evaluate_cases(
    cases: list[dict],
    *,
    k: int = 5,
    retriever: Callable[..., list[dict]] = query_knowledge_base,
) -> dict:
    if not cases:
        raise ValueError("benchmark must contain at least one case")

    case_results: list[dict] = []
    for case in cases:
        relevant = set(case["relevant_ids"])
        if not relevant:
            raise ValueError(f"case {case.get('id', 'unknown')!r} has no relevant IDs")
        matches = retriever(
            case["query"],
            top_k=k,
            doc_type=case.get("doc_type"),
        )
        retrieved = [
            str(match.get("metadata", {}).get("record_id", ""))
            for match in matches
            if match.get("metadata", {}).get("record_id")
        ]
        case_results.append(
            {
                "id": case.get("id"),
                "retrieved_ids": retrieved,
                "precision_at_k": precision_at_k(retrieved, relevant, k),
                "recall_at_k": recall_at_k(retrieved, relevant, k),
                "reciprocal_rank": reciprocal_rank(retrieved, relevant),
            }
        )

    return {
        "case_count": len(case_results),
        "k": k,
        "precision_at_k": mean(item["precision_at_k"] for item in case_results),
        "recall_at_k": mean(item["recall_at_k"] for item in case_results),
        "mean_reciprocal_rank": mean(item["reciprocal_rank"] for item in case_results),
        "cases": case_results,
    }


def load_cases(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark file must contain a JSON list")
    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OliveSoft semantic retrieval")
    parser.add_argument(
        "--dataset",
        default=str(Path(__file__).with_name("eval_dataset.json")),
        help="Path to labelled evaluation JSON",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--min-recall",
        type=float,
        default=0.70,
        help="Exit non-zero when Recall@K is below this value",
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_recall <= 1.0:
        parser.error("--min-recall must be between 0 and 1")

    report = evaluate_cases(load_cases(Path(args.dataset)), k=args.k)
    print(json.dumps(report, indent=2))
    return 0 if report["recall_at_k"] >= args.min_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
