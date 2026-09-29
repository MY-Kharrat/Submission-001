"""Reproducible in-memory RAG benchmark for the bundled synthetic corpus.

This uses the exact production chunkers and embedding model but does not need
Supabase. It is intended for judging, CI on a model-enabled runner, and local
regression checks before the vector database is provisioned.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from rag.embeddings import embed_batch, embed_text
from rag.evaluation import evaluate_cases, load_cases
from rag.ranking import ranking_score
from rag.structured_chunking import (
    chunk_cv_record,
    chunk_olivesoft_project_record,
    chunk_project_record,
    chunk_tool_record,
)


def load_corpus(
    cv_path: Path,
    project_path: Path,
    tool_path: Path | None = None,
    olivesoft_project_path: Path | None = None,
) -> list[dict]:
    documents: list[dict] = []
    sources = [
        (cv_path, "cv", chunk_cv_record),
        (project_path, "project", chunk_project_record),
    ]
    if tool_path is not None:
        sources.append((tool_path, "tool", chunk_tool_record))
    if olivesoft_project_path is not None:
        sources.append(
            (olivesoft_project_path, "project", chunk_olivesoft_project_record)
        )
    for path, doc_type, chunker in sources:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError(f"{path} must contain a JSON list")
        for record in records:
            for chunk in chunker(record):
                documents.append(
                    {
                        "content": chunk["content"],
                        "doc_type": doc_type,
                        "metadata": {
                            "record_id": chunk["record_id"],
                            **chunk["record_meta"],
                        },
                    }
                )
    if not documents:
        raise ValueError("benchmark corpus contains no chunks")
    return documents


class LocalRetriever:
    def __init__(
        self,
        documents: list[dict],
        embeddings: list[list[float]],
        *,
        query_embedder: Callable[[str], list[float]] = embed_text,
    ):
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have equal length")
        self.documents = documents
        self.embeddings = embeddings
        self.query_embedder = query_embedder

    def __call__(self, text: str, *, top_k: int, doc_type: str | None = None) -> list[dict]:
        query = self.query_embedder(text)
        best_by_record: dict[str, dict] = {}
        for document, vector in zip(self.documents, self.embeddings):
            if doc_type and document["doc_type"] != doc_type:
                continue
            similarity = sum(a * b for a, b in zip(query, vector))
            score = ranking_score(similarity, document["doc_type"], document["metadata"])
            record_id = str(document["metadata"]["record_id"])
            previous = best_by_record.get(record_id)
            if previous is None or score > previous["ranking_score"]:
                best_by_record[record_id] = {
                    **document,
                    "similarity": similarity,
                    "ranking_score": score,
                }
        return sorted(
            best_by_record.values(), key=lambda item: item["ranking_score"], reverse=True
        )[:top_k]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark bundled CV/project/capability semantic retrieval"
    )
    parser.add_argument("--cvs", type=Path, default=Path("Cvs dataset/fake_cvs.json"))
    parser.add_argument("--projects", type=Path, default=Path("Cvs dataset/fake_projects.json"))
    parser.add_argument("--tools", type=Path, default=Path("Cvs dataset/fake_tools.json"))
    parser.add_argument(
        "--olivesoft-projects",
        type=Path,
        default=Path("Cvs dataset/olivesoft_projects.json"),
    )
    parser.add_argument("--dataset", type=Path, default=Path("rag/eval_dataset.json"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.70)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    if not 0.0 <= args.min_recall <= 1.0:
        parser.error("--min-recall must be between 0 and 1")

    documents = load_corpus(
        args.cvs, args.projects, args.tools, args.olivesoft_projects
    )
    embeddings = embed_batch([document["content"] for document in documents])
    retriever = LocalRetriever(documents, embeddings)
    report = evaluate_cases(load_cases(args.dataset), k=args.k, retriever=retriever)
    report["corpus_chunks"] = len(documents)
    print(json.dumps(report, indent=2))
    return 0 if report["recall_at_k"] >= args.min_recall else 1


if __name__ == "__main__":
    raise SystemExit(main())
