"""Embed a query and return validated semantic matches from pgvector."""

import math
from typing import Any, Literal

from dotenv import load_dotenv

from rag.embeddings import embed_text
from rag.ranking import ranking_score
from rag.supabase_client import get_supabase

load_dotenv()

DocType = Literal["cv", "project", "tool"]


class RetrievalError(RuntimeError):
    """Raised when vector retrieval fails or returns an invalid contract."""


def query_knowledge_base(
    text: str,
    top_k: int = 5,
    doc_type: DocType | None = None,
    similarity_threshold: float = 0.0,
    client: Any | None = None,
) -> list[dict]:
    """
    Returns a list of top-k matching chunks, each like:
    {
        "id": "...",
        "content": "...",
        "doc_type": "cv",
        "source_file": "ahmed_cv.pdf",
        "similarity": 0.83
    }
    """
    # similarity_threshold is enforced by the vector RPC (match_threshold),
    # not re-checked here. 0.20 is an empirical floor for all-MiniLM-L6-v2
    # on tender text — 0.90 would return zero rows for every query.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("query text must be non-empty")
    if not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    if doc_type not in (None, "cv", "project", "tool"):
        raise ValueError("doc_type must be one of: cv, project, tool")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    query_embedding = embed_text(text)
    db = client or get_supabase()
    # Over-fetch chunks so several passages from one CV/project do not crowd
    # distinct candidate records out of the final top_k response.
    candidate_count = min(top_k * 8, 80)
    try:
        response = db.rpc(
            "match_knowledge_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": candidate_count,
                "filter_doc_type": doc_type,
                "match_threshold": similarity_threshold,
            },
        ).execute()
    except Exception as exc:
        raise RetrievalError("vector search failed") from exc

    data = getattr(response, "data", None)
    if not isinstance(data, list):
        raise RetrievalError("vector search returned an invalid response")

    required = {"id", "content", "doc_type", "source_file", "metadata", "similarity"}
    validated: list[dict] = []
    seen_records: set[tuple[str, str]] = set()
    for row in data:
        if not isinstance(row, dict) or not required.issubset(row):
            raise RetrievalError("vector search returned a malformed result row")
        if row["doc_type"] not in ("cv", "project", "tool"):
            raise RetrievalError("vector search returned an unknown document type")
        if not all(
            isinstance(row[field], str) and row[field].strip()
            for field in ("id", "content", "source_file")
        ):
            raise RetrievalError("vector search returned invalid text fields")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise RetrievalError("vector search returned invalid metadata")
        try:
            similarity = float(row["similarity"])
        except (TypeError, ValueError) as exc:
            raise RetrievalError("vector search returned invalid similarity") from exc
        if not math.isfinite(similarity) or not -1.0 <= similarity <= 1.0:
            raise RetrievalError("vector search returned invalid similarity")
        # Project explicitly onto the QueryResult contract. The RPC lives in a
        # database we do not control here, and QueryResult forbids extras —
        # one new column upstream would otherwise turn every response into
        # a 500 instead of being ignored.
        item = {
            "id": row["id"],
            "content": row["content"],
            "doc_type": row["doc_type"],
            "source_file": row["source_file"],
            "metadata": metadata,
            "similarity": similarity,
            "ranking_score": ranking_score(similarity, row["doc_type"], metadata),
        }
        record_key = str(item["metadata"].get("record_id") or item["id"])
        identity = (item["doc_type"], record_key)
        if identity in seen_records:
            continue
        seen_records.add(identity)
        validated.append(item)
    validated.sort(
        key=lambda item: (item["ranking_score"], item["similarity"]),
        reverse=True,
    )
    return validated[:top_k]


if __name__ == "__main__":
    # quick manual test -- run: python -m rag.query
    sample_tender = "Looking for a team experienced in React and payment gateway integrations for a fintech client."
    results = query_knowledge_base(sample_tender, top_k=5)

    for r in results:
        print(f"[{r['similarity']:.2f}] {r['doc_type']} / {r['source_file']}")
        print(f"    {r['content'][:120]}...")
