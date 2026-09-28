"""Transparent, bounded ranking policy for RAG results."""

from __future__ import annotations

from typing import Any


MAX_PRIORITY_BOOST = 0.10


def ranking_score(similarity: float, doc_type: str, metadata: dict[str, Any]) -> float:
    """Return semantic similarity plus an explicitly bounded business priority.

    OliveSoft-owned projects are preferred only after semantic retrieval and by
    at most ten percentage points. This prevents an unrelated internal record
    from outranking a strongly relevant result while making the policy visible
    and reproducible.
    """
    boost = 0.0
    # TODO: modify ranking more parms (CV banks, client portfolios, tool)
    ## For client portfolio, if it's regular client, we classify him in a different class
    if doc_type == "project" and metadata.get("project_origin") == "olivesoft":
        raw_boost = metadata.get("matching_priority_boost", 0.0)
        if isinstance(raw_boost, (int, float)) and not isinstance(raw_boost, bool):
            boost = max(0.0, min(float(raw_boost), MAX_PRIORITY_BOOST))
    return max(-1.0, min(1.0, similarity + boost))
