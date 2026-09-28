"""Lazy, validated access to the local sentence-transformers model."""

from functools import lru_cache
import os
from typing import Any

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384  # Must match sql/schema.sql.


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or returns bad data."""


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load the model on first use instead of making app import download it."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise EmbeddingError(
            "sentence-transformers is not installed; run pip install -r requirements.txt"
        ) from exc

    model_name = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:  # pragma: no cover - provider/cache dependent
        raise EmbeddingError(f"Unable to load embedding model {model_name!r}") from exc


def _validate_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("embedding input must be a non-empty string")
    return text.strip()


def _validate_vectors(vectors: list[list[float]], expected_count: int) -> list[list[float]]:
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"embedding model returned {len(vectors)} vectors for {expected_count} inputs"
        )
    for vector in vectors:
        if len(vector) != EMBEDDING_DIMENSIONS:
            raise EmbeddingError(
                f"embedding dimension mismatch: expected {EMBEDDING_DIMENSIONS}, got {len(vector)}"
            )
    return vectors


def embed_text(text: str) -> list[float]:
    """Embed one string as a normalized 384-dimensional vector."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a non-empty batch and enforce the database vector contract."""
    if not texts:
        return []
    cleaned = [_validate_text(text) for text in texts]
    try:
        encoded = get_model().encode(
            cleaned,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = encoded.tolist()
    except EmbeddingError:
        raise
    except Exception as exc:
        raise EmbeddingError("embedding generation failed") from exc
    return _validate_vectors(vectors, len(cleaned))
