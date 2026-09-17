"""
Thin wrapper around the embedding model. Runs fully locally using
sentence-transformers -- no API key, no credits, no internet call needed
after the model is first downloaded. Free forever.

Swap the implementation here if you switch to a paid provider later --
nothing else in the rag/ module needs to change.
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, good enough for MVP retrieval
EMBEDDING_DIMENSIONS = 384              # must match sql/schema.sql

_model = SentenceTransformer(EMBEDDING_MODEL)


def embed_text(text: str) -> list[float]:
    """Embed a single string, return its vector."""
    return _model.encode(text).tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many strings at once -- much faster than one-by-one."""
    return _model.encode(texts).tolist()