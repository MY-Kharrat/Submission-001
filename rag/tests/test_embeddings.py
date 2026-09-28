from unittest.mock import patch

import pytest

from rag.embeddings import EMBEDDING_DIMENSIONS, EmbeddingError, embed_batch, embed_text


class FakeEncoded:
    def __init__(self, vectors):
        self.vectors = vectors

    def tolist(self):
        return self.vectors


class FakeModel:
    def __init__(self, dimensions=EMBEDDING_DIMENSIONS):
        self.dimensions = dimensions
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return FakeEncoded([[0.1] * self.dimensions for _ in texts])


def test_embed_batch_normalizes_and_preserves_count():
    model = FakeModel()
    with patch("rag.embeddings.get_model", return_value=model):
        vectors = embed_batch([" first ", "second"])
    assert len(vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSIONS for vector in vectors)
    assert model.calls[0][0] == ["first", "second"]
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_embed_text_rejects_blank_input():
    with pytest.raises(ValueError, match="non-empty"):
        embed_text("   ")


def test_dimension_mismatch_fails_closed():
    with patch("rag.embeddings.get_model", return_value=FakeModel(dimensions=10)):
        with pytest.raises(EmbeddingError, match="dimension mismatch"):
            embed_text("query")
