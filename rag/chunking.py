"""
Splits raw document text into logical chunks before embedding.

Simple strategy: split on blank lines / section breaks first (paragraphs,
bullet points, CV sections). Falls back to a sliding window for very long
paragraphs so no chunk gets too big for the embedding model.
"""

import re

MAX_CHUNK_CHARS = 1000   # keep chunks small and semantically focused
MIN_CHUNK_CHARS = 20     # drop tiny fragments (stray blank lines, headers alone)


def split_into_paragraphs(text: str) -> list[str]:
    """Split on double newlines / bullet breaks -> logical sections."""
    if not isinstance(text, str):
        raise TypeError("document text must be a string")
    raw_parts = re.split(r"\n\s*\n|\n(?=[-*•])", text)
    return [p.strip() for p in raw_parts if p.strip()]


def sliding_window(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = 100) -> list[str]:
    """Fallback for a paragraph that's still too long on its own."""
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must satisfy 0 <= overlap < max_chars")
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_document(text: str) -> list[str]:
    """
    Main entry point: turns a full document's text into a list of chunk
    strings ready to be embedded individually.
    """
    paragraphs = split_into_paragraphs(text)

    chunks = []
    for para in paragraphs:
        if len(para) < MIN_CHUNK_CHARS:
            continue
        if len(para) > MAX_CHUNK_CHARS:
            chunks.extend(sliding_window(para))
        else:
            chunks.append(para)

    return chunks
