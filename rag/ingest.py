"""
Ingestion pipeline: reads files from a source folder (e.g. the repo's
"Cvs dataset" folder), chunks each one, embeds every chunk, and stores it
in the knowledge_chunks table in Supabase.

Idempotent and failure-safe: deterministic chunks are upserted first, and
obsolete chunks are removed only after every new batch succeeds.

Usage:
    python -m rag.ingest --folder "Cvs dataset" --doc_type cv
    python -m rag.ingest --folder "past_projects" --doc_type project
    python -m rag.ingest --folder "tools_docs" --doc_type tool
"""

import argparse
import pathlib
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from rag.chunking import chunk_document
from rag.embeddings import embed_batch
from rag.storage import build_rows, make_source_id, replace_source_chunks
from rag.supabase_client import get_supabase

BATCH_SIZE = 20  # how many chunks to embed per API call


def read_file_text(path: pathlib.Path) -> str:
    """Reads a file's text content, handling a few common formats."""
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        import docx
        doc = docx.Document(str(path))
        return "\n\n".join(p.text for p in doc.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix}")


SUPPORTED_EXTENSIONS = {".txt", ".pdf", ".docx"}


def ingest_folder(folder: str, doc_type: Literal["cv", "project", "tool"]) -> int:
    folder_path = pathlib.Path(folder)
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Source folder not found: {folder_path}")
    if doc_type not in ("cv", "project", "tool"):
        raise ValueError("doc_type must be one of: cv, project, tool")
    files = sorted(
        f for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    print(f"Found {len(files)} files in {folder}")
    ingested = 0
    db = get_supabase()

    for file in files:
        try:
            text = read_file_text(file)
        except Exception as e:
            print(f"  skip {file.name}: {e}")
            continue

        chunks = chunk_document(text)
        if not chunks:
            print(f"  {file.name}: no chunks extracted, skipping")
            continue

        # Finish every embedding before mutating this source in the database.
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            all_embeddings.extend(embed_batch(batch))

        source_id = make_source_id(file, doc_type)
        rows = build_rows(
            [{"content": text, "metadata": {"format": file.suffix.lower()}} for text in chunks],
            all_embeddings,
            doc_type=doc_type,
            source_id=source_id,
            source_file=file.name,
        )
        replace_source_chunks(rows, source_id=source_id, client=db)
        ingested += len(rows)
        print(f"  {file.name}: upserted {len(rows)} chunks")
    return ingested


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to folder of documents to ingest")
    parser.add_argument("--doc_type", required=True, choices=["cv", "project", "tool"])
    args = parser.parse_args()

    ingest_folder(args.folder, args.doc_type)
    print("Done.")
