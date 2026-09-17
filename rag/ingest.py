"""
Ingestion pipeline: reads files from a source folder (e.g. the repo's
"Cvs dataset" folder), chunks each one, embeds every chunk, and stores it
in the knowledge_chunks table in Supabase.

Idempotent: safe to re-run any time a CV/project file is added or
updated. For each file, old chunks (matched by source_file) are deleted
before new ones are inserted, so the database never accumulates
duplicates or stale content.

Usage:
    python -m rag.ingest --folder "Cvs dataset" --doc_type cv
    python -m rag.ingest --folder "past_projects" --doc_type project
    python -m rag.ingest --folder "tools_docs" --doc_type tool
"""

import argparse
import pathlib

from dotenv import load_dotenv
load_dotenv()

from rag.chunking import chunk_document
from rag.embeddings import embed_batch
from rag.supabase_client import supabase

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


def ingest_folder(folder: str, doc_type: str) -> None:
    folder_path = pathlib.Path(folder)
    files = [f for f in folder_path.iterdir() if f.is_file()]

    print(f"Found {len(files)} files in {folder}")

    for file in files:
        try:
            text = read_file_text(file)
        except Exception as e:
            print(f"  skip {file.name}: {e}")
            continue

        # remove any previously stored chunks for this file first, so
        # re-running ingestion after a CV/project update doesn't create
        # duplicates or leave stale chunks behind
        supabase.table("knowledge_chunks").delete().eq("source_file", file.name).execute()

        chunks = chunk_document(text)
        if not chunks:
            print(f"  {file.name}: no chunks extracted, skipping")
            continue

        # embed and insert in batches
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            embeddings = embed_batch(batch)

            rows = [
                {
                    "content": chunk_text,
                    "doc_type": doc_type,
                    "source_file": file.name,
                    "embedding": embedding,
                }
                for chunk_text, embedding in zip(batch, embeddings)
            ]
            supabase.table("knowledge_chunks").insert(rows).execute()

        print(f"  {file.name}: inserted {len(chunks)} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Path to folder of documents to ingest")
    parser.add_argument("--doc_type", required=True, choices=["cv", "project", "tool"])
    args = parser.parse_args()

    ingest_folder(args.folder, args.doc_type)
    print("Done.")