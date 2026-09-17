"""
Ingests structured JSON data (a list of CV records or project records)
into Supabase, using natural-language chunking suited to that structure.

Idempotent: re-running on the same file first deletes existing rows for
that source_file, so updates don't create duplicates.

Usage:
    python -m rag.ingest_json --file "Cvs dataset/cvs.json" --doc_type cv
    python -m rag.ingest_json --file "projects.json" --doc_type project
"""

import argparse
import json
import pathlib

from dotenv import load_dotenv
load_dotenv()

from rag.structured_chunking import chunk_cv_record, chunk_project_record
from rag.embeddings import embed_batch
from rag.supabase_client import supabase

BATCH_SIZE = 20


def ingest_json_file(file_path: str, doc_type: str) -> None:
    path = pathlib.Path(file_path)
    records = json.loads(path.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} records from {path.name}")

    # remove any previously stored chunks for this file -- keeps re-runs
    # after data updates from creating duplicates
    supabase.table("knowledge_chunks").delete().eq("source_file", path.name).execute()

    chunker = chunk_cv_record if doc_type == "cv" else chunk_project_record

    all_chunks = []
    for record in records:
        all_chunks.extend(chunker(record))

    print(f"Generated {len(all_chunks)} chunks, embedding and inserting...")

    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        embeddings = embed_batch(texts)

        rows = [
            {
                "content": c["content"],
                "doc_type": doc_type,
                "source_file": path.name,
                "metadata": {"record_id": c["record_id"], **c["record_meta"]},
                "embedding": embedding,
            }
            for c, embedding in zip(batch, embeddings)
        ]
        supabase.table("knowledge_chunks").insert(rows).execute()
        print(f"  inserted {i + len(batch)}/{len(all_chunks)}")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the JSON file to ingest")
    parser.add_argument("--doc_type", required=True, choices=["cv", "project"])
    args = parser.parse_args()

    ingest_json_file(args.file, args.doc_type)