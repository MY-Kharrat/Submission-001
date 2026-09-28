"""
Ingests structured JSON data (CV, project, OliveSoft project catalogue, or
capability/tool records)
into Supabase, using natural-language chunking suited to that structure.

Idempotent and failure-safe: deterministic chunks are upserted first, and
obsolete chunks are removed only after every new batch succeeds.

Usage:
    python -m rag.ingest_json --file "Cvs dataset/fake_cvs.json" --doc_type cv
    python -m rag.ingest_json --file "Cvs dataset/fake_projects.json" --doc_type project
    python -m rag.ingest_json --file "Cvs dataset/olivesoft_projects.json" --doc_type project --record-schema olivesoft
    python -m rag.ingest_json --file "Cvs dataset/fake_tools.json" --doc_type tool
"""

import argparse
import json
import pathlib
from typing import Literal

from dotenv import load_dotenv
load_dotenv()

from rag.structured_chunking import (
    chunk_cv_record,
    chunk_olivesoft_project_record,
    chunk_project_record,
    chunk_tool_record,
)
from rag.embeddings import embed_batch
from rag.storage import build_rows, make_source_id, replace_source_chunks
from rag.supabase_client import get_supabase

BATCH_SIZE = 20


def ingest_json_file(
    file_path: str,
    doc_type: Literal["cv", "project", "tool"],
    *,
    source_id: str | None = None,
    record_schema: Literal["standard", "olivesoft"] = "standard",
) -> int:
    path = pathlib.Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON source not found: {path}")
    if doc_type not in ("cv", "project", "tool"):
        raise ValueError("doc_type must be cv, project, or tool")
    if record_schema not in ("standard", "olivesoft"):
        raise ValueError("record_schema must be standard or olivesoft")
    if record_schema == "olivesoft" and doc_type != "project":
        raise ValueError("the olivesoft record schema is only valid for project documents")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError("JSON source must contain a non-empty list of records")
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("every JSON record must be an object")
    print(f"Loaded {len(records)} records from {path.name}")

    chunker = {
        "cv": chunk_cv_record,
        "project": (
            chunk_olivesoft_project_record
            if record_schema == "olivesoft"
            else chunk_project_record
        ),
        "tool": chunk_tool_record,
    }[doc_type]

    all_chunks: list[dict] = []
    for record in records:
        try:
            all_chunks.extend(chunker(record))
        except (KeyError, TypeError, ValueError) as exc:
            record_id = (
                record.get("cv_id")
                or record.get("project_id")
                or record.get("tool_id")
                or record.get("id")
                or "unknown"
            )
            raise ValueError(f"Invalid {doc_type} record {record_id!r}: {exc}") from exc

    print(f"Generated {len(all_chunks)} chunks, embedding and inserting...")

    # Complete all model work before touching the current database version.
    all_embeddings: list[list[float]] = []
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i:i + BATCH_SIZE]
        texts = [c["content"] for c in batch]
        all_embeddings.extend(embed_batch(texts))

    normalized_chunks = [
        {
            "content": chunk["content"],
            "metadata": {"record_id": chunk["record_id"], **chunk["record_meta"]},
        }
        for chunk in all_chunks
    ]
    logical_source = make_source_id(path, doc_type, source_id)
    rows = build_rows(
        normalized_chunks,
        all_embeddings,
        doc_type=doc_type,
        source_id=logical_source,
        source_file=path.name,
    )
    replace_source_chunks(rows, source_id=logical_source, client=get_supabase())

    print(f"Done. Upserted {len(rows)} chunks for {logical_source}.")
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to the JSON file to ingest")
    parser.add_argument("--doc_type", required=True, choices=["cv", "project", "tool"])
    parser.add_argument("--source-id", help="Stable logical source name (optional)")
    parser.add_argument(
        "--record-schema",
        choices=["standard", "olivesoft"],
        default="standard",
        help="Structured project schema; use olivesoft for the attributed catalogue",
    )
    args = parser.parse_args()

    ingest_json_file(
        args.file,
        args.doc_type,
        source_id=args.source_id,
        record_schema=args.record_schema,
    )
