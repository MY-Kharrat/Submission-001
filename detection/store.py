"""
SQLite persistence for tenders: WAL mode with a UNIQUE dedup_hash column
to keep duplicate ingests idempotent at the storage layer.
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

from shared.schemas import Tender

DB_PATH = Path(__file__).parent / "tenders.db"


def _row_to_tender(row: sqlite3.Row) -> Tender:
    d = dict(row)
    d["requirements"] = json.loads(d["requirements"])
    return Tender(**d)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tenders (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            issuer TEXT NOT NULL,
            sector TEXT NOT NULL,
            requirements TEXT NOT NULL,
            deadline TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            source TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            status TEXT NOT NULL,
            dedup_hash TEXT UNIQUE
        )
    """)
    conn.commit()
    conn.close()


def save_tender(tender: Tender, dedup_hash: str) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO tenders
           (id, title, issuer, sector, requirements, deadline, raw_text,
            source, detected_at, status, dedup_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            tender.id, tender.title, tender.issuer, tender.sector,
            json.dumps(tender.requirements), tender.deadline.isoformat(),
            tender.raw_text, tender.source, tender.detected_at.isoformat(),
            tender.status, dedup_hash,
        ),
    )
    conn.commit()
    conn.close()


def get_tender(tender_id: str) -> Optional[Tender]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tenders WHERE id = ?", (tender_id,)).fetchone()
    conn.close()
    return _row_to_tender(row) if row else None


def find_by_hash(dedup_hash: str) -> Optional[Tender]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM tenders WHERE dedup_hash = ?", (dedup_hash,)).fetchone()
    conn.close()
    return _row_to_tender(row) if row else None


def list_tenders(status: Optional[str] = None) -> list[Tender]:
    conn = _get_conn()
    if status:
        rows = conn.execute("SELECT * FROM tenders WHERE status = ?", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM tenders").fetchall()
    conn.close()
    return [_row_to_tender(r) for r in rows]
