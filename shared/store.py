"""
SQLite persistence for tenders and prospect research: WAL mode with a UNIQUE
dedup_hash column to keep duplicate ingests idempotent at the storage layer.
"""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from shared.schemas import ProspectResearch, Tender

DB_PATH = Path(__file__).parent / "sqlite.db"


def _row_to_tender(row: sqlite3.Row) -> Tender:
    d = dict(row)
    d["requirements"] = json.loads(d["requirements"])
    if isinstance(d.get("deadline"), str):
        d["deadline"] = date.fromisoformat(d["deadline"])
    if isinstance(d.get("detected_at"), str):
        d["detected_at"] = datetime.fromisoformat(d["detected_at"])
    d.pop("dedup_hash", None)
    return Tender(**d)


def _row_to_research(row: sqlite3.Row) -> ProspectResearch:
    d = dict(row)
    d.pop("tender_id", None)
    d["past_projects"] = json.loads(d["past_projects"])
    d["key_partners"] = json.loads(d["key_partners"])
    d["sources"] = json.loads(d["sources"])
    return ProspectResearch(**d)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS research (
            tender_id TEXT PRIMARY KEY REFERENCES tenders(id) ON DELETE CASCADE,
            sector TEXT,
            estimated_revenue TEXT,
            past_projects TEXT NOT NULL,
            key_partners TEXT NOT NULL,
            confidence TEXT NOT NULL,
            sources TEXT NOT NULL
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


def save_research(tender_id: str, research: ProspectResearch) -> None:
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO research
           (tender_id, sector, estimated_revenue, past_projects,
            key_partners, confidence, sources)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            tender_id, research.sector, research.estimated_revenue,
            json.dumps(research.past_projects), json.dumps(research.key_partners),
            research.confidence, json.dumps(research.sources),
        ),
    )
    conn.commit()
    conn.close()


def get_research(tender_id: str) -> Optional[ProspectResearch]:
    conn = _get_conn()
    row = conn.execute("SELECT * FROM research WHERE tender_id = ?", (tender_id,)).fetchone()
    conn.close()
    return _row_to_research(row) if row else None


def list_research() -> dict[str, ProspectResearch]:
    conn = _get_conn()
    rows = conn.execute("SELECT * FROM research").fetchall()
    conn.close()
    return {r["tender_id"]: _row_to_research(r) for r in rows}


def delete_research(tender_id: str) -> bool:
    conn = _get_conn()
    cur = conn.execute("DELETE FROM research WHERE tender_id = ?", (tender_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted
