"""
SQLite persistence for tenders and prospect research: WAL mode with a UNIQUE
dedup_hash column to keep duplicate ingests idempotent at the storage layer.

Connection discipline: every helper opens its own connection and closes it in
a ``finally`` block (via the ``_connect()`` context manager). The previous
version called ``conn.close()`` only on the happy path, so a failed INSERT
leaked the connection while the traceback kept the frame alive, holding a
write lock that broke every subsequent test with "database is locked".
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Iterator, Optional

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


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    """Yield a configured connection, always closing it — even on error."""
    conn = _get_conn()
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def init_db() -> None:
    with _connect() as conn:
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
                issuer TEXT NOT NULL DEFAULT '',
                sector TEXT,
                estimated_revenue TEXT,
                past_projects TEXT NOT NULL,
                key_partners TEXT NOT NULL,
                notes TEXT NOT NULL DEFAULT '',
                confidence TEXT NOT NULL,
                sources TEXT NOT NULL
            )
        """)
        # Migration for DBs created before issuer/notes columns existed.
        for col, ddl in (
            ("issuer", "ALTER TABLE research ADD COLUMN issuer TEXT NOT NULL DEFAULT ''"),
            ("notes", "ALTER TABLE research ADD COLUMN notes TEXT NOT NULL DEFAULT ''"),
        ):
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(research)").fetchall()}
            if col not in cols:
                conn.execute(ddl)
        conn.commit()


def save_tender(tender: Tender, dedup_hash: str) -> None:
    with _connect() as conn:
        try:
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
        except Exception:
            conn.rollback()
            raise


def get_tender(tender_id: str) -> Optional[Tender]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tenders WHERE id = ?", (tender_id,)).fetchone()
        return _row_to_tender(row) if row else None


def find_by_hash(dedup_hash: str) -> Optional[Tender]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM tenders WHERE dedup_hash = ?", (dedup_hash,)).fetchone()
        return _row_to_tender(row) if row else None


def list_tenders(status: Optional[str] = None) -> list[Tender]:
    with _connect() as conn:
        if status:
            rows = conn.execute("SELECT * FROM tenders WHERE status = ?", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tenders").fetchall()
        return [_row_to_tender(r) for r in rows]


def save_research(tender_id: str, research: ProspectResearch) -> None:
    with _connect() as conn:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO research
                   (tender_id, issuer, sector, estimated_revenue, past_projects,
                    key_partners, notes, confidence, sources)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    tender_id, research.issuer, research.sector, research.estimated_revenue,
                    json.dumps(research.past_projects), json.dumps(research.key_partners),
                    research.notes, research.confidence, json.dumps(research.sources),
                ),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def get_research(tender_id: str) -> Optional[ProspectResearch]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM research WHERE tender_id = ?", (tender_id,)).fetchone()
        return _row_to_research(row) if row else None


def get_research_by_issuer(issuer: str) -> Optional[ProspectResearch]:
    """Issuer-keyed cache lookup: latest stored research for any tender of this issuer.

    Lets a repeated run for the same organization (or a second tender from the
    same issuer) be served without re-spending search/LLM budget.
    """
    with _connect() as conn:
        # rowid is the insertion counter, so this is true recency. ORDER BY
        # tender_id would be lexicographic on an opaque id ("t10" < "t2").
        row = conn.execute(
            "SELECT * FROM research WHERE issuer = ? ORDER BY rowid DESC LIMIT 1",
            (issuer,),
        ).fetchone()
        return _row_to_research(row) if row else None

def get_research_by_tender_id(tender_id:str)-> Optional[ProspectResearch]:
    """tender_id-keyed cache lookup: latest stored research for specific tender

    Lets a repeated run for the same organization (or a second tender from the
    same issuer) be served without re-spending search/LLM budget.
    """
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM research WHERE tender_id = ? ",
            (tender_id,),
        ).fetchone()
        return _row_to_research(row) if row else None

def list_research() -> dict[str, ProspectResearch]:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM research").fetchall()
        return {r["tender_id"]: _row_to_research(r) for r in rows}


def delete_research(tender_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM research WHERE tender_id = ?", (tender_id,))
        conn.commit()
        return cur.rowcount > 0
