"""
Tender Detection API. All endpoint except /health require X-Internal-Token.
Validation failures surface as HTTP 422.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pydantic
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query

from shared.auth import require_internal_token
from shared.schemas import Tender
from detection import store
from detection.normalize import ValidationError, normalize_tender

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Tender Detection Service API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness health check endpoint."""
    return {"status": "ok"}


@app.post("/tenders/ingest", response_model=Tender)
async def ingest_tender(
    data: dict,
    _token: str = Depends(require_internal_token),
) -> Tender:
    """
    Ingests, validates, deduplicates, and classifies an incoming tender payload.

    Returns HTTP 422 on schema or validation errors.
    """
    try:
        tender, _is_existing = await normalize_tender(data)
        return tender
    except (ValidationError, ValueError, pydantic.ValidationError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/tenders/{tender_id}", response_model=Tender)
def get_tender(
    tender_id: str,
    _token: str = Depends(require_internal_token),
) -> Tender:
    """Retrieves a single tender by tender_id."""
    tender = store.get_tender(tender_id)
    if not tender:
        raise HTTPException(status_code=404, detail="Tender not found")
    return tender


@app.get("/tenders", response_model=list[Tender])
def list_tenders(
    status: Optional[str] = Query(None),
    _token: str = Depends(require_internal_token),
) -> list[Tender]:
    """Lists tenders, optionally filtered by status query parameter."""
    return store.list_tenders(status)


@app.post("/tenders/seed")
async def seed_tenders(
    _token: str = Depends(require_internal_token),
) -> dict:
    """
    Bulk-loads simulated tender JSON feeds for testing.
    A malformed file is recorded in the response rather than failing the whole seed.
    """
    feed_dir = Path(__file__).parent / "data" / "simulated_feed"
    if not feed_dir.exists():
        raise HTTPException(status_code=404, detail="Seed data directory not found")

    results = []
    for f in sorted(feed_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            tender, is_existing = await normalize_tender(data)
            results.append({
                "file": f.name,
                "id": tender.id,
                "title": tender.title,
                "existing": is_existing,
                "status": "success",
            })
        except Exception as exc:
            results.append({
                "file": f.name,
                "status": "error",
                "error": str(exc),
            })

    return {"seeded": len(results), "tenders": results}
