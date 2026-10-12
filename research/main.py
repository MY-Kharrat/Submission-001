"""
Prospect Research API. All endpoints except /health require X-Internal-Token.
Runs the agentic Runner for a stored tender and returns ProspectResearch.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException

from shared import store
from research.runner import Runner, TenderNotFoundError
from shared.auth import require_internal_token
from shared.schemas import ProspectResearch

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.init_db()
    yield


app = FastAPI(title="Prospect Research Service API", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/research", response_model=dict[str, ProspectResearch])
def list_research(
    _token: str = Depends(require_internal_token),
) -> dict[str, ProspectResearch]:
    return store.list_research()


@app.post("/research/{tender_id}", response_model=ProspectResearch)
async def run_research(
    tender_id: str,
    _token: str = Depends(require_internal_token),
) -> ProspectResearch:
    """Run (or re-run) prospect research.

    POST because a run spends search + LLM budget: it is neither safe nor
    idempotent-by-default, so it must not be a GET (prefetchers, crawlers
    and browser retries fire GETs). Exact-tender caching inside Runner makes
    unchanged repeated triggers cheap without reusing opportunity-specific
    evidence across separate RFPs from the same issuer.
    """
    try:
        # finalize() already persists exactly once — no second write here.
        return await Runner().run(tender_id)
    except TenderNotFoundError:
        raise HTTPException(status_code=404, detail="Tender not found")


@app.get("/research/{tender_id}", response_model=ProspectResearch)
def get_stored_research(
    tender_id: str,
    _token: str = Depends(require_internal_token),
) -> ProspectResearch:
    result = store.get_research(tender_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No research found for tender")
    return result
