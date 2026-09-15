"""
Prospect Research API. All endpoints except /health require X-Internal-Token.
Runs the agentic Runner for a stored tender and returns ProspectResearch.
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query

from shared import store
from research.runner import Runner
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


@app.get("/research/run", response_model=ProspectResearch)
async def run_research(
    tender_id: str = Query(...),
    _token: str = Depends(require_internal_token),
) -> ProspectResearch:
    result = await Runner().run(tender_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    store.save_research(tender_id, result)
    return result


@app.get("/research/{tender_id}", response_model=ProspectResearch)
def get_stored_research(
    tender_id: str,
    _token: str = Depends(require_internal_token),
) -> ProspectResearch:
    result = store.get_research(tender_id)
    if result is None:
        raise HTTPException(status_code=404, detail="No research found for tender")
    return result
