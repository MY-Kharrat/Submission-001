"""
FastAPI service exposing the RAG retrieval as an HTTP endpoint.

Run locally with:
    uvicorn rag.main:app --reload --port 8001

n8n (or your teammates' agents) can then POST to:
    http://localhost:8001/query
"""


from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from rag.embeddings import EmbeddingError
from shared.schemas import QueryRequest, QueryResult
from rag.query import RetrievalError, query_knowledge_base
from rag.supabase_client import RagConfigurationError
from shared.auth import require_internal_token

load_dotenv()

app = FastAPI(title="OliveSoft RAG Retrieval Service")



@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=list[QueryResult])
def query(
    request: QueryRequest,
    _token: str = Depends(require_internal_token),
) -> list[dict]:
    try:
        return query_knowledge_base(
            text=request.text,
            top_k=request.top_k,
            doc_type=request.doc_type,
            similarity_threshold=request.similarity_threshold,
        )
    except (RagConfigurationError, EmbeddingError, RetrievalError) as exc:
        # Do not expose provider URLs, credentials, or SDK exception details.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
