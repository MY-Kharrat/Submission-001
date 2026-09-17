"""
FastAPI service exposing the RAG retrieval as an HTTP endpoint.

Run locally with:
    uvicorn rag.main:app --reload --port 8001

n8n (or your teammates' agents) can then POST to:
    http://localhost:8001/query
"""

from fastapi import FastAPI
from pydantic import BaseModel

from rag.query import query_knowledge_base

app = FastAPI(title="OliveSoft RAG Retrieval Service")


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5
    doc_type: str | None = None   # optional filter: 'cv' | 'project' | 'tool'


class QueryResult(BaseModel):
    id: str
    content: str
    doc_type: str
    source_file: str
    similarity: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=list[QueryResult])
def query(request: QueryRequest):
    results = query_knowledge_base(
        text=request.text,
        top_k=request.top_k,
        doc_type=request.doc_type,
    )
    return results