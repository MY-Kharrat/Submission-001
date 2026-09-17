"""
Given a piece of text (e.g. a tender's requirements), embed it and return
the most relevant stored chunks (CVs, past projects, tools) from Supabase.
"""

from dotenv import load_dotenv
load_dotenv()

from rag.embeddings import embed_text
from rag.supabase_client import supabase


def query_knowledge_base(text: str, top_k: int = 5, doc_type: str | None = None) -> list[dict]:
    """
    Returns a list of matching chunks, each like:
    {
        "id": "...",
        "content": "...",
        "doc_type": "cv",
        "source_file": "ahmed_cv.pdf",
        "similarity": 0.83
    }
    """
    query_embedding = embed_text(text)

    response = supabase.rpc(
        "match_knowledge_chunks",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "filter_doc_type": doc_type,
        },
    ).execute()

    return response.data


if __name__ == "__main__":
    # quick manual test -- run: python -m rag.query
    sample_tender = "Looking for a team experienced in React and payment gateway integrations for a fintech client."
    results = query_knowledge_base(sample_tender, top_k=5)

    for r in results:
        print(f"[{r['similarity']:.2f}] {r['doc_type']} / {r['source_file']}")
        print(f"    {r['content'][:120]}...")