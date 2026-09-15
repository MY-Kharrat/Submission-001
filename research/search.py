import os
import re
from dataclasses import dataclass


@dataclass
class SearchResult:
    url: str
    content: str


def clean_text(raw: str) -> str:
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class SearchTool:
    def __init__(self, max_results: int = 3):
        self.max_results = min(max(1, max_results), 3)

    async def search(self, query: str) -> list[SearchResult]:
        """Uses Tavily to """
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return []
        try:
            from tavily import AsyncTavilyClient
        except ImportError:
            ## Should indicate this
            return await self._http_search(query, api_key)
        
        client = AsyncTavilyClient(api_key=api_key)
        
        # Search query across the web
        resp = await client.search(
            query,
            max_results=self.max_results,
            include_answer=False,
            include_raw_content=False,
        )
        out: list[SearchResult] = []
        
        for item in (resp.get("results") or [])[: self.max_results]: 
            url = str(item.get("url", ""))
            content = clean_text(str(item.get("content", "")))
            if url and content:
                out.append(SearchResult(url=url, content=content))
        return out

    async def _http_search(self, query: str, api_key: str) -> list[SearchResult]:
        import httpx

        timeout = int(os.environ.get("PER_CALL_TIMEOUT_SECONDS", "8"))
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                headers={"Content-Type": "application/json"},
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": self.max_results,
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        out: list[SearchResult] = []
        for item in (data.get("results") or [])[: self.max_results]:
            url = str(item.get("url", ""))
            content = clean_text(str(item.get("content", "")))
            if url and content:
                out.append(SearchResult(url=url, content=content))
        return out
