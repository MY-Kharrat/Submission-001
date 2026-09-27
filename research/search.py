import asyncio
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


def _per_call_timeout() -> float:
    try:
        return float(os.environ.get("PER_CALL_TIMEOUT_SECONDS", "8"))
    except ValueError:
        return 8.0


def _is_retryable(exc: BaseException) -> bool:
    """Timeouts and 5xx/network errors retry once; 4xx fail fast."""
    import httpx

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return True
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    if isinstance(exc, httpx.HTTPError):
        return True
    # Unknown SDK errors (no status): treat as transient, bounded by timeout.
    return True


def _is_client_error(exc: BaseException) -> bool:
    import httpx

    return (
        isinstance(exc, httpx.HTTPStatusError)
        and 400 <= exc.response.status_code < 500
    )


class SearchTool:
    def __init__(self, max_results: int = 3):
        self.max_results = min(max(1, max_results), 3)

    async def _sdk_search(self, query: str, api_key: str) -> list[SearchResult]:
        """Tavily SDK path, bounded by the per-call timeout with one retry."""
        from tavily import AsyncTavilyClient

        timeout = _per_call_timeout()
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                client = AsyncTavilyClient(api_key=api_key)
                resp = await asyncio.wait_for(
                    client.search(
                        query,
                        max_results=self.max_results,
                        include_answer=False,
                        include_raw_content=False,
                    ),
                    timeout=timeout,
                )
                out: list[SearchResult] = []
                for item in (resp.get("results") or [])[: self.max_results]:
                    url = str(item.get("url", ""))
                    content = clean_text(str(item.get("content", "")))
                    if url and content:
                        out.append(SearchResult(url=url, content=content))
                return out
            except Exception as exc:  # noqa: BLE001 — classified below
                if _is_client_error(exc):
                    raise
                last_exc = exc
                if not _is_retryable(exc) or attempt == 1:
                    return []  # transient exhausted: degrade, never hang the run
                await asyncio.sleep(0.5 * (attempt + 1))
        assert last_exc is not None
        return []

    async def search(self, query: str) -> list[SearchResult]:
        """Uses Tavily to search. Every call is bounded by the per-call
        timeout, with one retry-with-backoff on timeout/5xx and no retry
        on 4xx."""
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return []
        try:
            from tavily import AsyncTavilyClient  # noqa: F401
        except ImportError:
            return await self._http_search(query, api_key)
        return await self._sdk_search(query, api_key)

    async def _http_search(self, query: str, api_key: str) -> list[SearchResult]:
        import httpx

        timeout = _per_call_timeout()
        for attempt in range(2):
            try:
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
            except Exception as exc:  # noqa: BLE001 — classified below
                if _is_client_error(exc):
                    raise
                if not _is_retryable(exc) or attempt == 1:
                    return []  # transient exhausted: degrade, never hang the run
                await asyncio.sleep(0.5 * (attempt + 1))
        return []
