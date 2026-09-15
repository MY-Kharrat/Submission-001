import os
import time
from typing import Awaitable, Callable, Optional

from shared.schemas import ExtractorOutput, ProspectResearch, Tender
from research.audit import AuditLogger
from research.extractor import Extractor
from research.search import SearchResult, SearchTool
from detection.store import get_tender
from pathlib import Path

GAP_QUERIES: dict[str, str] = {
    "sector": "sector industry",
    "estimated_revenue": "estimated revenue annual budget",
    "past_projects": "past projects",
    "key_partners": "key partners",
}


class Runner:
    def __init__(
        self,
        search_tool: Optional[SearchTool] = None,
        extractor: Optional[Extractor] = None,
        audit: Optional[AuditLogger] = None,
        iteration_cap: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        fetch_tender: Optional[Callable[[str], Optional[Tender]]] = None,
    ):
        self.search_tool = search_tool or SearchTool()
        self.extractor = extractor or Extractor()
        
        # Log path
        base = Path(__file__).resolve().parent      
        log_dir = base / "audit"                    
        log_dir.mkdir(parents=True, exist_ok=True)  
        log_path = str(log_dir / "audit.log")
        
        self.audit = audit or AuditLogger(log_path=log_path)
        self.fetch_tender = fetch_tender or get_tender
        self.iteration_cap = iteration_cap or int(os.environ.get("MAX_SEARCH_ITERATIONS", "4"))
        self.timeout_seconds = timeout_seconds or float(os.environ.get("TOTAL_RUN_TIMEOUT_SECONDS", "30"))
        self.accumulated_facts: dict[str, list[ExtractorOutput]] = {}
        self.sources: set[str] = set()

    def _missing_fields(self) -> list[str]:
        return [k for k in GAP_QUERIES.keys() if not self.accumulated_facts.get(k)]

    def _build_query(self, issuer: str, gap: str) -> str:
        #label = next(v for k, v in GAP_QUERIES if k == gap)
        label = GAP_QUERIES.get(gap)
        return f"{issuer} {label}"

    def finalize(self) -> ProspectResearch:
        def first(field: str) -> Optional[str]:
            items = self.accumulated_facts.get(field) or []
            return items[0].fact if items else None

        sector = first("sector")
        estimated_revenue = first("estimated_revenue")
        past_projects = [o.fact for o in self.accumulated_facts.get("past_projects", [])]
        key_partners = [o.fact for o in self.accumulated_facts.get("key_partners", [])]
        populated = sum(
            [
                sector is not None,
                estimated_revenue is not None,
                len(past_projects) > 0,
                len(key_partners) > 0,
            ]
        )
        if populated == 4:
            confidence = "high"
        elif populated >= 2:
            confidence = "medium"
        else:
            confidence = "low"
        return ProspectResearch(
            sector=sector,
            estimated_revenue=estimated_revenue,
            past_projects=past_projects,
            key_partners=key_partners,
            confidence=confidence,
            sources=sorted(self.sources),
        )

    async def run(
        self,
        tender_id: str,
        search_fn: Optional[Callable[[str], Awaitable[list[SearchResult]]]] = None,
        extract_fn: Optional[Callable[[str, str], Awaitable[Optional[ExtractorOutput]]]] = None,
    ) -> ProspectResearch:
        start = time.monotonic()
        search = search_fn or self.search_tool.search
        tender = self.fetch_tender(tender_id)
        if(tender is None):
            return # Throw exception & log
        issuer = tender.issuer
        iterations = 0
        while iterations < self.iteration_cap:
            if time.monotonic() - start >= self.timeout_seconds:
                # Stops loop because exceeded timeout
                break
            # This returns list of missing fields (sector, estimated_revenu, key_partners, past_projects)
            missing = self._missing_fields() 
            if len(missing)<=0:
                # There's no missing facts
                self.audit.log("missing_fields", {})
                break

            gap = missing[0] # Fetches first missing field

            # Builds a query: <issuer, gap>
            query = self._build_query(issuer, gap) 
            try:
                # Performs a web search using Tavily
                results = await search(query)
            except Exception:
                iterations += 1
                continue
            results = (results or [])[:3]
            progressed = False

            # Now using the quarentined LLM on ONE FACT
            for r in results:
                ## For every search result, we log, extract facts from content & url
                if time.monotonic() - start >= self.timeout_seconds:
                    break
                self.audit.log("search", {"query": query}, r.content)

                # Extracts fact from result & URL
                if extract_fn is not None:
                    # Uses custom provided extractor
                    fact = await extract_fn(r.content, r.url)
                else:
                    # Uses the established Quarantined LLM 
                    fact = await self.extractor.extract(r.content, r.url)

                if fact is None:
                    # Fact couldn't be extracted
                    # TODO: If fact couldn't be inferred, we set it to unknown
                    continue

                self.accumulated_facts.setdefault(fact.relevance, []).append(fact)
                self.sources.add(fact.source_url)

                # It's true because we extracted ONE FACT successfuly
                progressed = True
                break
            iterations += 1
            # Either way, we would loop back, even if it's True
            if not progressed:
                ## TODO: We accumulate the fact & set it to None. Or, we ignore it, and let tool search for it again
                continue
        return self.finalize()
