import os
import tempfile
import time
from pathlib import Path
from typing import Awaitable, Callable, Optional

from shared.schemas import ExtractorOutput, Fact, ProspectResearch, Tender
from research.audit import AuditLogger
from research.extractor import Extractor
from research.search import SearchResult, SearchTool
from shared import store as _store
from shared.store import get_tender, save_research

GAP_QUERIES: dict[str, str] = {
    "sector": "sector industry",
    "estimated_revenue": "estimated revenue annual budget",
    "past_projects": "past projects",
    "key_partners": "key partners",
}


class TenderNotFoundError(LookupError):
    """Raised when Runner.run() is asked for a tender that does not exist."""


def compute_confidence(facts: list[Fact], sources: set[str]) -> str:
    """Deterministic, evidence-based confidence a judge can point at.

    - "high":   >= 2 distinct contributing source_urls AND at least one
                medium/high-confidence fact.
    - "medium": exactly one contributing source but at least one
                medium/high-confidence fact (one solid single-source fact).
    - "low":    everything else (thin or contradictory evidence — e.g. only
                low-confidence facts, or no facts at all).

    Note this deliberately ignores the *count* of populated fields: four
    facts from a single source_url are "medium", never "high".
    """

    ## TODO: Check issuer's tender history. Set 2 thresholds.
    ## If number of tenders > thresh1, conf_points++++. If conf_points> thresh2, conf=high
    solid = any(f.confidence in ("medium", "high") for f in facts)
    if len(sources) >= 2 and solid:
        return "high"
    if len(sources) >= 1 and solid:
        return "medium"
    return "low"


def _default_log_path() -> str:
    configured = os.environ.get("RESEARCH_AUDIT_LOG", "/var/log/oliveSoft/research.log")
    try:
        parent = Path(configured).parent
        parent.mkdir(parents=True, exist_ok=True)
        return configured
    except OSError:
        # Read-only container filesystem: never raise at construction;
        # fall back to the OS temp dir (always writable).
        fallback = Path(tempfile.gettempdir()) / "research-audit.log"
        try:
            fallback.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return str(fallback)


class Runner:
    def __init__(
        self,
        search_tool: Optional[SearchTool] = None,
        extractor: Optional[Extractor] = None,
        save_prospect: Optional[Callable[[str, ProspectResearch], None]] = None,
        audit: Optional[AuditLogger] = None,
        iteration_cap: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        fetch_tender: Optional[Callable[[str], Optional[Tender]]] = None,
    ):
        self.search_tool = search_tool or SearchTool()
        self.extractor = extractor or Extractor()

        # Audit log points at a configured/data volume, never the source tree,
        # so a read-only container filesystem can't 500 every request.
        self.audit = audit or AuditLogger(log_path=_default_log_path())
        self.fetch_tender = fetch_tender or get_tender
        self.save_research = save_prospect or save_research
        self.iteration_cap = iteration_cap or int(os.environ.get("MAX_SEARCH_ITERATIONS", "4"))
        self.timeout_seconds = timeout_seconds or float(os.environ.get("TOTAL_RUN_TIMEOUT_SECONDS", "30"))
        self.accumulated_facts: dict[str, list[Fact]] = {}
        self.sources: set[str] = set()
        # Current-run context (set at the top of run(); also the fallback
        # that lets finalize() be called with no args, as the tests do).
        self._current_tender_id: Optional[str] = None
        self._current_issuer: str = ""

    def _missing_fields(self) -> list[str]:
        # return [k for k in GAP_QUERIES.keys() if not self.accumulated_facts.get(k)]
        return ", ".join(k for k in GAP_QUERIES if not self.accumulated_facts.get(k))

    def _build_query(self, issuer: str, gap: str) -> str:
        label = GAP_QUERIES.get(gap)
        return f"{issuer} {label}"

    def _all_facts(self) -> list[Fact]:
        return [f for items in self.accumulated_facts.values() for f in items]

    def finalize(self, tender_id: Optional[str] = None, issuer: Optional[str] = None) -> ProspectResearch:
        tid = tender_id or self._current_tender_id or ""
        iss = issuer if issuer is not None else (self._current_issuer or "")
        if not iss and tid:
            # finalize(tender_id) without a prior run(): resolve the issuer
            # best-effort so ProspectResearch.issuer is never silently blank.
            try:
                tender = self.fetch_tender(tid)
                if tender is not None:
                    iss = tender.issuer
            except Exception:
                pass

        def first(field: str) -> Optional[str]:
            items = self.accumulated_facts.get(field) or []
            return items[0].value if items else None

        sector = first("sector")
        estimated_revenue = first("estimated_revenue")
        past_projects = [f.value for f in self.accumulated_facts.get("past_projects", [])]
        key_partners = [f.value for f in self.accumulated_facts.get("key_partners", [])]

        # Sector disagreement is recorded in notes (required behavior).
        sector_values = [f.value for f in self.accumulated_facts.get("sector", [])]
        distinct_sectors = sorted(set(sector_values))
        if len(distinct_sectors) > 1:
            notes = "Sector disagreement across sources: " + "; ".join(distinct_sectors)
        else:
            notes = ""

        confidence = compute_confidence(self._all_facts(), self.sources)
        pr = ProspectResearch(
            tender_id=tid,
            issuer=iss,
            sector=sector,
            estimated_revenue=estimated_revenue,
            past_projects=past_projects,
            key_partners=key_partners,
            notes=notes,
            confidence=confidence,  # type: ignore[arg-type]
            sources=sorted(self.sources),
        )
        self.save_research(tid, pr)
        return pr

    def fetch_cached(self,issuer)-> Optional[ProspectResearch]:
        try:
            ## TODO:change to get_research_by_tenderid
            cached = _store.get_research_by_issuer(issuer)
        except Exception:
            cached = None
        return cached
        
    async def run(
        self,
        tender_id: str,
        search_fn: Optional[Callable[[str], Awaitable[list[SearchResult]]]] = None,
        extract_fn: Optional[Callable[..., Awaitable[object]]] = None,
    ) -> ProspectResearch:
        # Reset per-run state so a reused Runner never leaks facts between tenders.
        self.accumulated_facts = {}
        self.sources = set()
        self._current_tender_id = tender_id

        start = time.monotonic()
        search = search_fn or self.search_tool.search
        tender = self.fetch_tender(tender_id)
        if tender is None:
            try:
                self.audit.log("tender_missing", {"tender_id": tender_id}, "")
            finally:
                raise TenderNotFoundError(f"Tender not found: {tender_id}")
        issuer = tender.issuer
        self._current_issuer = issuer

        # Issuer-keyed cache: a repeated run for the same organization — or a
        # second tender from the same issuer — is served without re-spending
        # search/LLM budget, making the operation idempotent across retries.
        cached = self.fetch_cached(issuer)
        if cached is not None:
            self.audit.log("fetch_cache", {"tender_id": tender_id, "issuer": issuer}, "")
            self.save_research(tender_id, cached)
            return cached
        
        iterations = 0
        while iterations < self.iteration_cap:
            if time.monotonic() - start >= self.timeout_seconds:
                # Stops loop because exceeded timeout
                break
            # This returns list of missing fields (sector, estimated_revenue, key_partners, past_projects)
            missing = self._missing_fields()
            if len(missing) <= 0:
                # There's no missing facts
                self.audit.log("missing_fields", {}, "There's no missing field!")
                break

            # # Round-robin over the *current* missing set: if the first gap can
            # # never be filled, later iterations still attempt the other fields
            # # instead of re-searching the same un-fillable gap forever.
            # gap = missing[iterations % len(missing)]

            # Builds a query: <issuer, gap>
            query = self._build_query(issuer, missing)
            try:
                # Performs a web search using Tavily
                results = await search(query)
            except Exception as e:
                self.audit.log("Search Error", {"query": query}, str(e))
                iterations += 1
                continue
            results = (results or [])

            # Process the top results per query (not just the first success):
            # audit every snippet, and let each extraction fail gracefully so
            # one flaky LLM call degrades instead of aborting the whole run.
            for r in results:
                if time.monotonic() - start >= self.timeout_seconds:
                    break
                self.audit.log("search", {"query": query}, r.content)

                try:
                    if extract_fn is not None:
                        # Uses custom provided extractor
                        out = await extract_fn(r.content, r.url)
                    else:
                        # Uses the established Quarantined LLM
                        out = await self.extractor.extract(r.content, r.url)
                except Exception:
                    continue

                for fact in out:
                    if fact.category not in GAP_QUERIES:
                        continue
                    self.accumulated_facts.setdefault(fact.category, []).append(fact)
                    self.sources.add(r.url)
            iterations += 1
        return self.finalize(tender_id)
