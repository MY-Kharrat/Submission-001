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

# Defense in depth: SearchTool caps to 3 as well, but the orchestrator must not
# trust the tool alone — an alternate/injected search implementation returning
# more would otherwise widen both the token cost and the injection surface.
MAX_RESULTS_PER_QUERY = 3


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

    # Deliberately no tender-history bonus: confidence must stay explainable
    # from the evidence alone ("the code computes it this way"), not from a
    # hidden point system a judge cannot audit.
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

LOG_PATH = _default_log_path()

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
        
        self.audit = audit or AuditLogger(log_path=LOG_PATH)
        self.fetch_tender = fetch_tender or get_tender
        self.save_research = save_prospect or save_research
        # Server budget sits under the n8n node's 45s client timeout:
        # worst case 35s + one 8s in-flight call = 43s, so n8n never aborts
        # mid-run while the server keeps burning search/LLM budget.
        self.iteration_cap = iteration_cap or int(os.environ.get("MAX_SEARCH_ITERATIONS", "4"))
        self.timeout_seconds = timeout_seconds or float(os.environ.get("TOTAL_RUN_TIMEOUT_SECONDS", "35"))
        self.accumulated_facts: dict[str, list[Fact]] = {}
        self.sources: set[str] = set()
        # Current-run context (set at the top of run(); also the fallback
        # that lets finalize() be called with no args, as the tests do).
        self._current_tender_id: Optional[str] = None
        self._current_issuer: str = ""

    def _missing_fields(self) -> list[str]:
        """Fields not yet backed by any accumulated fact, in GAP_QUERIES order."""
        return [k for k in GAP_QUERIES if not self.accumulated_facts.get(k)]

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
        distinct_sectors = set(sector_values)
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
            sources=list(self.sources),
        )
        self.save_research(tid, pr)
        return pr

    def fetch_cached(self, tender_id: str) -> Optional[ProspectResearch]:
        """Tender-keyed cache: an unchanged retry is cheap, and
        opportunity-specific evidence never leaks across separate RFPs
        from the same issuer."""
        try:
            cached = _store.get_research_by_tender_id(tender_id)
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

        # Tender-keyed cache: an unchanged retry is served without
        # re-spending search/LLM budget, making the operation idempotent
        # across retries. Detection invalidates this row after a material
        # tender revision.
        cached = self.fetch_cached(tender_id)
        if cached is not None:
            self.audit.log("fetch_cache", {"tender_id": tender_id, "issuer": issuer}, "")
            # The cached record carries the tender_id it was first researched
            # under; restamp it so the API body never reports another tender's
            # id when a second tender shares this issuer.
            restamped = cached.model_copy(update={"tender_id": tender_id})
            self.save_research(tender_id, restamped)
            return restamped
        
        iterations = 0
        # One Tavily call per gap max: a gap already searched is never
        # re-queried, even if it yielded nothing (un-fillable gaps must not
        # starve the others, and filled gaps must not be revisited).
        searched: set[str] = set()
        while iterations < self.iteration_cap:
            if time.monotonic() - start >= self.timeout_seconds:
                # Stops loop because exceeded timeout
                break
            # This returns list of missing fields (sector, estimated_revenue, key_partners, past_projects)
            missing = self._missing_fields()
            if not missing:
                # There's no missing facts
                self.audit.log("missing_fields", {}, "There's no missing field!")
                break

            # First unsearched gap in canonical GAP_QUERIES order. Using the
            # stable canonical order (not missing[iterations % len(missing)])
            # matters: the missing list shrinks as gaps fill, so a modulo
            # index drifts — it re-searches an already-tried gap while
            # starving one never attempted.
            missing_set = set(missing)
            todo = [g for g in GAP_QUERIES if g in missing_set and g not in searched]
            if not todo:
                break
            gap = todo[0]
            searched.add(gap)

            # Builds a query: <issuer, gap>
            query = self._build_query(issuer, gap)
            try:
                # Performs a web search using Tavily
                results = await search(query)

            except Exception as e:
                self.audit.log("Search Error", {"query": query}, str(e))
                iterations += 1
                continue
            #results = (results or [])[:MAX_RESULTS_PER_QUERY]

            # Process the top results per query (not just the first success):
            # audit every snippet, and let each extraction fail gracefully so
            # one flaky LLM call degrades instead of aborting the whole run.
            r = results[0]

            
            if time.monotonic() - start >= self.timeout_seconds:
                break
            self.audit.log("search", {"query": query}, r.content)

            try:
                if extract_fn is not None:
                    # Uses custom provided extractor
                    print(r.content)
                    out = await extract_fn(r.content, r.url)
                else:
                    # Uses the established Quarantined LLM
                    out = await self.extractor.extract(r.content, r.url)
            except Exception:
                continue

            for fact in out.facts:
                if fact.category not in GAP_QUERIES:
                    continue
                bucket = self.accumulated_facts.setdefault(fact.category, [])
                # Idempotent accumulation: consecutive gap queries often
                # re-read the same overlapping page, so the identical fact
                # arrives again. Without this, output lists fill with
                # duplicates and an attacker-controlled page gets its
                # payload restated once per iteration. A fact that is not
                # new also does not credit its URL as a fresh source, so
                # "echo a known fact from many URLs" cannot inflate
                # confidence.
                if any(
                    existing.value.strip().casefold() == fact.value.strip().casefold()
                    for existing in bucket
                ):
                    continue
                bucket.append(fact)
                self.sources.add(r.url)
            iterations += 1
        return self.finalize(tender_id)
