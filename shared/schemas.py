from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel

# Edit here if team skills change detection and agent both import this.
# These are OliveSoft's five published service lines, in the order they should
# win a tagging tie (earlier wins). Sector tagging targets work being bought,
# not the buyer's industry — that is ProspectResearch.sector, derived from the
# web by the agent and deliberately not constrained by this list.
CAPABILITY_TAXONOMY: list[str] = [
    "Data Integration",
    "AI Development",
    "BI & Dashboarding",
    "Salesforce Ecosystem",
    "Data Platform",
]


class Tender(BaseModel):
    id: str
    title: str
    issuer: str           # explicit, never inferred
    sector: str           # tagged against CAPABILITY_TAXONOMY; "unknown" when no match
    requirements: list[str]
    deadline: date
    raw_text: str         # size cap enforced in normalize.py, not here @TODO i need to fully dev after 5 SEPTEMBER 
    source: Literal["simulated_feed", "manual"]
    detected_at: datetime
    status: Literal["new", "processed", "archived"]


class ProspectResearch(BaseModel):
    tender_id: str
    issuer: str
    sector: str | None
    estimated_revenue: str | None
    past_projects: list[str]
    key_partners: list[str]
    notes: str
    confidence: Literal["low", "medium", "high"]  # computed by runner.py, not the model
    sources: list[str]  # only URLs that contributed a fact, not every URL searched
