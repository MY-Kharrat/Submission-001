from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# Edit here if team skills change detection and agent both import this.
CAPABILITY_TAXONOMY: list[str] = [
    "Security Assessment & Penetration Testing",
    "Custom Software Development",
    "Automation & Tooling",
    "Data/AI Integration Consulting",
]


class Tender(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    title: str
    issuer: str           # explicit, never inferred
    sector: str           # tagged against CAPABILITY_TAXONOMY; "unknown" when no match
    requirements: list[str]
    deadline: date
    raw_text: str         # size cap enforced in normalize.py (20_000 chars)
    source: Literal["simulated_feed", "manual"]
    detected_at: datetime
    status: Literal["new", "processed", "archived"]


class ExtractorOutput(BaseModel):
    """Strict schema for the Quarantined Extractor. No other output allowed."""

    model_config = ConfigDict(strict=True)

    fact: str
    relevance: Literal["sector", "estimated_revenue", "past_projects", "key_partners"]
    source_url: str


class ProspectResearch(BaseModel):
    """Final output of the Agentic Prospect Research."""

    model_config = ConfigDict(strict=True)

    sector: Optional[str] = None
    estimated_revenue: Optional[str] = None
    past_projects: list[str] = Field(default_factory=list)
    key_partners: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "low"
    sources: list[str] = Field(default_factory=list)

