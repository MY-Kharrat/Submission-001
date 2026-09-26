from datetime import date, datetime
from typing import Literal, Optional, List

from pydantic import BaseModel, ConfigDict, Field, field_validator

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


class Fact(BaseModel):
    model_config = ConfigDict(strict=True)

    value: str = Field(min_length=1)
    category: Literal["sector", "estimated_revenue", "past_projects", "key_partners"]
    confidence: Literal["low", "medium", "high"]

    @field_validator("value")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # A blank fact is noise, and an attacker could pad facts=[] with empty
        # entries to make sparse output look evidence-rich.
        stripped = v.strip()
        if not stripped:
            raise ValueError("fact value must not be blank")
        return stripped


class ExtractorOutput(BaseModel):
    """Strict schema for the Quarantined Extractor. No other output allowed."""

    # extra="forbid" is a contract boundary, not pedantry: a reply carrying any
    # key outside {facts} (a smuggled "command"/"tool" field, or an
    # attacker-supplied "source_url" that would spoof provenance) is rejected
    # wholesale and degrades to facts=[]. Note the tradeoff: this is fail-closed,
    # so a model that starts adding a harmless "summary" key loses ALL facts for
    # that snippet. research/tools/llm_contract_probe.py exists to detect exactly
    # that drift against the live model, which the mocked test suite cannot.
    model_config = ConfigDict(strict=True, extra="forbid")

    # Empty list, never null: "no fact found" is facts=[].
    facts: List[Fact]

class ProspectResearch(BaseModel):
    """Final output of the Agentic Prospect Research."""

    model_config = ConfigDict(strict=True)
    tender_id: str
    issuer: str
    sector: str | None
    estimated_revenue: str | None
    past_projects: list[str]
    key_partners: list[str]
    notes: str
    confidence: Literal["low", "medium", "high"]  # computed by runner.py, not the model
    sources: list[str]  # only URLs that contributed a fact, not every URL searched
