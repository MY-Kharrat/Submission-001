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

    # extra="forbid" is a contract boundary, not pedantry. To be precise about
    # the threat: the old default (ignore) was NOT an exploitable hole here,
    # because nothing downstream ever read an undeclared key, so a smuggled
    # "command" or "source_url" could not reach the runner. What forbidding
    # buys is that an unrecognised key is now *observable* rather than silently
    # discarded -- so a model that starts emitting a different shape shows up
    # as a hard failure at this boundary instead of quietly drifting.
    #
    # The tradeoff is real: this is fail-closed, so a model that adds one
    # harmless "summary" key loses ALL facts for that snippet. That is why
    # research/tools/llm_contract_probe.py exists -- it detects exactly this
    # drift against the live model, which a fully mocked test suite cannot.
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
