"""
Tender normalization: validate → dedup → extract requirements → tag sector.

Validation and dedup run before any LLM call. Requirements extraction only fires
when the payload omits a clean requirements list; sector tagging uses keyword
matching first and falls back to LLM classification for unknown sectors.
"""

import hashlib
import json
import logging
import re
import uuid
from datetime import date, datetime, timezone

import httpx

from shared.schemas import CAPABILITY_TAXONOMY, Tender
from shared import store
from detection.llm import llm_call

logger = logging.getLogger(__name__)

RAW_TEXT_MAX_CHARS = 20_000
TITLE_MAX_CHARS = 300
ISSUER_MAX_CHARS = 200
REQUIREMENT_MAX_CHARS = 1_000
REQUIREMENTS_MAX_ITEMS = 100

# Keyphrases per service line, drawn from the tools OliveSoft publishes against
# each line. Declaration order here is not the tie-break authority — CAPABILITY_TAXONOMY
# is, and the drift guard below keeps the two key sets identical.
SECTOR_KEYPHRASES: dict[str, list[str]] = {
    "Data Integration": [
        "mulesoft", "talend", "ssis", "boomi", "workato", "n8n",
        "data integration", "integration platform", "api integration",
        "system integration", "erp integration", "crm integration",
        "etl", "elt", "esb", "middleware",
    ],
    "AI Development": [
        "artificial intelligence", "machine learning", "deep learning", "neural network",
        "natural language processing", "nlp", "computer vision", "generative ai",
        "large language model", "llm", "retrieval augmented generation", "rag",
        "chatbot", "conversational ai", "predictive analytics", "predictive model",
        "forecasting", "demand prediction", "anomaly detection", "text classification",
        "text mining", "mlops",
    ],
    "BI & Dashboarding": [
        "power bi", "tableau", "qlik", "qlik sense", "looker studio", "business intelligence",
        "dashboard", "dashboards", "semantic model", "data visualization",
        "data visualisation", "kpi", "reporting layer",
    ],
    "Salesforce Ecosystem": [
        "salesforce", "sales cloud", "service cloud", "marketing cloud", "commerce cloud",
        "crm analytics", "crm", "customer relationship management",
    ],
    "Data Platform": [
        "snowflake", "databricks", "kafka", "bigquery", "azure data", "azure synapse",
        "gcp", "google cloud", "data platform", "data warehouse", "data lake",
        "lakehouse", "data lakehouse", "cloud data", "analytics platform",
    ],
}

if set(SECTOR_KEYPHRASES) != set(CAPABILITY_TAXONOMY):
    _missing = sorted(set(CAPABILITY_TAXONOMY) - set(SECTOR_KEYPHRASES))
    _extra = sorted(set(SECTOR_KEYPHRASES) - set(CAPABILITY_TAXONOMY))
    raise RuntimeError(
        "SECTOR_KEYPHRASES and CAPABILITY_TAXONOMY have drifted apart. "
        f"No keyphrases for: {_missing}. Keyphrases for unknown categories: {_extra}."
    )



class ValidationError(Exception):
    """Raised when an inbound payload fails schema or size-cap validation."""


def compute_dedup_hash(title: str, issuer: str) -> str:
    """16-char SHA-256 of canonical `title|issuer` — natural key for tenders.

    Canonicalization (whitespace-collapsed, casefolded) keeps sheet re-reads
    and n8n retries from minting duplicate rows over cosmetic differences."""

    def canonical(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip().casefold()

    return hashlib.sha256(
        f"{canonical(title)}|{canonical(issuer)}".encode("utf-8")
    ).hexdigest()[:16]


def validate_fields(data: dict) -> None:
    """Reject missing required fields, bad dates, or over-cap raw_text before any DB/LLM work."""
    for field in ("title", "issuer", "deadline", "raw_text"):
        val = data.get(field)
        if val is None or not isinstance(val, str) or not val.strip():
            raise ValidationError(f"Missing required field: {field}")

    try:
        date.fromisoformat(data["deadline"].strip())
    except (ValueError, AttributeError):
        raise ValidationError(
            f"Invalid deadline date format: '{data.get('deadline')}'. Expected YYYY-MM-DD."
        )

    raw_text = data["raw_text"]
    if len(raw_text) > RAW_TEXT_MAX_CHARS:
        raise ValidationError(
            f"raw_text exceeds {RAW_TEXT_MAX_CHARS} character cap ({len(raw_text)} chars)"
        )
    if len(data["title"].strip()) > TITLE_MAX_CHARS:
        raise ValidationError(f"title exceeds {TITLE_MAX_CHARS} character cap")
    if len(data["issuer"].strip()) > ISSUER_MAX_CHARS:
        raise ValidationError(f"issuer exceeds {ISSUER_MAX_CHARS} character cap")

    source = data.get("source")
    if source is not None and source not in {"manual", "simulated_feed"}:
        raise ValidationError("source must be 'manual' or 'simulated_feed'")

    explicit_id = data.get("id")
    if explicit_id is not None and (
        not isinstance(explicit_id, str) or not explicit_id.strip()
    ):
        raise ValidationError("id must be a non-empty string when supplied")

    requirements = data.get("requirements")
    if requirements is not None:
        if not isinstance(requirements, list):
            raise ValidationError("requirements must be a list of strings")
        if len(requirements) > REQUIREMENTS_MAX_ITEMS:
            raise ValidationError(
                f"requirements exceeds {REQUIREMENTS_MAX_ITEMS} item cap"
            )
        if any(
            not isinstance(item, str)
            or not item.strip()
            or len(item.strip()) > REQUIREMENT_MAX_CHARS
            for item in requirements
        ):
            raise ValidationError(
                "each requirement must be a non-empty string of at most "
                f"{REQUIREMENT_MAX_CHARS} characters"
            )

    detected_at = data.get("detected_at")
    if detected_at is not None and not isinstance(detected_at, datetime):
        if not isinstance(detected_at, str):
            raise ValidationError("detected_at must be an ISO-8601 timestamp")
        try:
            datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("Invalid detected_at timestamp") from exc


def tag_sector(raw_text: str, title: str) -> str:
    """
    Best-keyphrase-matching service line, or 'unknown' when nothing matches.

    Highest score wins. Ties go to whichever category comes first in
    CAPABILITY_TAXONOMY, so the result depends on the declared taxonomy order
    and not on the order keyphrases happen to be listed in.
    """
    text_lower = f"{title} {raw_text}".lower()
    scores: dict[str, int] = {}

    for category, keyphrases in SECTOR_KEYPHRASES.items():
        count = 0
        for phrase in keyphrases:
            pattern = r"\b" + re.escape(phrase) + r"\b"
            if re.search(pattern, text_lower):
                count += 1
        if count > 0:
            scores[category] = count

    if not scores:
        return "unknown"

    precedence = {category: rank for rank, category in enumerate(CAPABILITY_TAXONOMY)}
    return min(scores, key=lambda category: (-scores[category], precedence[category]))


def extract_json_payload(text: str) -> str:
    """Pull JSON out of an LLM response, stripping Markdown fences if present."""
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    array_match = re.search(r"\[\s*[\s\S]*?\s*\]", text)
    if array_match:
        return array_match.group(0).strip()
    return text.strip()


async def extract_requirements_llm(raw_text: str) -> list[str]:
    """Extracts structured requirements bullets from unstructured text using LLM."""
    prompt = (
        "Extract 3-6 concise bullet-point requirements from this tender text. "
        "Return a JSON array of strings, nothing else.\n\n"
        f"Text:\n{raw_text[:5000]}"
    )
    try:
        result = await llm_call(prompt)
    except (httpx.HTTPError, KeyError, IndexError):
        # Transient transport failure or malformed provider envelope:
        # degrade to deterministic sentence splitting instead of 500ing the
        # ingest. Config errors (ValueError) and 4xx propagate — a missing
        # API key must fail loudly, never masquerade as real requirements.
        logger.warning(
            "Requirement extraction LLM unavailable; using deterministic fallback"
        )
        sentences = [
            " ".join(item.split()).strip(" -•\t")
            for item in re.split(r"(?<=[.!?;])\s+|\n+", raw_text)
        ]
        usable = [item[:300] for item in sentences if len(item) >= 12]
        return usable[:6] or ["General technical requirement compliance"]
    cleaned = extract_json_payload(result)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list) and all(isinstance(r, str) for r in parsed):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Non-JSON LLM output still yields a usable list rather than failing the ingest
    lines = []
    for line in result.strip().splitlines():
        cleaned_line = line.strip(" `*-\t\r\n")
        if (
            cleaned_line
            and not cleaned_line.startswith("[")
            and not cleaned_line.startswith("]")
            and not cleaned_line.startswith("```")
        ):
            lines.append(cleaned_line)
    return lines or ["General technical requirement compliance"]


async def tag_sector_llm(raw_text: str, title: str) -> str:
    """Fallback LLM sector classification for tenders failing keyphrase matching."""
    categories_str = "\n".join(f"- {c}" for c in CAPABILITY_TAXONOMY)
    prompt = (
        f"Classify this tender into exactly one category. Return only the category name.\n\n"
        f"Categories:\n{categories_str}\n\n"
        f"Title: {title}\nText:\n{raw_text[:3000]}"
    )
    try:
        result = (await llm_call(prompt)).strip()
    except (httpx.HTTPError, KeyError, IndexError):
        logger.warning("Sector classification LLM unavailable; retaining unknown sector")
        return "unknown"
    if result in CAPABILITY_TAXONOMY:
        return result
    return "unknown"


async def normalize_tender(data: dict) -> tuple[Tender, bool]:
    """
    Executes tender ingestion and normalization pipeline.

    Returns:
        tuple[Tender, is_existing]: Tender model and a boolean indicating whether a duplicate
        was retrieved from store without re-processing.
    """
    validate_fields(data)

    title = data["title"].strip()
    issuer = data["issuer"].strip()
    deadline = date.fromisoformat(data["deadline"].strip())
    requested_source = data.get("source")
    dedup_hash = compute_dedup_hash(title, issuer)

    explicit_id = data.get("id") if isinstance(data.get("id"), str) else None
    existing_by_id = store.get_tender(explicit_id) if explicit_id else None
    existing_by_hash = store.find_by_hash(dedup_hash)
    if (
        existing_by_id is not None
        and existing_by_hash is not None
        and existing_by_id.id != existing_by_hash.id
    ):
        raise ValidationError(
            "Tender id and title/issuer identify different existing records"
        )
    existing = existing_by_id or existing_by_hash
    source = requested_source or (existing.source if existing else "manual")

    supplied_requirements = data.get("requirements")
    unchanged = existing is not None and all(
        (
            existing.title == title,
            existing.issuer == issuer,
            existing.deadline == deadline,
            existing.raw_text == data["raw_text"],
            existing.source == source,
            supplied_requirements in (None, [])
            or existing.requirements == supplied_requirements,
        )
    )
    if unchanged:
        # Duplicate ingest (n8n retries, sheet re-reads): return the stored
        # record without re-spending LLM budget or touching research.
        assert existing is not None  # narrowed by `unchanged`
        return existing, True

    requirements = data.get("requirements") or []
    if not requirements:
        requirements = await extract_requirements_llm(data["raw_text"])

    sector = tag_sector(data["raw_text"], title)
    if sector == "unknown":
        sector = await tag_sector_llm(data["raw_text"], title)

    detected_at = data.get("detected_at")
    if detected_at is None:
        detected_at = existing.detected_at if existing else datetime.now(timezone.utc)
    elif isinstance(detected_at, str):
        try:
            detected_at = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationError("Invalid detected_at timestamp") from exc

    # `is_existing` means "no new row": the natural key already identified a
    # tender. A re-announcement with new scope still refreshes the stored row
    # (a new deadline must take effect) but stays the same tender.
    is_existing = existing is not None

    tender = Tender(
        id=existing.id if existing else data.get("id", str(uuid.uuid4())),
        title=title,
        issuer=issuer,
        sector=sector,
        requirements=requirements,
        deadline=deadline,
        raw_text=data["raw_text"],
        source=source,
        detected_at=detected_at,
        status="new",
    )

    store.save_tender(tender, dedup_hash)
    if is_existing:
        assert existing is not None  # narrowed by `is_existing`
        # Invalidate derived research only when its actual inputs changed.
        # Deadline/raw_text-only edits (the re-announcement case) refresh the
        # row without forcing a full research re-run on every schedule poll.
        research_inputs_changed = (
            existing.issuer != issuer
            or existing.title != title
            or existing.requirements != requirements
            or existing.sector != sector
        )
        if research_inputs_changed:
            store.delete_research(tender.id)
    return tender, is_existing
