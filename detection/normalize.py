"""
Tender normalization: validate → dedup → extract requirements → tag sector.

Validation and dedup run before any LLM call. Requirements extraction only fires
when the payload omits a clean requirements list; sector tagging uses keyword
matching first and falls back to LLM classification for unknown sectors.
"""

import hashlib
import json
import re
import uuid
from datetime import date, datetime, timezone

from shared.schemas import CAPABILITY_TAXONOMY, Tender
from detection import store
from detection.llm import llm_call

RAW_TEXT_MAX_CHARS = 20_000

SECTOR_KEYPHRASES: dict[str, list[str]] = {
    "Security Assessment & Penetration Testing": [
        "penetration testing", "pen test", "pentest", "vulnerability assessment",
        "security audit", "owasp", "security assessment", "soc2", "iso 27001"
    ],
    "Custom Software Development": [
        "custom software", "software development", "web application development",
        "full stack", "backend development", "frontend development", "custom application"
    ],
    "Automation & Tooling": [
        "automation", "workflow automation", "ci/cd", "devops", "process automation", "rpa"
    ],
    "Data/AI Integration Consulting": [
        "ai integration", "data integration", "machine learning", "llm", "rag",
        "artificial intelligence", "data pipeline", "analytics platform"
    ]
}


class ValidationError(Exception):
    """Raised when an inbound payload fails schema or size-cap validation."""


def compute_dedup_hash(title: str, issuer: str) -> str:
    """16-char SHA-256 of `title|issuer` — natural key for the tenders table."""
    return hashlib.sha256(f"{title.strip()}|{issuer.strip()}".encode()).hexdigest()[:16]


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


def tag_sector(raw_text: str, title: str) -> str:
    """Best-keyphrase-matching category, or 'unknown' when nothing matches."""
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

    best_category, _ = max(scores.items(), key=lambda x: x[1])
    return best_category


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
    result = await llm_call(prompt)
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
    result = (await llm_call(prompt)).strip()
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

    dedup_hash = compute_dedup_hash(data["title"], data["issuer"])
    existing = store.find_by_hash(dedup_hash)
    if existing:
        return existing, True

    requirements = data.get("requirements") or []
    if not requirements:
        requirements = await extract_requirements_llm(data["raw_text"])

    sector = tag_sector(data["raw_text"], data["title"])
    if sector == "unknown":
        sector = await tag_sector_llm(data["raw_text"], data["title"])

    tender = Tender(
        id=data.get("id", str(uuid.uuid4())),
        title=data["title"].strip(),
        issuer=data["issuer"].strip(),
        sector=sector,
        requirements=requirements,
        deadline=date.fromisoformat(data["deadline"].strip()),
        raw_text=data["raw_text"],
        source=data.get("source", "manual"),
        detected_at=data.get("detected_at", datetime.now(timezone.utc)),
        status="new",
    )

    store.save_tender(tender, dedup_hash)
    return tender, False
