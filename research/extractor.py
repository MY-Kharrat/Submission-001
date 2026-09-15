import json
import re
from typing import Optional

from detection.llm import llm_call
from shared.schemas import ExtractorOutput

SYSTEM_PROMPT = (
    "You are a quarantined fact extractor. Read the single provided text and "
    "output ONLY a JSON object matching {fact, relevance, source_url} or null "
    "if no relevant fact exists. relevance must be one of: sector, "
    "estimated_revenue, past_projects, key_partners. No other output allowed."
)


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


class Extractor:
    async def extract(self, content: str, source_url: str) -> Optional[ExtractorOutput]:
        prompt = (
            f"{SYSTEM_PROMPT}\n\nSource URL: {source_url}\n"
            f"Text:\n{content[:4000]}\n\nJSON or null:"
        )
        raw = (await llm_call(prompt)).strip()
        if raw.lower() == "null" or not raw:
            return None
        cleaned = _strip_fences(raw)
        if cleaned.lower() == "null":
            return None
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return None
        if data is None:
            return None
        if not isinstance(data, dict):
            return None
        data.setdefault("source_url", source_url)
        try:
            return ExtractorOutput(**data)
        except Exception:
            return None
