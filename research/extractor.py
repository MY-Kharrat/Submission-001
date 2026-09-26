import json
import re
from typing import Optional

from detection.llm import llm_call
from shared.schemas import ExtractorOutput
from .prompts import EXTRACTOR_SYSTEM_PROMPT, SNIPPET_CAP_CHARS

FACT_CAP = 8

def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()

class Extractor:
    async def extract(self, content: str, source_url: Optional[str] = None) -> ExtractorOutput:
        # Quarantine: trusted instructions travel via system=, the untrusted
        # snippet (capped) via the user message — never concatenated.
        try:
            # Cap the untrusted snippet before it reaches the model, so a huge
            # page cannot blow the context window or smuggle extra instructions
            # past review.
            capped = content[:SNIPPET_CAP_CHARS]
            raw = (await llm_call(f"Source text:\n{capped}\n\nJSON:",
                                  system=EXTRACTOR_SYSTEM_PROMPT)).strip()
        except Exception as e:
            print(f"Error[Extractor]: {e}")
            return ExtractorOutput(facts=[])
        
        if not raw or raw.lower() == "null":
            return ExtractorOutput(facts=[])
        
        cleaned = _strip_fences(raw)
        if not cleaned or cleaned.lower() == "null":
            return ExtractorOutput(facts=[])
        
        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            return ExtractorOutput(facts=[])
        
        if data is None:
            return ExtractorOutput(facts=[])
        
        if not isinstance(data, dict):
            return ExtractorOutput(facts=[])

        try:
            return ExtractorOutput(**data)
        except Exception:
            return ExtractorOutput(facts=[])
