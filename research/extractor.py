import json
import logging
import re
from typing import Optional

from pydantic import ValidationError

from detection.llm import llm_call
from shared.redaction import SecretScrubber
from shared.schemas import ExtractorOutput
from .prompts import EXTRACTOR_SYSTEM_PROMPT, SNIPPET_CAP_CHARS

logger = logging.getLogger(__name__)

# On the logger, not a handler, so records are scrubbed before any handler
# sees them. This is the logger that logs exc_info for a failed provider call,
# and that traceback is rendered from a provider exception we do not control.
if not any(isinstance(f, SecretScrubber) for f in logger.filters):
    logger.addFilter(SecretScrubber())

# A single page cannot plausibly support more than a handful of facts. Bounding
# this keeps one verbose or hijacked page from bloating the research object and
# the audit trail.
FACT_CAP = 8

# Enough of a malformed reply to diagnose contract drift, not enough to flood
# the log with page text.
_REPLY_LOG_CHARS = 200


def _log_reply(reason: str, raw: str) -> None:
    logger.warning(
        "Extractor contract violation (%s); reply was: %r",
        reason,
        raw[:_REPLY_LOG_CHARS],
    )


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
        except Exception:
            # Transport/auth/timeout: distinct from a contract violation, and
            # worth an operator's attention.
            logger.warning("Extractor LLM call failed", exc_info=True)
            return ExtractorOutput(facts=[])

        if not raw or raw.lower() == "null":
            # Ambiguous: legitimately no facts, or an empty reply. Low signal.
            logger.debug("Extractor got an empty reply")
            return ExtractorOutput(facts=[])

        cleaned = _strip_fences(raw)
        if not cleaned or cleaned.lower() == "null":
            logger.debug("Extractor got an empty reply after fence stripping")
            return ExtractorOutput(facts=[])

        try:
            data = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            _log_reply("not valid JSON", raw)
            return ExtractorOutput(facts=[])

        if data is None:
            logger.debug("Extractor reply was JSON null")
            return ExtractorOutput(facts=[])

        if not isinstance(data, dict):
            _log_reply(f"top-level {type(data).__name__}, expected object", raw)
            return ExtractorOutput(facts=[])

        try:
            out = ExtractorOutput(**data)
        except ValidationError as exc:
            # This is the signal that matters: ExtractorOutput forbids extra
            # keys, so a model that starts adding e.g. "summary" or "notes"
            # fails closed and yields zero facts. That is the safe direction,
            # but it is indistinguishable from "no facts found" unless it is
            # logged — which is exactly the drift we need to see before it
            # silently empties real research in production.
            extra = sorted(
                {
                    str(e["loc"][0])
                    for e in exc.errors()
                    if e["type"] == "extra_forbidden" and e["loc"]
                }
            )
            if extra:
                logger.warning(
                    "Extractor reply carried forbidden extra key(s) %s; "
                    "facts discarded (fail-closed)",
                    extra,
                )
            else:
                _log_reply("schema mismatch", raw)
            return ExtractorOutput(facts=[])

        if len(out.facts) > FACT_CAP:
            logger.info(
                "Extractor returned %d facts; capping at FACT_CAP=%d",
                len(out.facts),
                FACT_CAP,
            )
            out = ExtractorOutput(facts=out.facts[:FACT_CAP])

        return out
