"""Backwards-compat shim: prompts live in research.prompts (plural)."""
from .prompts import EXTRACTOR_SYSTEM_PROMPT, ORCHESTRATOR_SYSTEM_PROMPT, SNIPPET_CAP_CHARS

__all__ = ["EXTRACTOR_SYSTEM_PROMPT", "ORCHESTRATOR_SYSTEM_PROMPT", "SNIPPET_CAP_CHARS"]
