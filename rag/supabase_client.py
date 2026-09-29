"""Lazy Supabase client construction with explicit configuration failures."""

from functools import lru_cache
import os
from typing import Any


class RagConfigurationError(RuntimeError):
    """Raised when required RAG configuration is absent or invalid."""


@lru_cache(maxsize=1)
def get_supabase() -> Any:
    """Return the server-side Supabase client without failing app import."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_KEY", "").strip()
    missing = [
        name
        for name, value in (("SUPABASE_URL", url), ("SUPABASE_SERVICE_KEY", key))
        if not value
    ]
    if missing:
        raise RagConfigurationError(
            "Missing required environment variable(s): " + ", ".join(missing)
        )

    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - depends on local installation
        raise RagConfigurationError(
            "supabase is not installed; run pip install -r requirements.txt"
        ) from exc

    try:
        return create_client(url, key)
    except Exception as exc:  # pragma: no cover - SDK/config dependent
        raise RagConfigurationError("Unable to initialize Supabase client") from exc


def reset_supabase_client() -> None:
    """Clear the cached client; primarily useful for tests and key rotation."""
    get_supabase.cache_clear()
