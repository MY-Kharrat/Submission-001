import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException


def require_internal_token(x_internal_token: Optional[str] = Header(None)) -> str:
    """
    Enforces internal service authentication via shared secret header (X-Internal-Token).

    Why secrets.compare_digest and not ``==``: this service sits on an open
    port (likely tunneled publicly for the demo) and every non-health endpoint
    gates a *paid* LLM/search call. A plain ``==`` compares byte-by-byte and
    returns at the first mismatch, so the response time leaks how many leading
    characters of the guess were correct — enough to recover the token one
    character at a time. secrets.compare_digest is constant-time: it always
    examines the full length, so the timing carries no information about the
    guess. This is not a substitute for real auth (no rotation, no per-caller
    identity, no rate limiting) — it is a lock on a door that would otherwise
    have none, costing one function call.

    Returns HTTP 401 on missing or invalid tokens. An unset
    INTERNAL_SERVICE_TOKEN fails closed rather than authenticating everyone.
    """
    expected = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    if not x_internal_token or not expected or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Token")
    return x_internal_token
