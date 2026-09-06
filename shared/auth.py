import os
import secrets
from typing import Optional

from fastapi import Header, HTTPException
#TODO secrets.compare_digest choosed over == i will add comments explaining

def require_internal_token(x_internal_token: Optional[str] = Header(None)) -> str:
    """
    Enforces internal service authentication via shared secret header (X-Internal-Token).
    Uses constant-time comparison (secrets.compare_digest) to prevent timing side-channels.
    Returns HTTP 401 on missing or invalid tokens.
    """
    expected = os.environ.get("INTERNAL_SERVICE_TOKEN", "")
    if not x_internal_token or not expected or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing X-Internal-Token")
    return x_internal_token
