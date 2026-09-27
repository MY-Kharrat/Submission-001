"""The internal-token auth boundary.

Every non-health endpoint gates a paid LLM/search call, and the service is
reachable on an open port. These tests pin the properties that make the lock
worth having, particularly the one that fails *closed*.
"""

import inspect

import pytest
from fastapi import HTTPException

from shared import auth


def _call(token, expected):
    """Invoke the dependency the way FastAPI would, with a configured env."""
    import os

    prev = os.environ.get("INTERNAL_SERVICE_TOKEN")
    if expected is None:
        os.environ.pop("INTERNAL_SERVICE_TOKEN", None)
    else:
        os.environ["INTERNAL_SERVICE_TOKEN"] = expected
    try:
        return auth.require_internal_token(token)
    finally:
        if prev is None:
            os.environ.pop("INTERNAL_SERVICE_TOKEN", None)
        else:
            os.environ["INTERNAL_SERVICE_TOKEN"] = prev


def test_correct_token_is_accepted():
    assert _call("s3cret-value", "s3cret-value") == "s3cret-value"


@pytest.mark.parametrize("presented", [None, "", "wrong", "s3cret-valu", "s3cret-value "])
def test_missing_or_wrong_token_is_rejected(presented):
    with pytest.raises(HTTPException) as exc:
        _call(presented, "s3cret-value")
    assert exc.value.status_code == 401
    assert exc.value.detail == "Invalid or missing X-Internal-Token"


def test_unset_server_token_fails_closed():
    """An unset INTERNAL_SERVICE_TOKEN must reject everyone, not allow all.

    This is the failure that would otherwise be silent and catastrophic: with
    no token configured, a naive `if expected == token` would authenticate any
    caller presenting an empty header against an unset default.
    """
    for presented in (None, "", "anything", "s3cret-value"):
        with pytest.raises(HTTPException) as exc:
            _call(presented, None)
        assert exc.value.status_code == 401


def test_token_comparison_is_constant_time():
    """Guards the timing side-channel fix from a well-meaning simplification.

    Rewriting `secrets.compare_digest(a, b)` as `a == b` is a one-character
    change that no functional test would notice, while reintroducing a timing
    oracle that leaks the token one character at a time. The property is not
    observable from the return value, so this asserts on the source.
    """
    src = inspect.getsource(auth.require_internal_token)
    assert "compare_digest" in src, (
        "require_internal_token no longer uses a constant-time comparison"
    )
    assert "== token" not in src and "token ==" not in src, (
        "a plain == comparison reintroduces a timing oracle"
    )
