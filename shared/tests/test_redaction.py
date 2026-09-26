"""Secret scrubbing of log records.

The guarantee under test is that no credential reaches a log stream, including
via a provider exception whose rendering we do not control.
"""

import logging
from unittest.mock import patch

from shared.redaction import REDACTED, SecretScrubber, scrub

SECRET = "sk-live-abcdef0123456789XYZ"


def _record(msg="failed", args=None, exc_info=None):
    return logging.LogRecord(
        name="research.extractor", level=logging.WARNING, pathname=__file__,
        lineno=1, msg=msg, args=args, exc_info=exc_info,
    )


def test_scrub_replaces_a_known_secret():
    with patch.dict("os.environ", {"LLM_API_KEY": SECRET}):
        assert scrub(f"key={SECRET} end") == f"key={REDACTED} end"


def test_scrub_leaves_text_without_secrets_untouched():
    text = "extracted 3 facts for tender 42"
    with patch.dict("os.environ", {"LLM_API_KEY": SECRET}):
        assert scrub(text) == text


def test_short_env_values_are_not_treated_as_secrets():
    """A placeholder token must not blank out ordinary log text."""
    text = "retrying (dev build, x attempts)"
    with patch.dict("os.environ", {"INTERNAL_SERVICE_TOKEN": "x", "LLM_API_KEY": "dev"}):
        assert scrub(text) == text


def test_scrub_covers_every_configured_secret_var():
    for var in ("LLM_API_KEY", "TAVILY_API_KEY", "INTERNAL_SERVICE_TOKEN"):
        with patch.dict("os.environ", {var: SECRET}):
            assert scrub(SECRET) == REDACTED, f"{var} was not scrubbed"


def test_filter_scrubs_interpolated_arguments():
    rec = _record(msg="calling api with %s", args=(f"x-api-key={SECRET}",))
    with patch.dict("os.environ", {"LLM_API_KEY": SECRET}):
        SecretScrubber().filter(rec)
    assert SECRET not in rec.getMessage()
    assert REDACTED in rec.getMessage()


def test_filter_scrubs_the_rendered_traceback_and_drops_raw_exception():
    """The traceback is the interesting vector, and the hardest to control.

    The filter must pre-render it scrubbed *and* clear exc_info, because
    logging.Formatter falls back to formatting exc_info whenever exc_text is
    empty -- which would undo the scrub.
    """
    try:
        raise RuntimeError(f"auth failed, x-api-key={SECRET}")
    except RuntimeError:
        import sys

        rec = _record(exc_info=sys.exc_info())

    with patch.dict("os.environ", {"LLM_API_KEY": SECRET}):
        SecretScrubber().filter(rec)
    assert rec.exc_info is None, "raw exception left available for re-rendering"
    assert SECRET not in (rec.exc_text or "")

    # What a real handler would print, via the real formatting path.
    printed = logging.Formatter("%(message)s").format(rec)
    assert SECRET not in printed
    assert REDACTED in printed


def test_filter_is_idempotent():
    rec = _record(msg=f"token {SECRET}")
    scrubber = SecretScrubber()
    with patch.dict("os.environ", {"LLM_API_KEY": SECRET}):
        scrubber.filter(rec)
    once = rec.getMessage()
    scrubber.filter(rec)
    assert rec.getMessage() == once
