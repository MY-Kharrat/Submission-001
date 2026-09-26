"""Secret scrubbing for log records.

Logging a provider failure with ``exc_info=True`` is worth it -- a traceback is
the difference between "auth is broken" and "the pool is timing out". But the
traceback is rendered from a third-party exception, and rendering is not under
our control: httpx, or any future provider client, is free to put the request
that failed into that string, and a failed request carries the credential in
its headers. Some providers also authenticate by query parameter, so even the
URL on the line above the traceback can hold the key.

Relying on httpx happening to keep headers out of ``str(exc)`` is an accident,
not a control. This module makes the guarantee explicit instead, by scrubbing
the fully rendered text -- message, arguments and traceback together -- before
any handler sees it.

Scope note: this is defence in depth, not the primary control. Secrets are
still kept out of messages by not putting them there, and the audit log
redacts sensitive *parameter names* separately. This closes the path where a
library we do not control decides to render a credential.
"""

import logging
import os

REDACTED = "***REDACTED***"

# Only values read by this codebase. A provider client that reads its own
# credential internally never exposes it to us, so it cannot reach a log.
SECRET_ENV_VARS = ("LLM_API_KEY", "TAVILY_API_KEY", "INTERNAL_SERVICE_TOKEN")

# Below this length a "secret" is almost certainly a placeholder ("x", "dev",
# "ci-token"). Scrubbing those would blank out ordinary log text, so short
# values are skipped. Real credentials are far longer than 8 characters.
MIN_SECRET_LEN = 8


def _active_secrets() -> list:
    """Secret values currently in the environment, long enough to be specific.

    Read per call rather than cached at import: keys get rotated without a
    restart, and a test that sets one should be covered immediately.
    """
    found = []
    for var in SECRET_ENV_VARS:
        value = os.environ.get(var)
        if value and len(value) >= MIN_SECRET_LEN:
            found.append(value)
    return found


def scrub(text: str) -> str:
    """Replace any known secret value found in ``text`` with REDACTED."""
    for secret in _active_secrets():
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


class SecretScrubber(logging.Filter):
    """Scrub a record's rendered output, traceback included.

    Attached to the *logger*, not a handler, so the scrubbing happens before
    records are handed to any handler at all -- including the handler pytest's
    ``caplog`` installs, which is exactly how a leak would otherwise slip past
    the test that is supposed to be guarding it.

    The traceback is pre-rendered into ``record.exc_text`` (scrubbed) and
    ``record.exc_info`` is then cleared. ``logging.Formatter`` consults
    ``exc_text`` first and only falls back to ``exc_info``, so every handler
    downstream prints the scrubbed text and nothing can re-render the raw
    exception.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # getMessage() applies %-args, so args are already folded in and are
        # cleared afterwards to avoid them being interpolated a second time.
        record.msg = scrub(record.getMessage())
        record.args = ()

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = scrub(self._format_exception(record))
            # Always cleared, never conditionally: leaving exc_info set would
            # let any handler re-render the raw exception, and would undo the
            # scrub even when exc_text was already populated.
            record.exc_info = None
        return True

    @staticmethod
    def _format_exception(record: logging.LogRecord) -> str:
        return "".join(
            logging.Formatter().formatException(record.exc_info)
        )
