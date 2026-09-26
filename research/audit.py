import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional
# This approach doesn't rely on machine's logrotate.
# The python instance is the one responsable for rotating this microservice's logs.

# Param keys containing one of these fragments are secrets (api_key, token,
# secret, auth header...) — their values never reach the log file in cleartext.
SENSITIVE_KEY_PARTS = ("key", "token", "secret", "auth", "password")

REDACTED = "***REDACTED***"

# Untrusted tool output is capped before logging so a huge page cannot
# blow up the audit volume; the hash still covers exactly what is stored.
CONTENT_CAP_CHARS = 2000


def _redact_params(params: dict) -> dict:
    """Return a copy of params with secret values replaced by REDACTED."""
    redacted = {}
    for k, v in (params or {}).items():
        if any(part in str(k).lower() for part in SENSITIVE_KEY_PARTS):
            redacted[k] = REDACTED
        else:
            redacted[k] = v
    return redacted


def _cap_content(raw_content: str) -> str:
    if len(raw_content) > CONTENT_CAP_CHARS:
        return raw_content[:CONTENT_CAP_CHARS] + f"...[truncated {len(raw_content) - CONTENT_CAP_CHARS} chars]"
    return raw_content


@dataclass(frozen=True) # Append-only, can't modify.
class AuditEntry:
    timestamp: str
    tool_name: str
    params: dict
    hash: str
    content: str

    def __str__(self):
        # Single source of truth for the log line. AuditLogger.log() renders
        # exactly this, so there is one serialization of an entry, not two
        # divergent ones. Ready as-is for shipping to ElasticSearch et al.
        return (
            f"{self.timestamp} | INFO | {self.tool_name} | "
            f"{json.dumps(self.params, ensure_ascii=False)} | "
            f"{self.hash} | {self.content}"
        )


class AuditLogger:

    def __init__(self, log_path: str = "/var/log/oliveSoft/research.log",
                 logger_name: str = "audit"):
        self.log_path = log_path

        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        # Unique logger per instance: a shared name broadcasts every record
        # to ALL of its handlers, so reusing a name with a new path would
        # silently duplicate (or misroute) lines into the first file.
        self.logger = logging.getLogger(f"{logger_name}.{id(self):x}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = TimedRotatingFileHandler(
            log_path,
            when="midnight",
            backupCount=365,
            utc=True,
            encoding="utf-8",
        )
        handler.suffix = "%Y-%m-%d"
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

    @staticmethod
    def _sha256(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    def log(self, tool_name: str, params: Optional[dict] = None, raw_content: str = "") -> None:

        safe_params = _redact_params(params or {})
        content = _cap_content(raw_content or "")
        # isoformat() already carries the +00:00 offset — no extra "Z" suffix.
        timestamp = datetime.now(timezone.utc).isoformat()

        # Tamper-evidence: the hash covers the full stored entry, not just
        # the raw snippet, and it is written into the log line itself.
        canonical = json.dumps({
            "timestamp": timestamp,
            "tool_name": tool_name,
            "params": safe_params,
            "content": content,
        }, sort_keys=True, ensure_ascii=False)
        entry_hash = self._sha256(canonical)

        entry = AuditEntry(
            timestamp=timestamp,
            tool_name=tool_name,
            params=safe_params,
            content=content,
            hash=entry_hash,
        )

        # Rendered by the entry itself — one serialization, not two.
        self.logger.info("%s", entry)
