import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from logging.handlers import TimedRotatingFileHandler
import json
# This approach doesn't rely on machine's logrotate. 
# The python instance is the one responsable for rotating this microservice's logs.


@dataclass(frozen=True) # Append-only, can't modify.
class AuditEntry:
    timestamp: str
    tool_name: str
    params: dict
    hash: str
    content: str

    def __str__(self):
        ## Ready if I needed to punch the logs into tools like ElasticSearch
        return json.dumps({
            "timestamp": self.timestamp,
            "tool_name": self.tool_name,
            "params": self.params,
            "hash": self.hash,
            "content": self.content
        }, ensure_ascii=False)


class AuditLogger:

    def __init__(self, log_path: str = "/var/log/oliveSoft/research.log",
                 logger_name: str = "audit"):
        self.log_path = log_path

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
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

    def log(self, tool_name: str, params: dict={}, raw_content: str=""):

        ## TODO: Make hash uses all audit entry json to prevent log-tampering
        entry_hash = self._sha256(raw_content)

        entry = AuditEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tool_name=tool_name,
            params=params,
            content=raw_content,
            hash=entry_hash,
        )

        self.logger.info("%sZ | INFO | %s | %s | %s", entry.timestamp, entry.tool_name, entry.params,entry.content)