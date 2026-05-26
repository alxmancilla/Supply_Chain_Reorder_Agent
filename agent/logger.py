"""
Structured JSON logger for the Supply Chain Reorder Agent.

Usage:
    from agent.logger import get_logger
    log = get_logger(__name__)

    log.info("coverage gap computed", extra={
        "phase": "assess_alert", "sku": "MED-3017",
        "location": "DC-Texas", "coverage_gap": 45,
    })

Each log line is a single JSON object on stdout — one object per line.
Fields always present: timestamp, level, module.
Optional context fields passed via extra={}: phase, alert_id, sku, location,
event, duration_ms, tokens_used, and any arbitrary key-value pairs.

Log level is controlled by the LOG_LEVEL environment variable (default INFO).
"""

import json
import logging
import os
from datetime import datetime, timezone


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level":     record.levelname,
            "module":    record.name,
            "event":     record.getMessage(),
        }
        # Copy any extra fields the caller attached via extra={...}
        for key, value in record.__dict__.items():
            if key not in {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }:
                base[key] = value
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base)


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-formatted logger for the given module name.

    Idempotent — calling get_logger with the same name twice returns the same
    logger without adding duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    return logger
