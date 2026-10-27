"""
Structured logging — Milestone 8 (production hardening).

- JSON or text log format selectable via ETI_LOG_FORMAT.
- Sensitive fields (api_key, secret_key, authorization headers, passwords)
  are redacted before being written.
- User-supplied input is sanitized via eti.security.sanitize_for_log.
- Configured once at application startup.
"""
from __future__ import annotations

import json
import logging
import logging.config
import sys
from typing import Any

from .config import get_settings
from .security import sanitize_for_log

# Keys that are always redacted, regardless of where they appear.
REDACT_KEYS = {
    "api_key",
    "secret_key",
    "password",
    "passwd",
    "authorization",
    "x-api-key",
    "token",
    "access_token",
    "refresh_token",
    "credential",
}

REDACTED = "[REDACTED]"


class _RedactingFormatter(logging.Formatter):
    """Formatter that redacts sensitive keys and sanitizes user input."""

    def __init__(self, json_format: bool = False) -> None:
        super().__init__()
        self.json_format = json_format

    def _redact_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in data.items():
            klower = str(key).lower()
            if klower in REDACT_KEYS or "key" in klower or "secret" in klower or "token" in klower:
                out[key] = REDACTED
            elif isinstance(value, dict):
                out[key] = self._redact_dict(value)
            elif isinstance(value, (list, tuple)):
                out[key] = [self._redact_dict(i) if isinstance(i, dict) else sanitize_for_log(i) for i in value]
            else:
                out[key] = sanitize_for_log(value)
        return out

    def format(self, record: logging.LogRecord) -> str:
        # Build a structured payload.
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_for_log(record.getMessage()),
        }
        if record.exc_info:
            # Do not include full tracebacks in production JSON logs; just note the error type.
            payload["exception_type"] = record.exc_info[0].__name__ if record.exc_info[0] else "unknown"
            payload["exception"] = sanitize_for_log(str(record.exc_info[1]))
        if hasattr(record, "extra") and record.extra:
            payload.update(self._redact_dict(record.extra))

        if self.json_format:
            return json.dumps(payload, ensure_ascii=False)
        # Text format: level | logger | message [extra]
        extra_parts = " ".join(f"{k}={v}" for k, v in payload.items() if k not in ("level", "logger", "message"))
        return f"{payload['level']} | {payload['logger']} | {payload['message']}" + (
            f" [{extra_parts}]" if extra_parts else ""
        )


def configure_logging() -> None:
    """Configure the root logger. Call once at application startup."""
    settings = get_settings()
    json_format = settings.log_format == "json"
    formatter: logging.Formatter = _RedactingFormatter(json_format=json_format)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level, logging.INFO))

    # Reduce noise from third-party libraries unless DEBUG.
    logging.getLogger("uvicorn").setLevel(logging.WARNING if settings.log_level != "DEBUG" else logging.DEBUG)
    logging.getLogger("fastapi").setLevel(logging.WARNING if settings.log_level != "DEBUG" else logging.DEBUG)

    logging.getLogger(__name__).info("Logging configured", extra={"log_format": settings.log_format, "level": settings.log_level})
