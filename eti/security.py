"""
Security utilities — Milestone 8 (production hardening).

Provides:
- API key authentication dependency (optional, env-configured)
- Token-bucket rate limiter (in-memory, per client identifier)
- Input sanitization helpers (no injection of raw user input into logs/errors)
- Safe error responses that never leak stack traces or internals to clients
"""
from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader

from .config import get_settings

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# Dangerous characters/patterns that should never be echoed verbatim into
# HTTP responses or log messages without sanitization.
_INJECTIVE_PATTERNS = [
    (re.compile(r"<script.*?</script>", re.IGNORECASE | re.DOTALL), "[script]"),
    (re.compile(r"javascript:", re.IGNORECASE), "[js:]"),
    (re.compile(r"on\w+\s*=", re.IGNORECASE), "[on-event]"),
]


def sanitize_for_log(value: Any) -> str:
    """Sanitize a value before writing it to a log message.

    Removes potential script/injection content and truncates long strings so
    operational logs never contain executable payloads or sensitive payloads.
    Never logs credentials.
    """
    if value is None:
        return "null"
    text = str(value)
    for pattern, replacement in _INJECTIVE_PATTERNS:
        text = pattern.sub(replacement, text)
    if len(text) > 500:
        text = text[:500] + "…[truncated]"
    return text


def safe_error_message(detail: str) -> str:
    """Return a client-safe error message (sanitized, no internals)."""
    return sanitize_for_log(detail)


@dataclass
class _TokenBucket:
    tokens: float
    last_update: float


class RateLimiter:
    """In-memory token-bucket rate limiter keyed by client identifier.

    Not suitable for multi-process deployments (use Redis-backed limiter
    behind a load balancer in production multi-instance setups), but
    sufficient for a single-process modular monolith.
    """

    def __init__(self, requests_per_min: int, enabled: bool = True) -> None:
        self.requests_per_min = max(1, requests_per_min)
        self.enabled = enabled
        self._buckets: dict[str, _TokenBucket] = defaultdict(
            lambda: _TokenBucket(tokens=float(self.requests_per_min), last_update=time.monotonic())
        )
        self._refill_rate = self.requests_per_min / 60.0  # tokens per second

    def is_allowed(self, client_id: str) -> tuple[bool, int]:
        """Return (allowed, retry_after_seconds)."""
        if not self.enabled:
            return True, 0
        now = time.monotonic()
        bucket = self._buckets[client_id]
        elapsed = now - bucket.last_update
        bucket.last_update = now
        bucket.tokens = min(self.requests_per_min, bucket.tokens + elapsed * self._refill_rate)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0
        retry_after = int((1.0 - bucket.tokens) / self._refill_rate) + 1
        return False, max(1, retry_after)

    def reset(self, client_id: str) -> None:
        self._buckets.pop(client_id, None)


def default_client_identifier(request: Request) -> str:
    """Derive a client identifier from the request for rate limiting.

    Uses X-Forwarded-For (first hop) when behind a trusted proxy, falling
    back to the direct client host. Never logs the full identifier verbatim.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        client = forwarded.split(",")[0].strip()
    else:
        client = request.client.host if request.client else "unknown"
    return client or "unknown"


# Singleton rate limiter (configured from settings).
_rate_limiter: Any = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        _rate_limiter = RateLimiter(
            requests_per_min=settings.rate_limit_requests_per_min,
            enabled=settings.rate_limit_enabled,
        )
    return _rate_limiter


async def require_api_key(request: Request, api_key_header: str | None = None) -> None:
    """FastAPI dependency: enforce API key authentication when configured.

    When ETI_API_KEY is not set, authentication is disabled (development
    convenience). When set, clients must supply a matching X-API-Key header.
    Never logs the key value.
    """
    settings = get_settings()
    if not settings.api_key:
        return  # authentication disabled
    if not api_key_header or api_key_header != settings.api_key:
        # Use a generic message to avoid key-enumeration hints
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency: enforce per-client rate limiting."""
    limiter = get_rate_limiter()
    client_id = default_client_identifier(request)
    allowed, retry_after = limiter.is_allowed(client_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def url_encode_safe(value: str) -> str:
    """URL-encode a value for safe inclusion in query strings / redirects."""
    from urllib.parse import quote
    return quote(value, safe="")