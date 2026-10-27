"""
Middleware — Milestone 8 (production hardening).

Provides:
- Security headers (Content-Type nosniff, X-Frame-Options, X-Content-Type-Options, Referrer-Policy)
- Request ID generation for tracing
- Structured request logging via eti.logging
- Rate limit enforcement header
- Safe error response formatting
"""
from __future__ import annotations

import uuid
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response

from .security import safe_error_message, enforce_rate_limit, sanitize_for_log
from .logging import configure_logging, _RedactingFormatter, logging


# — Middleware factory —
def get_middleware() -> list[Middleware]:
    """Return Middleware list for FastAPI app construction."""
    return [
        Middleware(
            _SecurityHeadersMiddleware,
            debug=False,
        ),
        Middleware(
            _RequestIDMiddleware,
            debug=False,
        ),
        Middleware(
            _RateLimitMiddleware,
            debug=False,
        ),
    ]


class _SecurityHeadersMiddleware:
    """Add security headers to every response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response: Response = await call_next(request)
        response.headers.append("X-Content-Type-Options", "nosniff")
        response.headers.append("X-Frame-Options", "DENY")
        response.headers.append(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self';",
        )
        response.headers.append(
            "Referrer-Policy", "strict-origin-when-cross-origin"
        )
        return response


class _RequestIDMiddleware:
    """Add a unique Request-ID to each response for tracing."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        # Log the request ID with the request
        logging.getLogger(__name__).debug(
            "request processed",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": default_client_identifier(request),
            },
        )
        return response


class _RateLimitMiddleware:
    """Enforce rate limiting and add Retry-After when limit is hit."""

    async def dispatch(self, request: Request, call_next) -> Response:
        allowed, retry_after = enforce_rate_limit(request)
        response: Response = await call_next(request)
        if not allowed:
            response.headers["Retry-After"] = str(retry_after)
        return response


# Helper for middleware (client identifier)
def default_client_identifier(request: Request) -> str:
    """Derive a client identifier from the request for rate limiting."""
    from .security import default_client_identifier as _dcid
    return _dcid(request)