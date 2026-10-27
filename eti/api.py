"""
API — Milestone 8 (production hardening).

FastAPI application with:
- Optional API key authentication (ETI_API_KEY env-configured).
- Per-client rate limiting (token bucket).
- Safe error responses that sanitize details via eti.security.safe_error_message.
- Health check that reports configuration validity.
- All endpoints guarded by rate limiting.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.security import APIKeyHeader

from .config import get_settings, validate_startup_config
from .security import (
    API_KEY_HEADER,
    enforce_rate_limit,
    require_api_key,
    safe_error_message,
)
from .logging import configure_logging

APP_DIR = Path(__file__).resolve().parents[1]
MINUTE_PATH = Path(os.getenv("ETI_MINUTE_CSV", APP_DIR / "data/raw/EURUSD1.csv"))
DAILY_PATH = Path(os.getenv("ETI_DAILY_CSV", APP_DIR / "data/raw/EURUSD1440.csv"))

app = FastAPI(
    title="Egreed Trade Intelligence API",
    version="0.1.0",
    description=(
        "Historical EUR/USD research prototype. Not a live trading service. "
        "All prices are historical, not live quotes."
    ),
)

# ---------------------------------------------------------------------------
# Startup: configure logging and report config warnings
# ---------------------------------------------------------------------------
configure_logging()

startup_warnings = validate_startup_config()
if startup_warnings:
    import logging as _logging
    for w in startup_warnings:
        _logging.getLogger(__name__).warning(w)


# ---------------------------------------------------------------------------
# Dependency: API key authentication (no-op when ETI_API_KEY not set)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def api_key_auth_middleware(request: Request, call_next):
    """Enforce API key when ETI_API_KEY is configured; no-op otherwise."""
    if not __import__("eti.config").get_settings().api_key:
        # No API key configured; skip authentication
        return await call_next(request)
    api_key_name = API_KEY_HEADER.name
    api_key_value = request.headers.get(api_key_name)
    if not await require_api_key(request, api_key_value):  # type: ignore
        return  # require_api_key already raises HTTPException
    return await call_next(request)


# ---------------------------------------------------------------------------
# Dependency: rate limiting (no-op when disabled)
# ---------------------------------------------------------------------------
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next) -> Response:
    """Enforce rate limiting when configured; no-op otherwise."""
    settings = __import__("eti.config").get_settings()
    if not settings.rate_limit_enabled:
        return await call_next(request)
    return await enforce_rate_limit(request)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health", include_in_schema=False)
def health():
    """Health check endpoint.

    Returns service status and selected config warnings.
    """
    settings = get_settings()
    warnings = validate_startup_config()
    return {
        "status": "ok" if not warnings else "degraded",
        "service": "eti-api",
        "version": "0.1.0",
        "environment": settings.app_env,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
@app.get("/api/summary")
def api_summary(timeframe: str = Query("1d", pattern="^(1m|15m|1h|4h|1d)$")):
    df = get_dataset(timeframe)
    return summary(df, f"EUR/USD {timeframe}")


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------
@app.get("/api/bars")
def api_bars(timeframe: str = Query("1d", pattern="^(1m|15m|1h|4h|1d)$"),
             limit: int = Query(500, ge=1, le=5000)):
    df = add_indicators(get_dataset(timeframe)).tail(limit).copy()
    df = df.where(df.notna(), None)
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
@app.get("/api/backtest")
def api_backtest(
    timeframe: str = Query("1d", pattern="^(1m|15m|1h|4h|1d)$"),
    fast: int = Query(20, ge=2, le=200),
    slow: int = Query(50, ge=3, le=500),
    cost_bps: float = Query(1.0, ge=0, le=1000),
):
    try:
        return sma_crossover_backtest(get_dataset(timeframe), fast, slow, cost_bps)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=safe_error_message(str(exc)),
        ) from exc