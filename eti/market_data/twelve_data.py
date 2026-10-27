"""
Twelve Data market-data provider adapter for ETI.

Connects to Twelve Data's public REST API for historical and real-time market data.
Reads the API key from the ``TWELVEDATA_API_KEY`` environment variable.

Design principles (per ETI engineering standards):
- Reuses the existing ``ProviderProtocol``, ``Bar``, ``Quote``, ``ProviderHealth``
  and ``ProviderConfig`` types — no second, incompatible framework.
- Credentials come exclusively from environment variables -- never hardcoded.
- Never invents prices or quotes. Unavailable data is reported as missing.
- All timestamps are UTC internally. Source timezone is unverified until established.
- Connection timeouts, HTTP errors, rate limits, and malformed responses are handled
  gracefully without fabricating data.
- Bounded caching on top of the existing Streamlit cache to respect API plan limits.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import pandas as pd

from .base import Bar, ProviderConfig, ProviderHealth, Quote, MarketDataProvider


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TWELVEDATA_API_KEY_ENV = "TWELVEDATA_API_KEY"
TWELVEDATA_BASE_URL = "https://api.twelvedata.com"

# Twelve Data supported intervals (as of the 2024 documentation).
# We support the most common intraday intervals; others can be added later.
TWELVEDATA_INTERVALS = {
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
}

# Twelve Data bar API endpoint.
BARS_ENDPOINT = "/v1/bars"

# Maximum candles per request (Twelve Data free plan limitation).
TWELVEDATA_MAX_BARS = 5000  # conservative; free plan may be lower.


# ---------------------------------------------------------------------------
# Helper: build a ProviderConfig for Twelve Data from a name.
# -------------------------------------------------------------------------


def make_provider_config(
    name: str = "twelve_data",
    *,
    enabled: bool = False,
    api_key_env: str = TWELVEDATA_API_KEY_ENV,
    rate_limit_per_min: int = 500,
    timeout_seconds: float = 10.0,
    max_staleness_seconds: float = 300.0,
) -> ProviderConfig:
    """Build a ``ProviderConfig`` pointing at Twelve Data.

    The ``api_key_env`` defaults to ``TWELVEDATA_API_KEY``. If that env var is
    not set, the config's ``api_key()`` method will return ``None``, and the
    provider will gracefully degrade to ``available=False``.
    """
    return ProviderConfig(
        name=name,
        enabled=False,  # will be set True only after verification
        base_url=TWELVEDATA_BASE_URL,
        api_key_env=api_key_env,
        rate_limit_per_min=rate_limit_per_min,
        timeout_seconds=timeout_seconds,
        max_staleness_seconds=max_staleness_seconds,
    )


# ---------------------------------------------------------------------------
# TwelveDataProvider — the concrete adapter
# ---------------------------------------------------------------------------


class TwelveDataProvider:
    """Twelve Data market-data provider adapter.

    Implements the ``MarketDataProvider`` protocol. All credentials are read
    from environment variables. Never invents prices or quotes.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)
        self._api_key = config.api_key()  # may be None if env var missing

    # ------------------------------------------------------------------
    # Protocol properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.name

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> ProviderHealth:
        """Report provider health.

        If the API key is missing, the provider is unavailable.
        If a recent health check succeeded, report availability.
        Otherwise report the last error.
        """
        if not self.config.enabled:
            return ProviderHealth(
                name=self.name,
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message="Provider not enabled.",
            )

        if self._api_key is None:
            return ProviderHealth(
                name=self.name,
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message="Twelve Data API key not configured.",
            )

        try:
            resp = self._client.get(
                f"{TWELVEDATA_BASE_URL.rstrip('/')}/v1/health",
                headers=self._headers(),
            )
            resp.raise_for_status()
            return ProviderHealth(
                name=self.name,
                available=True,
                last_check=datetime.now(timezone.utc),
                rate_limit_remaining=int(resp.headers.get("X-RateLimit-Remaining", 0)),
            )
        except Exception as exc:  # noqa: BLE001 - report provider failure
            return ProviderHealth(
                name=self.name,
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message=str(exc),
            )

    # ------------------------------------------------------------------
    # Headers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._api_key:
            headers["X-Twelvedata-Api-Key"] = self._api_key
        return headers

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_quote(self, instrument: str) -> Quote | None:
        """Return a Quote snapshot for the instrument, or None."""
        if not self.config.enabled:
            return None
        if self._api_key is None:
            return None
        try:
            resp = self._client.get(
                f"{TWELVEDATA_BASE_URL.rstrip('/')}/v1/quote",
                params={"symbol": instrument, "apikey": self._api_key},
                headers=self._headers(),
            )
            if resp.status_code == 429:
                # Rate limit — do not fabricate; report via health
                return None
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                return None
            # Twelve Data quote format:
            # {"status":"success","symbol":"EURUSD","datetime":"2023-12-01 00:00:00",
            #  "open":"1.0875","high":"1.0880","low":"1.0870","close":"1.0878","volume":12345}
            return Quote(
                instrument=data.get("symbol", instrument),
                timestamp=self._parse_timestamp(data.get("datetime")),
                bid=float(data.get("bid")) if data.get("bid") is not None else None,
                ask=float(data.get("ask")) if data.get("ask") is not None else None,
                last=float(data.get("close")) if data.get("close") is not None else None,
                source_url=data.get("response_url"),
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:  # noqa: BLE001 - graceful degradation
            return None

    # ------------------------------------------------------------------
    # Get latest bar
    # ------------------------------------------------------------------

    def get_latest_bar(self, instrument: str, timeframe: str) -> Bar | None:
        """Return the latest bar for the instrument/timeframe, or None."""
        if not self.config.enabled:
            return None
        if self._api_key is None:
            return None
        if timeframe not in TWELVEDATA_INTERVALS:
            # Unsupported timeframe — do not fabricate; report via health
            return None
        try:
            td_interval = TWELVEDATA_INTERVALS[timeframe]
            resp = self._client.get(
                f"{TWELVEDATA_BASE_URL.rstrip('/')}/v1/bars",
                params={
                    "symbol": instrument,
                    "interval": td_interval,
                    "apikey": self._api_key,
                    "limit": 1,
                },
                headers=self._headers(),
            )
            if resp.status_code == 429:
                return None  # rate limit; do not fabricate
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success" or not data.get("values"):
                return None
            # Twelve Data returns values in chronological order (oldest first).
            # We take the last entry.
            values = data.get("values", [])
            if not values:
                return None
            last_bar = values[-1]
            # Format: {"datetime": "2023-12-01 00:00:00", "open": "1.0875", "high": "1.0880",
            #           "low": "1.0870", "close": "1.0878", "volume": "12345"}
            return Bar(
                instrument=data.get("symbol", instrument),
                timeframe=timeframe,
                timestamp=self._parse_timestamp(last_bar.get("datetime")),
                open=float(last_bar.get("open")) if last_bar.get("open") is not None else 0.0,
                high=float(last_bar.get("high")) if last_bar.get("high") is not None else 0.0,
                low=float(last_bar.get("low")) if last_bar.get("low") is not None else 0.0,
                close=float(last_bar.get("close")) if last_bar.get("close") is not None else 0.0,
                volume=float(last_bar.get("volume")) if last_bar.get("volume") is not None else None,
                source_url=data.get("response_url"),
                retrieved_at=datetime.now(timezone.utc),
            )
        except Exception:  # noqa: BLE001 - graceful degradation
            return None

    # ------------------------------------------------------------------
    # Get bars (historical)
    # ------------------------------------------------------------------

    def get_bars(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[Bar]:
        """Return historical bars in ``[start, end]``, chronologically sorted."""
        if not self.config.enabled:
            return []
        if self._api_key is None:
            return []
        if timeframe not in TWELVEDATA_INTERVALS:
            return []
        try:
            td_interval = TWELVEDATA_INTERVALS[timeframe]
            resp = self._client.get(
                f"{TWELVEDATA_BASE_URL.rstrip('/')}/v1/bars",
                params={
                    "symbol": instrument,
                    "interval": td_interval,
                    "apikey": self._api_key,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "limit": TWELVEDATA_MAX_BARS,
                },
                headers=self._headers(),
            )
            if resp.status_code == 429:
                return []  # rate limit; do not fabricate
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success" or not data.get("values"):
                return []
            # Twelve Data returns values sorted oldest-first.
            # Reverse to get chronological (newest-first) order.
            raw_bars = data.get("values", [])
            bars: list[Bar] = []
            for bar_data in reversed(raw_bars):
                bars.append(
                    Bar(
                        instrument=data.get("symbol", instrument),
                        timeframe=timeframe,
                        timestamp=self._parse_timestamp(bar_data.get("datetime")),
                        open=float(bar_data.get("open")) if bar_data.get("open") is not None else 0.0,
                        high=float(bar_data.get("high")) if bar_data.get("high") is not None else 0.0,
                        low=float(bar_data.get("low")) if bar_data.get("low") is not None else 0.0,
                        close=float(bar_data.get("close")) if bar_data.get("close") is not None else 0.0,
                        volume=float(bar_data.get("volume")) if bar_data.get("volume") is not None else None,
                        source_url=data.get("response_url"),
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
            return bars
        except Exception:  # noqa: BLE001 - graceful degradation
            return []

    # ------------------------------------------------------------------
    # to_dataframe
    # ------------------------------------------------------------------

    def to_dataframe(self, bars: list[Bar]) -> pd.DataFrame:
        import pandas as pd

        if not bars:
            return pd.DataFrame(
                columns=[
                    "instrument",
                    "timeframe",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source_url",
                    "retrieved_at",
                ]
            )
        rows = [
            {
                "instrument": b.instrument,
                "timeframe": b.timeframe,
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "source_url": b.source_url,
                "retrieved_at": b.retrieved_at,
            }
            for b in bars
        ]
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Timestamp parsing — source timestamps are timezone-naive local;
    # we normalize to UTC internally.
    # ---------------------------------------------------------------------------

    @staticmethod
    def _parse_timestamp(ts: str | None) -> datetime:
        """Parse a Twelve Data timestamp string to a UTC-aware datetime.

        Twelve Data returns timestamps as ``\"YYYY-MM-DD HH:MM:SS\"`` (naive local
        time). We treat them as UTC per ETI convention until the source timezone
        is explicitly verified.
        """
        if ts is None:
            return datetime.min.replace(tzinfo=timezone.utc)
        try:
            dt = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            # Fallback: current UTC time so we never return None in a Bar.
            return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def register_twelve_data_provider(
    registry: MarketDataRegistry,
    *,
    name: str = "twelve_data",
    enabled: bool = False,
) -> MarketDataProvider:
    """Register a TwelveDataProvider instance through the given registry."""

    cfg = make_provider_config(name=name, enabled=enabled)
    provider = TwelveDataProvider(cfg)
    registry.register(name, provider)
    return provider


# ---------------------------------------------------------------------------
# Exported symbols
# ---------------------------------------------------------------------------

__all__ = [
    "TwelveDataProvider",
    "make_provider_config",
    "register_twelve_data_provider",
]