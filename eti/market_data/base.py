"""
Live market-data provider adapters.

Design principles (per ETI engineering standards):
- Providers are replaceable through a common adapter interface.
- No provider is connected by default. The default/active provider is a
  ``NullMarketDataProvider`` that returns no live data (graceful degradation).
- Credentials come exclusively from environment variables -- never hardcoded.
- All timestamps are timezone-aware (UTC) internally. Source timezone and
  candle conventions remain unverified until explicitly established.
- Providers never invent prices or quotes. Unavailable/stale data is reported
  as missing, never fabricated.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Quote:
    """A single live quote snapshot.

    All fields are required; nothing is invented. ``bid``/``ask`` may be None
    if the provider only publishes a last/trade price.
    """
    instrument: str
    timestamp: datetime  # timezone-aware UTC
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    source_url: str | None = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return self.last

    def is_stale(self, max_age_seconds: float) -> bool:
        age = (datetime.now(timezone.utc) - self.retrieved_at).total_seconds()
        return age > max_age_seconds


@dataclass(frozen=True)
class Bar:
    """A single OHLCV bar from a live data source."""
    instrument: str
    timeframe: str  # e.g. "1m", "1h", "1d"
    timestamp: datetime  # timezone-aware UTC (bar open / left-closed boundary)
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None
    source_url: str | None = None
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_stale(self, max_age_seconds: float) -> bool:
        age = (datetime.now(timezone.utc) - self.retrieved_at).total_seconds()
        return age > max_age_seconds


@dataclass(frozen=True)
class ProviderHealth:
    """Provider availability / health status."""
    name: str
    available: bool
    last_check: datetime
    error_message: str | None = None
    rate_limit_remaining: int | None = None
    rate_limit_reset: datetime | None = None


@dataclass(frozen=True)
class ProviderConfig:
    """Configuration for a market-data provider.

    Secrets (api_key, secret) must be supplied via environment variables at
    runtime, never embedded here.
    """
    name: str
    enabled: bool = False
    base_url: str | None = None
    api_key_env: str | None = None  # name of env var holding the key
    secret_env: str | None = None
    rate_limit_per_min: int | None = None
    timeout_seconds: float = 10.0
    max_staleness_seconds: float = 300.0
    extra: dict = field(default_factory=dict)

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)

    def secret(self) -> str | None:
        if not self.secret_env:
            return None
        return os.environ.get(self.secret_env)


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------


@runtime_checkable
class MarketDataProvider(Protocol):
    """Protocol every live market-data provider must implement."""

    name: str
    config: ProviderConfig

    def health(self) -> ProviderHealth: ...
    def get_quote(self, instrument: str) -> Quote | None: ...
    def get_latest_bar(
        self, instrument: str, timeframe: str
    ) -> Bar | None: ...
    def get_bars(
        self, instrument: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]: ...
    def to_dataframe(self, bars: list[Bar]) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# Null provider (default / graceful fallback)
# ---------------------------------------------------------------------------


class NullMarketDataProvider:
    """A provider that returns no live data.

    This is the default active provider. It guarantees ETI never fabricates
    a live quote or price. Any code depending on a live feed degrades
    gracefully: callers receive None / empty lists and must handle them.
    """

    def __init__(self) -> None:
        self.config = ProviderConfig(name="null", enabled=False)

    @property
    def name(self) -> str:
        return "null"

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            name="null",
            available=False,
            last_check=datetime.now(timezone.utc),
            error_message="No live market-data provider configured.",
        )

    def get_quote(self, instrument: str) -> None:
        return None

    def get_latest_bar(self, instrument: str, timeframe: str) -> None:
        return None

    def get_bars(
        self, instrument: str, timeframe: str, start: datetime, end: datetime
    ) -> list[Bar]:
        return []

    def to_dataframe(self, bars: list[Bar]) -> pd.DataFrame:
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
