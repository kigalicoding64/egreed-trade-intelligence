"""
Official economic-calendar provider adapters.

Design principles:
- Replaceable adapter interface.
- No provider connected by default (NullEconomicCalendarProvider returns no events).
- Credentials from environment variables only.
- Every event carries source_url, publication_time, retrieval_time, event_time,
  impact, and a ``kind`` distinguishing confirmed facts from forecasts. Events
  are never invented.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EconomicEvent:
    """A single scheduled economic event.

    ``kind``:
      - "confirmed_fact": event officially scheduled/published
      - "forecast": consensus/analyst forecast value
      - "interpretation": expected market impact framing
    """
    kind: str  # confirmed_fact | forecast | interpretation
    event: str  # e.g. "Non-Farm Payrolls", "CPI m/m"
    event_time: datetime  # timezone-aware UTC (when the event occurs/releases)
    instrument: str | None = None  # e.g. "EUR/USD" if relevant
    publication_time: datetime | None = None  # when the calendar entry was published
    retrieval_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    forecast_value: str | None = None
    previous_value: str | None = None
    actual_value: str | None = None  # filled after release
    impact: str | None = None  # e.g. "high", "medium", "low"
    source_url: str | None = None
    provider: str | None = None

    def is_stale(self, max_age_seconds: float = 86400.0) -> bool:
        age = (datetime.now(timezone.utc) - self.retrieval_time).total_seconds()
        return age > max_age_seconds

    def is_upcoming(self) -> bool:
        return self.event_time > datetime.now(timezone.utc)


@dataclass(frozen=True)
class ProviderHealth:
    name: str
    available: bool
    last_check: datetime
    error_message: str | None = None


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    enabled: bool = False
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 10.0
    extra: dict = field(default_factory=dict)

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env)


@runtime_checkable
class EconomicCalendarProvider(Protocol):
    name: str
    config: ProviderConfig

    def health(self) -> ProviderHealth: ...
    def get_events(
        self, start: datetime, end: datetime, instrument: str | None = None
    ) -> list[EconomicEvent]: ...


class NullEconomicCalendarProvider:
    """Default calendar provider: returns no events (never invents events)."""

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
            error_message="No live economic-calendar provider configured.",
        )

    def get_events(
        self, start: datetime, end: datetime, instrument: str | None = None
    ) -> list[EconomicEvent]:
        return []
