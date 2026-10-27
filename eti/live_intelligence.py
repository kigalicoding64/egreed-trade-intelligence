"""
Live market-intelligence facade.

Coordinates the market-data, news, and economic-calendar provider domains.
Never connects to a provider by default: all active providers are Null
providers until an operator explicitly registers and enables a verified one.

Key safeguards:
- No provider auto-connects; credentials come from env vars only.
- Stale data is detected and surfaced, never silently served.
- Confirmed facts are distinguished from forecasts/sentiment/interpretation.
- Nothing is invented: missing providers return None/empty lists.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from eti.economic_calendar import EconomicEvent, default_registry as ec_registry
from eti.market_data import Bar, Quote, default_registry as md_registry
from eti.news import NewsItem, default_registry as news_registry


@dataclass(frozen=True)
class IntelligenceSnapshot:
    """Point-in-time snapshot of all live-intelligence domains."""
    instrument: str
    retrieved_at: datetime
    quote: Quote | None
    quote_stale: bool
    latest_bar: Bar | None
    bar_stale: bool
    news: list[NewsItem]
    news_stale_count: int
    events: list[EconomicEvent]
    events_stale_count: int
    provider_health: dict[str, dict[str, Any]]
    note: str = (
        "Live intelligence is optional and provider-dependent. "
        "Null providers return no data until a verified provider is configured."
    )


class LiveIntelligence:
    """Facade over the three live-intelligence provider domains."""

    def __init__(
        self,
        max_quote_staleness_seconds: float = 300.0,
        max_bar_staleness_seconds: float = 3600.0,
        max_news_staleness_seconds: float = 86400.0,
        max_event_staleness_seconds: float = 86400.0,
    ) -> None:
        self.max_quote_staleness_seconds = max_quote_staleness_seconds
        self.max_bar_staleness_seconds = max_bar_staleness_seconds
        self.max_news_staleness_seconds = max_news_staleness_seconds
        self.max_event_staleness_seconds = max_event_staleness_seconds

    # ------------------------------------------------------------------
    # Provider access
    # ------------------------------------------------------------------
    @property
    def market_data_provider(self):
        return md_registry.active()

    @property
    def news_provider(self):
        return news_registry.active()

    @property
    def calendar_provider(self):
        return ec_registry.active()

    def provider_health(self) -> dict[str, dict[str, Any]]:
        return {
            "market_data": {n: h for n, h in md_registry.health_all().items()},
            "news": {n: h for n, h in news_registry.health_all().items()},
            "economic_calendar": {n: h for n, h in ec_registry.health_all().items()},
        }

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    def snapshot(self, instrument: str = "EUR/USD") -> IntelligenceSnapshot:
        now = datetime.now(timezone.utc)

        md = self.market_data_provider
        quote = md.get_quote(instrument)
        quote_stale = quote.is_stale(self.max_quote_staleness_seconds) if quote else True

        latest_bar = md.get_latest_bar(instrument, "1m")
        bar_stale = latest_bar.is_stale(self.max_bar_staleness_seconds) if latest_bar else True

        news = self.news_provider.get_news(instrument=instrument, limit=20)
        news_stale_count = sum(1 for n in news if n.is_stale())
        # Attach staleness to each item for UI consumption
        news = [n for n in news]

        now_utc = datetime.now(timezone.utc)
        events = self.calendar_provider.get_events(
            start=now_utc,
            end=now_utc + timedelta(days=7),
            instrument=instrument,
        )
        events_stale_count = sum(1 for e in events if e.is_stale())

        return IntelligenceSnapshot(
            instrument=instrument,
            retrieved_at=now,
            quote=quote,
            quote_stale=quote_stale,
            latest_bar=latest_bar,
            bar_stale=bar_stale,
            news=news,
            news_stale_count=news_stale_count,
            events=events,
            events_stale_count=events_stale_count,
            provider_health=self.provider_health(),
        )

    # ------------------------------------------------------------------
    # Confirmed-facts filtering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def confirmed_news(items: list[NewsItem]) -> list[NewsItem]:
        """Return only news items classified as confirmed facts."""
        return [i for i in items if i.kind == "confirmed_fact"]

    @staticmethod
    def confirmed_events(items: list[EconomicEvent]) -> list[EconomicEvent]:
        """Return only calendar events classified as confirmed facts."""
        return [i for i in items if i.kind == "confirmed_fact"]

    @staticmethod
    def upcoming_events(items: list[EconomicEvent]) -> list[EconomicEvent]:
        return [i for i in items if i.is_upcoming()]
