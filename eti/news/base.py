"""
Financial news provider adapters.

Design principles:
- Replaceable adapter interface.
- No provider connected by default (NullNewsProvider returns no items).
- Credentials from environment variables only.
- Every news item carries source_url, publication_time, retrieval_time, and a
  ``kind`` distinguishing confirmed facts from sentiment / interpretation /
  forecasts. Headlines are never invented.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class NewsItem:
    """A single financial news item.

    ``kind`` classifies the nature of the content:
      - "confirmed_fact": directly reported, verifiable event/quote
      - "forecast": a prediction or estimate
      - "sentiment": opinion/market sentiment
      - "interpretation": analysis framing
    """
    kind: str  # confirmed_fact | forecast | sentiment | interpretation
    headline: str
    source_url: str
    publication_time: datetime  # timezone-aware UTC
    retrieval_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    instrument: str | None = None  # e.g. "EUR/USD" if applicable
    summary: str | None = None
    provider: str | None = None
    is_stale_threshold_seconds: float = 86400.0  # 1 day default

    def is_stale(self) -> bool:
        age = (datetime.now(timezone.utc) - self.retrieval_time).total_seconds()
        return age > self.is_stale_threshold_seconds


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
class NewsProvider(Protocol):
    name: str
    config: ProviderConfig

    def health(self) -> ProviderHealth: ...
    def get_news(
        self, instrument: str | None = None, limit: int = 20
    ) -> list[NewsItem]: ...


class NullNewsProvider:
    """Default news provider: returns no items (never invents headlines)."""

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
            error_message="No live news provider configured.",
        )

    def get_news(self, instrument: str | None = None, limit: int = 20) -> list[NewsItem]:
        return []
