"""
Tests for Milestone 5 — Live market intelligence adapter framework.

These tests verify:
- Null providers return None/empty (never fabricate live data).
- Provider registry registration, active switching, and health.
- Quote/Bar/NewsItem/EconomicEvent models and staleness detection.
- Confirmed-fact vs forecast/sentiment distinction.
- LiveIntelligence snapshot with null providers (all empty).
- Credentials come from environment variables (never hardcoded).
"""
import os
from datetime import datetime, timedelta, timezone

import pytest

from eti.live_intelligence import LiveIntelligence
from eti.market_data import (
    Bar,
    MarketDataRegistry,
    NullMarketDataProvider,
    ProviderConfig,
    Quote,
)
from eti.market_data.null_provider import NullMarketDataProvider as N2
from eti.news import NewsItem, NullNewsProvider
from eti.news.registry import NewsRegistry
from eti.economic_calendar import EconomicEvent, NullEconomicCalendarProvider
from eti.economic_calendar.registry import EconomicCalendarRegistry


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Null providers return no data (never fabricate)
# ---------------------------------------------------------------------------
def test_null_market_data_provider_returns_none():
    p = NullMarketDataProvider()
    assert p.get_quote("EUR/USD") is None
    assert p.get_latest_bar("EUR/USD", "1m") is None
    assert p.get_bars("EUR/USD", "1m", datetime.now(UTC), datetime.now(UTC)) == []
    h = p.health()
    assert h.available is False
    assert h.error_message is not None


def test_null_news_provider_returns_empty():
    p = NullNewsProvider()
    assert p.get_news("EUR/USD", 20) == []
    h = p.health()
    assert h.available is False


def test_null_calendar_provider_returns_empty():
    p = NullEconomicCalendarProvider()
    assert p.get_events(datetime.now(UTC), datetime.now(UTC) + timedelta(days=7)) == []
    h = p.health()
    assert h.available is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def test_market_data_registry_default_is_null():
    reg = MarketDataRegistry()
    assert reg.active_name() == "null"
    assert isinstance(reg.active(), NullMarketDataProvider)
    assert "null" in reg.list_providers()


def test_market_data_registry_register_and_switch():
    reg = MarketDataRegistry()
    reg.register("custom", NullMarketDataProvider())
    reg.set_active("custom")
    assert reg.active_name() == "custom"
    with pytest.raises(KeyError):
        reg.set_active("does_not_exist")


def test_news_registry_default_is_null():
    reg = NewsRegistry()
    assert reg.active_name() == "null"
    assert isinstance(reg.active(), NullNewsProvider)


def test_calendar_registry_default_is_null():
    reg = EconomicCalendarRegistry()
    assert reg.active_name() == "null"
    assert isinstance(reg.active(), NullEconomicCalendarProvider)


# ---------------------------------------------------------------------------
# ProviderConfig reads credentials from env only (never hardcoded)
# ---------------------------------------------------------------------------
def test_provider_config_api_key_from_env(monkeypatch):
    monkeypatch.setenv("ETI_TEST_API_KEY", "secret-123")
    cfg = ProviderConfig(name="test", api_key_env="ETI_TEST_API_KEY")
    assert cfg.api_key() == "secret-123"
    # Not hardcoded: missing env var -> None
    monkeypatch.delenv("ETI_TEST_API_KEY", raising=False)
    assert cfg.api_key() is None


def test_provider_config_secret_from_env(monkeypatch):
    monkeypatch.setenv("ETI_TEST_SECRET", "s3cr3t")
    cfg = ProviderConfig(name="test", secret_env="ETI_TEST_SECRET")
    assert cfg.secret() == "s3cr3t"


# ---------------------------------------------------------------------------
# Quote / Bar models and staleness
# ---------------------------------------------------------------------------
def test_quote_mid_and_staleness():
    now = datetime.now(UTC)
    q = Quote(
        instrument="EUR/USD",
        timestamp=now,
        bid=1.1000,
        ask=1.1002,
        last=1.1001,
        source_url="https://example.com/quote",
        retrieved_at=now,
    )
    assert q.mid() == pytest.approx(1.1001)
    assert q.is_stale(max_age_seconds=60.0) is False
    old = Quote(
        instrument="EUR/USD",
        timestamp=now - timedelta(hours=1),
        bid=1.1,
        ask=1.1002,
        retrieved_at=now - timedelta(hours=1),
    )
    assert old.is_stale(max_age_seconds=60.0) is True


def test_bar_model_and_staleness():
    now = datetime.now(UTC)
    b = Bar(
        instrument="EUR/USD",
        timeframe="1m",
        timestamp=now,
        open=1.1,
        high=1.1005,
        low=1.0998,
        close=1.1002,
        volume=100.0,
        retrieved_at=now,
    )
    assert b.is_stale(max_age_seconds=60.0) is False
    old_bar = Bar(
        instrument="EUR/USD",
        timeframe="1m",
        timestamp=now - timedelta(hours=1),
        open=1.1,
        high=1.1005,
        low=1.0998,
        close=1.1002,
        retrieved_at=now - timedelta(hours=1),
    )
    assert old_bar.is_stale(max_age_seconds=60.0) is True


# ---------------------------------------------------------------------------
# NewsItem: confirmed vs forecast/sentiment
# ---------------------------------------------------------------------------
def test_news_item_kind_filtering():
    now = datetime.now(UTC)
    items = [
        NewsItem(
            kind="confirmed_fact",
            headline="ECB holds rates steady",
            source_url="https://ecb.europa.eu/press/pr/1",
            publication_time=now,
            instrument="EUR/USD",
        ),
        NewsItem(
            kind="forecast",
            headline="Analysts expect rate cut",
            source_url="https://example.com/forecast",
            publication_time=now,
            instrument="EUR/USD",
        ),
        NewsItem(
            kind="sentiment",
            headline="Market sentiment turns cautious",
            source_url="https://example.com/sentiment",
            publication_time=now,
            instrument="EUR/USD",
        ),
    ]
    confirmed = LiveIntelligence.confirmed_news(items)
    assert len(confirmed) == 1
    assert confirmed[0].headline == "ECB holds rates steady"

    forecasts = [i for i in items if i.kind == "forecast"]
    assert len(forecasts) == 1


def test_news_item_staleness():
    now = datetime.now(UTC)
    fresh = NewsItem(
        kind="confirmed_fact",
        headline="Fresh headline",
        source_url="https://example.com/1",
        publication_time=now,
        retrieval_time=now,
        is_stale_threshold_seconds=60.0,
    )
    assert fresh.is_stale() is False
    stale = NewsItem(
        kind="confirmed_fact",
        headline="Stale headline",
        source_url="https://example.com/2",
        publication_time=now - timedelta(days=2),
        retrieval_time=now - timedelta(days=2),
        is_stale_threshold_seconds=60.0,
    )
    assert stale.is_stale() is True


# ---------------------------------------------------------------------------
# EconomicEvent: confirmed vs forecast, upcoming
# ---------------------------------------------------------------------------
def test_economic_event_kind_and_upcoming():
    now = datetime.now(UTC)
    events = [
        EconomicEvent(
            kind="confirmed_fact",
            event="ECB Meeting",
            instrument="EUR/USD",
            event_time=now + timedelta(days=1),
            publication_time=now,
            impact="high",
        ),
        EconomicEvent(
            kind="forecast",
            event="NFP Forecast",
            instrument="USD",
            event_time=now + timedelta(days=2),
            publication_time=now,
            forecast_value="180K",
            impact="high",
        ),
        EconomicEvent(
            kind="confirmed_fact",
            event="Past Release",
            instrument="USD",
            event_time=now - timedelta(days=1),
            publication_time=now - timedelta(days=2),
            actual_value="175K",
            impact="high",
        ),
    ]
    confirmed = LiveIntelligence.confirmed_events(events)
    assert len(confirmed) == 2
    upcoming = LiveIntelligence.upcoming_events(events)
    # Both future events are upcoming regardless of kind
    assert len(upcoming) == 2


# ---------------------------------------------------------------------------
# LiveIntelligence snapshot with null providers
# ---------------------------------------------------------------------------
def test_live_intelligence_snapshot_null_providers():
    li = LiveIntelligence()
    snap = li.snapshot("EUR/USD")
    assert snap.instrument == "EUR/USD"
    assert snap.quote is None
    assert snap.quote_stale is True
    assert snap.latest_bar is None
    assert snap.bar_stale is True
    assert snap.news == []
    assert snap.news_stale_count == 0
    assert snap.events == []
    assert snap.events_stale_count == 0
    # Health should show null providers unavailable
    health = snap.provider_health
    assert health["market_data"]["null"].available is False
    assert health["news"]["null"].available is False
    assert health["economic_calendar"]["null"].available is False


def test_live_intelligence_staleness_flags():
    li = LiveIntelligence(max_quote_staleness_seconds=0.0)  # anything is stale
    snap = li.snapshot("EUR/USD")
    # With null provider there's no quote; quote_stale True by default
    assert snap.quote_stale is True
