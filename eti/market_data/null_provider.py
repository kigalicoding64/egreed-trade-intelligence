"""Null / default market-data provider (no live data)."""
from __future__ import annotations

from .base import NullMarketDataProvider, ProviderConfig

__all__ = ["NullMarketDataProvider", "ProviderConfig"]
