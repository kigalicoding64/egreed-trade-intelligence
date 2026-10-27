import os

path = 'eti/market_data/__init__.py'
content = """"""Live market-data provider package."""
from __future__ import annotations

from .base import Bar, MarketDataProvider, ProviderConfig, ProviderHealth, Quote
from .null_provider import NullMarketDataProvider
from .registry import MarketDataRegistry, default_registry
from .twelve_data import TwelveDataProvider

__all__ = [
    "Bar",
    "MarketDataProvider",
    "ProviderConfig",
    "ProviderHealth",
    "Quote",
    "NullMarketDataProvider",
    "MarketDataRegistry",
    "default_registry",
    "TwelveDataProvider",
]
"""
with open(path, 'w') as f:
    f.write(content)
print('File written successfully')