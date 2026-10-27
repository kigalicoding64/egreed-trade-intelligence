import os
project = r'C:\Users\Blue Code technology\Downloads\ETI_working_core_v0_1\egreed-trade-intelligence'
init_path = os.path.join(project, 'eti', 'market_data', '__init__.py')
with open(init_path, 'w') as f:
    f.write('''"""Live market-data provider package."""
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
''')
print('File written successfully')