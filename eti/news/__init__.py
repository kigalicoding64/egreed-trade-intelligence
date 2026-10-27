"""Financial news provider package."""
from __future__ import annotations

from .base import NewsItem, NewsProvider, ProviderConfig, ProviderHealth
from .null_provider import NullNewsProvider
from .registry import NewsRegistry, default_registry

__all__ = [
    "NewsItem",
    "NewsProvider",
    "ProviderConfig",
    "ProviderHealth",
    "NullNewsProvider",
    "NewsRegistry",
    "default_registry",
]
