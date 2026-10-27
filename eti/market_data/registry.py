"""
Provider registry for live market data.

Maps provider names to provider instances. The default active provider is the
NullMarketDataProvider (no live data). Providers are enabled only through
explicit configuration; none connect automatically.
"""
from __future__ import annotations

from .base import (
    MarketDataProvider,
    NullMarketDataProvider,
    ProviderConfig,
)
from .twelve_data import (
    TwelveDataProvider,
    make_provider_config,
    register_twelve_data_provider,
)


class MarketDataRegistry:
    """Registry of available market-data providers."""

    def __init__(self) -> None:
        self._providers: dict[str, MarketDataProvider] = {}
        # Default: no live provider.
        self._providers["null"] = NullMarketDataProvider()
        self._active: str = "null"

    def register(self, name: str, provider: MarketDataProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> MarketDataProvider | None:
        return self._providers.get(name)

    def set_active(self, name: str) -> None:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered. Available: {list(self._providers)}")
        self._active = name

    def active(self) -> MarketDataProvider:
        return self._providers[self._active]

    def active_name(self) -> str:
        return self._active

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def health_all(self) -> dict[str, object]:
        return {name: p.health() for name, p in self._providers.items()}

    def register_twelve_data(self, enabled: bool = False) -> TwelveDataProvider:
        """Convenience: build, register, and return a TwelveDataProvider."""
        provider = register_twelve_data_provider(self, name="twelve_data", enabled=enabled)
        return provider


# Module-level default registry.
default_registry = MarketDataRegistry()

__all__ = ["MarketDataRegistry", "default_registry"]
