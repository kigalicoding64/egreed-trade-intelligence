"""Provider registry for economic calendars."""
from __future__ import annotations

from .base import EconomicCalendarProvider, NullEconomicCalendarProvider


class EconomicCalendarRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, EconomicCalendarProvider] = {
            "null": NullEconomicCalendarProvider()
        }
        self._active: str = "null"

    def register(self, name: str, provider: EconomicCalendarProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> EconomicCalendarProvider | None:
        return self._providers.get(name)

    def set_active(self, name: str) -> None:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered. Available: {list(self._providers)}")
        self._active = name

    def active(self) -> EconomicCalendarProvider:
        return self._providers[self._active]

    def active_name(self) -> str:
        return self._active

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def health_all(self) -> dict[str, object]:
        return {name: p.health() for name, p in self._providers.items()}


default_registry = EconomicCalendarRegistry()

__all__ = ["EconomicCalendarRegistry", "default_registry", "NullEconomicCalendarProvider"]
