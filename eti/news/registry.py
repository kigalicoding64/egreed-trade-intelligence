"""Provider registry for financial news."""
from __future__ import annotations

from .base import NewsProvider, NullNewsProvider


class NewsRegistry:
    def __init__(self) -> None:
        self._providers: dict[str, NewsProvider] = {"null": NullNewsProvider()}
        self._active: str = "null"

    def register(self, name: str, provider: NewsProvider) -> None:
        self._providers[name] = provider

    def get(self, name: str) -> NewsProvider | None:
        return self._providers.get(name)

    def set_active(self, name: str) -> None:
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered. Available: {list(self._providers)}")
        self._active = name

    def active(self) -> NewsProvider:
        return self._providers[self._active]

    def active_name(self) -> str:
        return self._active

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def health_all(self) -> dict[str, object]:
        return {name: p.health() for name, p in self._providers.items()}


default_registry = NewsRegistry()

__all__ = ["NewsRegistry", "default_registry", "NullNewsProvider"]
