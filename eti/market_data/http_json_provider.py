"""
Example HTTP/JSON market-data provider connector (skeleton).

This is a *template* for connecting a real third-party JSON API. It is NOT
connected by default and makes no network calls unless explicitly instantiated
and enabled via configuration. All credentials are read from environment
variables (never hardcoded).

Usage:
    from eti.market_data.http_json_provider import HttpJsonMarketDataProvider
    from eti.market_data.registry import default_registry

    cfg = ProviderConfig(
        name="example",
        enabled=False,                      # do not enable until verified
        base_url="https://api.example.com/v1",
        api_key_env="ETI_EXAMPLE_API_KEY",  # set in the environment
        rate_limit_per_min=60,
    )
    provider = HttpJsonMarketDataProvider(cfg)
    default_registry.register("example", provider)
    default_registry.set_active("example")   # only after verification
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

import httpx

from .base import Bar, ProviderConfig, Quote


class HttpJsonMarketDataProvider:
    """Skeleton HTTP/JSON market-data provider.

    Implements the provider protocol generically. Subclass and override
    ``_parse_quote`` / ``_parse_bar`` / request paths for a specific vendor.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._client = httpx.Client(timeout=config.timeout_seconds)

    @property
    def name(self) -> str:
        return self.config.name

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        key = self.config.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def health(self):
        from .base import ProviderHealth

        if not self.config.enabled:
            return ProviderHealth(
                name=self.name,
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message="Provider not enabled.",
            )
        try:
            resp = self._client.get(
                f"{self.config.base_url.rstrip('/')}/health", headers=self._headers()
            )
            resp.raise_for_status()
            return ProviderHealth(
                name=self.name,
                available=True,
                last_check=datetime.now(timezone.utc),
                rate_limit_remaining=resp.headers.get("X-RateLimit-Remaining", None),
            )
        except Exception as exc:  # noqa: BLE001 - report provider failure
            return ProviderHealth(
                name=self.name,
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message=str(exc),
            )

    def get_quote(self, instrument: str):
        """Return a Quote or None if unavailable. Never invents a price."""
        if not self.config.enabled:
            return None
        try:
            resp = self._client.get(
                f"{self.config.base_url.rstrip('/')}/quote/{instrument}",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            return self._parse_quote(instrument, data)
        except Exception:  # noqa: BLE001 - graceful degradation
            return None

    def get_latest_bar(self, instrument: str, timeframe: str):
        if not self.config.enabled:
            return None
        try:
            resp = self._client.get(
                f"{self.config.base_url.rstrip('/')}/bars/{instrument}",
                params={"timeframe": timeframe, "limit": 1},
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            bars = data.get("bars", []) or data.get("data", [])
            if not bars:
                return None
            return self._parse_bar(instrument, timeframe, bars[0])
        except Exception:  # noqa: BLE001
            return None

    def get_bars(self, instrument: str, timeframe: str, start, end):
        if not self.config.enabled:
            return []
        try:
            resp = self._client.get(
                f"{self.config.base_url.rstrip('/')}/bars/{instrument}",
                params={
                    "timeframe": timeframe,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                },
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
            raw_bars = data.get("bars", []) or data.get("data", [])
            return [
                self._parse_bar(instrument, timeframe, b) for b in raw_bars
            ]
        except Exception:  # noqa: BLE001
            return []

    def to_dataframe(self, bars):
        import pandas as pd

        if not bars:
            return pd.DataFrame(
                columns=[
                    "instrument",
                    "timeframe",
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "source_url",
                    "retrieved_at",
                ]
            )
        return pd.DataFrame([b.__dict__ for b in bars])

    # ------------------------------------------------------------------
    # Vendor-specific parsing hooks (override in subclass).
    # ------------------------------------------------------------------
    def _parse_quote(self, instrument: str, data: dict[str, Any]):
        raise NotImplementedError("Subclass must implement _parse_quote")

    def _parse_bar(self, instrument: str, timeframe: str, data: dict[str, Any]) -> Bar:
        raise NotImplementedError("Subclass must implement _parse_bar")

    def close(self) -> None:
        self._client.close()
