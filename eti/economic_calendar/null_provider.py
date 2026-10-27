"""Null economic-calendar provider (default / graceful fallback)."""
from __future__ import annotations

from .base import NullEconomicCalendarProvider, ProviderConfig

__all__ = ["NullEconomicCalendarProvider", "ProviderConfig"]
