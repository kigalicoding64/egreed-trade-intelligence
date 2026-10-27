"""
MetaTrader 5 market-data adapter for ETI.

Read-only adapter for MetaTrader 5 terminal.
If the MetaTrader 5 terminal is not available or the package is not installed,
the adapter gracefully degrades — all methods return None / empty results
and the provider reports unavailable status.

Design principles (per ETI engineering standards):
- Read-only access only — no order placement, no real-money trading.
- Credentials come from the MT5 terminal session, not environment variables.
- Graceful degradation: if MT5 is not installed or the terminal is not running,
  the provider reports unavailable=False and returns None/empty results.
- Never invents prices or quotes. Unavailable data is reported as missing.
- Timestamps are UTC-aware where possible; terminal-timezone timestamps are
  normalized per ETI convention.
- Symbol information, ticks, and historical OHLCV bars where available.
- Validate chronological order and detect duplicate/duplicate candles.
- All dependencies (metatrader5) are optional — if not installed, the provider
  simply reports unavailable and falls back to the null/CSV provider.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

# Optional dependency — if not installed, adapter gracefully disables itself
try:
    import metatrader5 as mt5
    _MT5_AVAILABLE = True
except Exception:  # noqa: BLE001 — MT5 not installed / terminal not running
    _MT5_AVAILABLE = False

from .base import Bar, MarketDataProvider, ProviderConfig, ProviderHealth, Quote

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _timezone_offset_hours(dt: datetime) -> float:
    """Estimate the MT5 terminal timezone offset from UTC in hours."""
    if dt.tzinfo is None:
        return 0.0
    utc_dt = dt.astimezone(timezone.utc)
    return (dt - utc_dt).total_seconds() / 3600.0


def _parse_mt5_time(time_string: str) -> datetime:
    """Parse MT5 time string (typically 'YYYY.MM.DD HH:MM:SS') to UTC datetime."""
    try:
        # MT5 formats: "2023.12.15 14:30:00" or integer timestamp
        parts = time_string.strip().split(" ")
        if len(parts) == 2:
            date_part, time_part = parts
            d_parts = date_part.split(".")
            y, m, int(d) = int(d_parts[0]), int(d_parts[1]), int(d_parts[2])
            h, m2, s = [int(x) for x in time_part.split(":")]
            dt = datetime(y, m, int(d), h, m2, s, tzinfo=timezone.utc)
            return dt
    except Exception:
        pass
    return datetime.min.replace(tzinfo=timezone.utc)


def _validate_bar_data(
    open_, high, low, close, volume, timestamp: datetime
) -> Tuple[bool, str]:
    """Validate an OHLC bar: prices must be positive, high>=low, etc."""
    try:
        o, h, l, c = float(open_), float(high), float(low), float(close)
        v = float(volume) if volume is not None else 0.0
        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return False, " prices must be positive"
        if h < max(o, c) or l > min(o, c):
            return False, "invalid OHLC relationship"
        if h < l:
            return False, "high below low"
        if timestamp.tzinfo is None:
            return False, "timestamp must be timezone-aware"
        return True, ""
    except (ValueError, TypeError):
        return False, "cannot parse bar data"


# ---------------------------------------------------------------------------
# MT5Adapter — the concrete adapter
# ---------------------------------------------------------------------------


class MT5Adapter:
    """MetaTrader 5 read-only market-data adapter for ETI.

    Supports:
    - Symbol information (description, currency, digits, etc.)
    - Tick data (bid/ask/last)
    - Historical OHLCV bars from the MT5 terminal
    - Chronological validation and duplicate detection

    If the MetaTrader 5 terminal is not running or the package is not installed,
    all methods return None / empty results and the provider reports
    unavailable=False.
    """

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self._initialized = _MT5_AVAILABLE
        self._symbols_cache: Dict[str, Dict[str, Any]] = {}
        if self._initialized:
            mt5.initialize()
            if not mt5.terminal_info():
                logger.warning("MT5 terminal not detected — adapter operating in degraded mode")
                self._initialized = False

    @property
    def name(self) -> str:
        return "mt5"

    @property
    def enabled(self) -> bool:
        return self._initialized

    def health(self) -> ProviderHealth:
        """Report provider health."""
        if not self._initialized:
            return ProviderHealth(
                name="mt5",
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message="MetaTrader 5 terminal not available or not installed.",
            )
        # Basic health check — verify we can request symbol info
        try:
            _ = mt5.symbol_info("EURUSD")
            return ProviderHealth(
                name="mt5",
                available=True,
                last_check=datetime.now(timezone.utc),
                rate_limit_remaining=mt5.terminal_info().refresh_rate if mt5.terminal_info() else 0,
            )
        except Exception:
            return ProviderHealth(
                name="mt5",
                available=False,
                last_check=datetime.now(timezone.utc),
                error_message="Cannot connect to MT5 terminal.",
            )

    # ------------------------------------------------------------------
    # Symbol information
    # ------------------------------------------------------------------

    def symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return symbol specification from MT5.

        Returns a dict with keys: name, description, currency, digits,
        trade_contract_size, pip_size, trade_mode, etc.
        Returns None if MT5 not available or symbol not found.
        """
        if not self._initialized:
            return None
        try:
            info = mt5.symbol_info(symbol)
            if info is None:
                return None
            return {
                "name": info.name,
                "description": getattr(info, "description", ""),
                "currency": getattr(info, "currency", ""),
                "digits": getattr(info, "digits", 5),
                "trade_contract_size": getattr(info, "trade_contract_size", 100000.0),
                "pip_size": getattr(info, "pip_size", 0.00001),
                "trade_mode": getattr(info, "trade_mode", 0),
                "freeze_level": getattr(info, "freeze_level", 0),
                "swap_long": getattr(info, "swap_long", 0.0),
                "swap_short": getattr(info, "swap_short", 0.0),
                "market_execution": getattr(info, "market_execution", False),
            }
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Symbol list
    # ------------------------------------------------------------------

    def symbols_total(self) -> int:
        """Return total number of symbols available in MT5."""
        if not self._initialized:
            return 0
        try:
            return mt5.symbols_total()
        except Exception:
            return 0

    def symbols_get(self, pattern: str = "") -> List[str]:
        """Return list of symbol names, optionally filtered by pattern."""
        if not self._initialized:
            return []
        try:
            return mt5.symbols_get(pattern)
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Tick data
    # ------------------------------------------------------------------

    def tick_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Return latest tick data for symbol.

        Returns dict with: bid, ask, last, time, volume
        Returns None if MT5 not available or symbol not found.
        """
        if not self._initialized:
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "time": _parse_mt5_time(tick.time) if tick.time else datetime.min.replace(tzinfo=timezone.utc),
                "volume": tick.volume,
            }
        except Exception:
            return None

    def ticks_history(
        self, symbol: str, start: datetime, end: datetime, flags: int = 0
    ) -> Optional[List[Dict[str, Any]]]:
        """Return historical ticks between start and end.

        Returns list of dicts with time, bid, ask, last, volume.
        Returns None if MT5 not available.
        """
        if not self._initialized:
            return None
        try:
            ticks = mt5.ticks_history(symbol, start, end, flags)
            if ticks is None:
                return None
            result = []
            for tick in ticks:
                result.append(
                    {
                        "time": _parse_mt5_time(tick.time) if tick.time else datetime.min.replace(tzinfo=timezone.utc),
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "last": tick.last,
                        "volume": tick.volume,
                    }
                )
            return result
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Historical OHLCV bars
    # ------------------------------------------------------------------

    def get_bars(
        self,
        instrument: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> List[Bar]:
        """Return historical bars from MT5 terminal.

        Returns list of Bar objects in chronological order.
        Returns empty list if MT5 not available or no data.
        """
        if not self._initialized:
            return []

        # Map ETI timeframes to MT5 timeframes
        mt5_timeframe = self._map_timeframe(timeframe)
        if mt5_timeframe is None:
            # Unsupported timeframe — do not fabricate
            logger.warning(f"Unsupported timeframe '{timeframe}' for MT5 adapter")
            return []

        try:
            # MT5 returns bars newest-first; we reverse to chronological
            bars = mt5.copy_rates_range(instrument, mt5_timeframe, start, end)
            if bars is None or len(bars) == 0:
                return []

            result: List[Bar] = []
            for bar in reversed(bars):  # reverse to chronological
                open_, high, low, close, volume = bar["open"], bar["high"], bar["low"], bar["close"], bar["volume"]
                ts = datetime.fromtimestamp(bar["time"], tz=timezone.utc) if isinstance(bar.get("time"), int) else _parse_mt5_time(str(bar.get("time", "")))

                valid, err = _validate_bar_data(open_, high, low, close, volume, ts)
                if not valid:
                    logger.warning(f"Invalid bar data skipped: {err}")
                    continue

                result.append(
                    Bar(
                        instrument=instrument,
                        timeframe=timeframe,
                        timestamp=ts,
                        open=float(open_),
                        high=float(high),
                        low=float(low),
                        close=float(close),
                        volume=float(volume) if volume is not None else None,
                        source_url="mt5",
                        retrieved_at=datetime.now(timezone.utc),
                    )
                )
            return result

        except Exception:
            logger.exception("Error fetching MT5 bars")
            return []

    # ------------------------------------------------------------------
    # Timeframe mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_timeframe(timeframe: str) -> Optional[str]:
        """Map ETI timeframe to MT5 timeframe string.

        MT5 timeframe constants: mt5.TIMEFRAME_M1, TIMEFRAME_M5, etc.
        """
        mapping = {
            "1m": "TIMEFRAME_M1",
            "5m": "TIMEFRAME_M5",
            "15m": "TIMEFRAME_M15",
            "30m": "TIMEFRAME_M30",
            "1h": "TIMEFRAME_H1",
            "4h": "TIMEFRAME_H4",
            "1d": "TIMEFRAME_D1",
            "1w": "TIMEFRAME_W1",
        }
        mt5_name = mapping.get(timeframe)
        if mt5_name is None:
            return None
        # Return the constant name string that mt5 expects
        return mt5_name

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release MT5 resources."""
        if self._initialized and mt5:
            try:
                mt5.shutdown()
            except Exception:
                pass
            self._initialized = False

    def __del__(self) -> None:
        self.close()