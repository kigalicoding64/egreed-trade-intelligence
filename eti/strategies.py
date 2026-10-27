from __future__ import annotations

import pandas as pd

from .quant import reference_rsi, reference_sma


def sma_crossover_signals(
    df: pd.DataFrame, fast_window: int = 20, slow_window: int = 50
) -> pd.Series:
    """Trend-following: long when fast SMA > slow SMA, flat otherwise.

    Returns a raw position signal (1 = long, 0 = flat) computed from data
    available at each bar. The backtester applies it one bar later to prevent
    look-ahead bias.
    """
    close = df["close"]
    fast = reference_sma(close, fast_window)
    slow = reference_sma(close, slow_window)
    return (fast > slow).astype(int)


def rsi_mean_reversion_signals(
    df: pd.DataFrame,
    rsi_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Mean-reversion: long when RSI < oversold, flat otherwise.

    A position is entered when RSI falls below the oversold threshold and is
    held until the signal flattens (RSI >= oversold). This is a simplified
    research baseline: re-entry is allowed whenever RSI < oversold again.
    """
    rsi = reference_rsi(df["close"], rsi_window)
    return (rsi < oversold).astype(int)


def momentum_signals(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Momentum: long when price is above its price ``lookback`` bars ago."""
    close = df["close"]
    return (close > close.shift(lookback)).astype(int)


def no_trade_signals(df: pd.DataFrame) -> pd.Series:
    """No-trade baseline: always flat (position 0). Used as a control."""
    return pd.Series(0, index=df.index, dtype=int)


# Registry of available baseline strategies for discovery / CLI use.
BASELINE_STRATEGIES: dict[str, tuple] = {
    "sma_crossover": (
        sma_crossover_signals,
        {"fast_window": 20, "slow_window": 50},
        "Trend-following SMA crossover (long/flat).",
    ),
    "rsi_mean_reversion": (
        rsi_mean_reversion_signals,
        {"rsi_window": 14, "oversold": 30.0, "overbought": 70.0},
        "Mean-reversion RSI (long when oversold, otherwise flat).",
    ),
    "momentum": (
        momentum_signals,
        {"lookback": 20},
        "Absolute momentum (long when price > price N bars ago).",
    ),
    "no_trade": (
        no_trade_signals,
        {},
        "No-trade control baseline (always flat).",
    ),
}


def list_strategies() -> dict[str, str]:
    """Return {strategy_name: description} for available baselines."""
    return {name: info[2] for name, info in BASELINE_STRATEGIES.items()}


def get_strategy(name: str):
    """Return (signal_fn, default_params) for a registered baseline strategy."""
    if name not in BASELINE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {sorted(BASELINE_STRATEGIES)}"
        )
    signal_fn, default_params, _ = BASELINE_STRATEGIES[name]
    return signal_fn, dict(default_params)
