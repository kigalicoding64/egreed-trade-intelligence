from __future__ import annotations
import pandas as pd
import numpy as np
from .data import add_indicators


def sma_crossover_backtest(
    bars: pd.DataFrame,
    fast_window: int = 20,
    slow_window: int = 50,
    cost_bps_per_turn: float = 1.0,
) -> dict:
    """Long/flat SMA crossover research baseline.

    Signals are shifted one bar so the strategy cannot trade on the close
    used to compute the signal. Cost is deducted when position changes.
    This is a simplified research baseline, not a broker execution simulator.
    """
    if fast_window < 2 or slow_window <= fast_window:
        raise ValueError("Require 2 <= fast_window < slow_window")
    if cost_bps_per_turn < 0:
        raise ValueError("cost_bps_per_turn must be non-negative")
    if len(bars) < slow_window + 3:
        raise ValueError(f"Need at least {slow_window + 3} bars")
    x = bars[["time", "close"]].copy().sort_values("time").reset_index(drop=True)
    x["fast"] = x["close"].rolling(fast_window, min_periods=fast_window).mean()
    x["slow"] = x["close"].rolling(slow_window, min_periods=slow_window).mean()
    x["signal"] = (x["fast"] > x["slow"]).astype(int)
    x["position"] = x["signal"].shift(1).fillna(0)
    x["asset_return"] = x["close"].pct_change().fillna(0)
    x["turnover"] = x["position"].diff().abs().fillna(x["position"].abs())
    x["strategy_return"] = x["position"] * x["asset_return"] - (
        x["turnover"] * cost_bps_per_turn / 10000
    )
    x["equity"] = (1 + x["strategy_return"]).cumprod()
    x["buy_hold_equity"] = (1 + x["asset_return"]).cumprod()
    running_peak = x["equity"].cummax()
    drawdown = x["equity"] / running_peak - 1
    returns = x["strategy_return"]
    active = x["position"] > 0
    total_return = float(x["equity"].iloc[-1] - 1)
    periods = max(len(x) - 1, 1)
    # Annualized Sharpe is deliberately omitted because the correct periods/year
    # depends on timeframe, market calendar and the data source.
    trades = int(((x["position"].diff().fillna(x["position"]).abs()) > 0).sum())
    return {
        "strategy": "SMA crossover long/flat",
        "fast_window": fast_window,
        "slow_window": slow_window,
        "cost_bps_per_turn": cost_bps_per_turn,
        "bars": int(len(x)),
        "start": str(x["time"].min()),
        "end": str(x["time"].max()),
        "total_return_pct": total_return * 100,
        "buy_hold_return_pct": float((x["buy_hold_equity"].iloc[-1] - 1) * 100),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "exposure_pct": float(active.mean() * 100),
        "position_changes": trades,
        "positive_bar_pct": float((returns > 0).mean() * 100),
        "warning": "Exploratory baseline only. No train/test split, spread model, slippage model, or strategy validation is included.",
        "equity_curve": [
            {"time": str(t), "equity": float(e), "buy_hold": float(b)}
            for t, e, b in zip(x["time"], x["equity"], x["buy_hold_equity"])
        ],
    }
