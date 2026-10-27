from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .strategies import get_strategy, list_strategies

# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostModel:
    """Transaction cost assumptions in basis points (1 bps = 0.01%).

    total_bps is deducted on every unit of turnover (position change).
    These are simplified research assumptions; real execution costs depend on
    the broker, instrument, size, and market conditions.
    """
    spread_bps: float = 0.1
    commission_bps: float = 0.0
    slippage_bps: float = 0.1

    @property
    def total_bps(self) -> float:
        return self.spread_bps + self.commission_bps + self.slippage_bps

    def to_dict(self) -> dict[str, float]:
        return {
            "spread_bps": self.spread_bps,
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "total_bps": self.total_bps,
        }


# ---------------------------------------------------------------------------
# Data versioning (provenance)
# ---------------------------------------------------------------------------


def data_version(path: str | Path) -> str:
    """SHA-256 of a source data file, used as the data version for reproducibility."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Chronological train / validation / test split
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitPeriod:
    start: str | None
    end: str | None
    rows: int


def chronological_split(
    n: int,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float | None = None,
) -> tuple[SplitPeriod, SplitPeriod, SplitPeriod]:
    """Split ``n`` chronologically-ordered bars into train / validation / test.

    The split is strictly chronological — never random — to prevent look-ahead
    and data leakage. The test period is left untouched until final evaluation.

    Returns (train, validation, test) SplitPeriods with row ranges.
    """
    if train_frac < 0 or val_frac < 0:
        raise ValueError("fractions must be non-negative")
    if test_frac is None:
        test_frac = 1.0 - train_frac - val_frac
    if abs((train_frac + val_frac + test_frac) - 1.0) > 1e-9:
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    # test gets the remainder
    n_test = n - n_train - n_val

    if n_train < 1 or n_val < 1 or n_test < 1:
        raise ValueError(
            f"Split too small for n={n}: train={n_train}, val={n_val}, test={n_test}"
        )

    def period(start_idx: int, end_idx: int) -> SplitPeriod:
        return SplitPeriod(
            start=str(start_idx),
            end=str(end_idx - 1),
            rows=end_idx - start_idx,
        )

    train = period(0, n_train)
    val = period(n_train, n_train + n_val)
    test = period(n_train + n_val, n)
    return train, val, test


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    strategy: str
    strategy_version: str = "1.0.0"
    parameters: dict[str, Any] = field(default_factory=dict)
    data_source: str | None = None
    data_version: str | None = None
    timeframe: str | None = None
    train_period: SplitPeriod | None = None
    validation_period: SplitPeriod | None = None
    test_period: SplitPeriod | None = None
    cost_model: CostModel = field(default_factory=CostModel)
    periods_per_year: float | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    look_ahead_prevention: str = (
        "signals computed at bar t are applied at bar t+1 (shifted by one bar)"
    )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cost_model"] = self.cost_model.to_dict()
        return d


def _annualized_return(total_return: float, n_periods: int, periods_per_year: float) -> float:
    if n_periods <= 0 or periods_per_year <= 0:
        return None  # type: ignore[return-value]
    years = n_periods / periods_per_year
    if years <= 0:
        return None  # type: ignore[return-value]
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series, periods_per_year: float) -> float | None:
    if returns.empty or periods_per_year <= 0:
        return None
    mean = returns.mean()
    std = returns.std(ddof=1)
    if std is None or std == 0 or np.isnan(std):
        return None
    return float((mean / std) * np.sqrt(periods_per_year))


def _sortino(returns: pd.Series, periods_per_year: float) -> float | None:
    if returns.empty or periods_per_year <= 0:
        return None
    downside = returns[returns < 0]
    if downside.empty:
        return None
    dd = downside.std(ddof=1)
    if dd is None or dd == 0 or np.isnan(dd):
        return None
    mean = returns.mean()
    return float((mean / dd) * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Return (max_drawdown_fraction, drawdown_duration_in_periods)."""
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_dd = float(drawdown.min()) if len(drawdown) else 0.0
    # Duration: longest consecutive stretch below peak
    below = drawdown < 0
    # Compute streak lengths of consecutive True
    streak = 0
    max_streak = 0
    for is_below in below:
        if is_below:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_dd, int(max_streak)


def _trade_returns(
    position: pd.Series, asset_returns: pd.Series
) -> tuple[list[float], int]:
    """Identify completed round-trip trades and their returns.

    A trade opens when position goes 0->1 and closes when 1->0. Returns the
    list of per-trade returns and the number of completed trades.
    """
    trades: list[float] = []
    in_trade = False
    entry_ret = 1.0  # cumulative asset return since entry
    for pos, ret in zip(position, asset_returns):
        if not in_trade and pos == 1:
            in_trade = True
            entry_ret = 1.0 + ret
        elif in_trade:
            entry_ret *= 1.0 + ret
            if pos == 0:
                trades.append(entry_ret - 1.0)
                in_trade = False
    # If still in a trade at the end, it's an open position (not counted).
    return trades, len(trades)


def run_backtest(
    bars: pd.DataFrame,
    signal_fn,
    params: dict[str, Any] | None = None,
    cost_model: CostModel | None = None,
    periods_per_year: float | None = 252,
) -> dict[str, Any]:
    """Run a single long/flat backtest on the full dataset.

    Look-ahead bias is prevented by shifting the strategy signal by one bar:
    the signal computed at bar t (from data known at bar t) is applied as the
    position held during bar t+1.

    Returns a metrics dict (not the full BacktestResult; use
    ``run_split_backtest`` for train/validation/test with reproducibility
    metadata).
    """
    if bars.empty:
        raise ValueError("Cannot backtest an empty dataset")
    if len(bars) < 3:
        raise ValueError("Need at least 3 bars to backtest")

    params = dict(params or {})
    cost_model = cost_model or CostModel()

    x = bars[["time", "close"]].copy().sort_values("time").reset_index(drop=True)
    close = x["close"]

    # Raw signal from strategy (computed from data known at each bar).
    raw_signal = signal_fn(x, **params)
    raw_signal = raw_signal.reindex(x.index).fillna(0)
    # Shift by one bar: signal at bar t -> position held at bar t+1.
    position = raw_signal.shift(1).fillna(0).astype(int)

    asset_return = close.pct_change().fillna(0.0)
    turnover = position.diff().abs().fillna(position.abs())

    cost_per_turnover = cost_model.total_bps / 10000.0
    strategy_return = position * asset_return - turnover * cost_per_turnover

    equity = (1.0 + strategy_return).cumprod()
    buy_hold_equity = (1.0 + asset_return).cumprod()

    total_return = float(equity.iloc[-1] - 1.0)
    buy_hold_return = float(buy_hold_equity.iloc[-1] - 1.0)
    n_periods = max(len(x) - 1, 1)

    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    max_dd, dd_duration = _max_drawdown(equity)

    cagr = _annualized_return(total_return, n_periods, periods_per_year)
    sharpe = _sharpe(strategy_return, periods_per_year)
    sortino = _sortino(strategy_return, periods_per_year)

    trades, trade_count = _trade_returns(position, asset_return)
    win_rate = float(sum(1 for t in trades if t > 0) / len(trades)) if trades else None

    total_turnover = float(turnover.sum())
    total_cost_bps = total_turnover * cost_model.total_bps
    # Approximate return impact of costs (compounded costs are path-dependent;
    # this is a linear approximation for reporting).
    cost_return_impact = float(-(turnover * cost_per_turnover).sum())

    active = position > 0
    exposure = float(active.mean() * 100.0) if len(position) else 0.0

    positive_bar_pct = float((strategy_return > 0).mean() * 100.0)

    equity_curve = [
        {
            "time": str(t),
            "equity": float(e),
            "buy_hold": float(b),
        }
        for t, e, b in zip(x["time"], equity, buy_hold_equity)
    ]

    return {
        "strategy": getattr(signal_fn, "__name__", "strategy"),
        "parameters": params,
        "bars": int(len(x)),
        "start": str(x["time"].min()),
        "end": str(x["time"].max()),
        "periods_per_year": periods_per_year,
        "cost_model": cost_model.to_dict(),
        "total_return_pct": total_return * 100.0,
        "buy_hold_return_pct": buy_hold_return * 100.0,
        "cagr_pct": None if cagr is None else cagr * 100.0,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": max_dd * 100.0,
        "max_drawdown_duration_periods": dd_duration,
        "calmar_pct": None
        if (cagr is None or max_dd == 0)
        else (cagr / abs(max_dd)) * 100.0,
        "exposure_pct": exposure,
        "trade_count": trade_count,
        "win_rate_pct": None if win_rate is None else win_rate * 100.0,
        "positive_bar_pct": positive_bar_pct,
        "total_turnover": total_turnover,
        "total_cost_bps": total_cost_bps,
        "cost_return_impact_pct": cost_return_impact * 100.0,
        "look_ahead_prevention": (
            "signals computed at bar t are applied at bar t+1 (shifted by one bar)"
        ),
        "warning": (
            "Exploratory research baseline. Not a validated strategy. Costs are "
            "simplified; spread, slippage, financing, and execution are "
            "approximated. Never describe an unvalidated strategy as profitable."
        ),
        "equity_curve": equity_curve,
    }


# ---------------------------------------------------------------------------
# Train / validation / test backtest with reproducibility metadata
# ---------------------------------------------------------------------------


def run_split_backtest(
    bars: pd.DataFrame,
    strategy_name: str,
    params: dict[str, Any] | None = None,
    cost_model: CostModel | None = None,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    periods_per_year: float | None = 252,
    data_source: str | None = None,
    data_path: str | Path | None = None,
    timeframe: str | None = None,
) -> BacktestResult:
    """Run a backtest with chronological train/validation/test split.

    The test period is the headline (untouched) result. Train and validation
    are provided for research comparison but must not be used to claim
    out-of-sample performance without further validation.

    Returns a BacktestResult with full reproducibility metadata (strategy,
    parameters, data version, cost assumptions, periods).
    """
    signal_fn, default_params = get_strategy(strategy_name)
    params = {**default_params, **(params or {})}
    cost_model = cost_model or CostModel()

    if bars.empty:
        raise ValueError("Cannot backtest an empty dataset")
    n = len(bars)
    train, val, test = chronological_split(n, train_frac, val_frac)

    def subset(split: SplitPeriod) -> pd.DataFrame:
        start = int(split.start)
        end = int(split.end) + 1
        return bars.iloc[start:end].reset_index(drop=True)

    # Full-dataset run for the equity curve (reporting); metrics computed per
    # period separately below.
    full = run_backtest(bars, signal_fn, params, cost_model, periods_per_year)

    train_df = subset(train)
    val_df = subset(val)
    test_df = subset(test)

    train_metrics = run_backtest(train_df, signal_fn, params, cost_model, periods_per_year)
    val_metrics = run_backtest(val_df, signal_fn, params, cost_model, periods_per_year)
    test_metrics = run_backtest(test_df, signal_fn, params, cost_model, periods_per_year)

    dv = data_version(data_path) if data_path else None

    result = BacktestResult(
        strategy=strategy_name,
        strategy_version="1.0.0",
        parameters=params,
        data_source=data_source,
        data_version=dv,
        timeframe=timeframe,
        train_period=train,
        validation_period=val,
        test_period=test,
        cost_model=cost_model,
        periods_per_year=periods_per_year,
        metrics={
            "train": {
                "total_return_pct": train_metrics["total_return_pct"],
                "sharpe": train_metrics["sharpe"],
                "max_drawdown_pct": train_metrics["max_drawdown_pct"],
                "trade_count": train_metrics["trade_count"],
                "exposure_pct": train_metrics["exposure_pct"],
            },
            "validation": {
                "total_return_pct": val_metrics["total_return_pct"],
                "sharpe": val_metrics["sharpe"],
                "max_drawdown_pct": val_metrics["max_drawdown_pct"],
                "trade_count": val_metrics["trade_count"],
                "exposure_pct": val_metrics["exposure_pct"],
            },
            "test": {
                "total_return_pct": test_metrics["total_return_pct"],
                "buy_hold_return_pct": test_metrics["buy_hold_return_pct"],
                "cagr_pct": test_metrics["cagr_pct"],
                "sharpe": test_metrics["sharpe"],
                "sortino": test_metrics["sortino"],
                "max_drawdown_pct": test_metrics["max_drawdown_pct"],
                "max_drawdown_duration_periods": test_metrics[
                    "max_drawdown_duration_periods"
                ],
                "calmar_pct": test_metrics["calmar_pct"],
                "exposure_pct": test_metrics["exposure_pct"],
                "trade_count": test_metrics["trade_count"],
                "win_rate_pct": test_metrics["win_rate_pct"],
                "positive_bar_pct": test_metrics["positive_bar_pct"],
                "total_turnover": test_metrics["total_turnover"],
                "total_cost_bps": test_metrics["total_cost_bps"],
                "cost_return_impact_pct": test_metrics["cost_return_impact_pct"],
            },
        },
        equity_curve=full["equity_curve"],
    )
    return result


def save_result(result: BacktestResult, path: str | Path) -> Path:
    """Persist a BacktestResult to JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return path


def load_result(path: str | Path) -> BacktestResult:
    """Load a BacktestResult from JSON."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    cm_data = d.pop("cost_model", {})
    # total_bps is a derived property, not a constructor field.
    cm_data.pop("total_bps", None)
    cm = CostModel(**cm_data)
    # Reconstruct SplitPeriods
    for key in ("train_period", "validation_period", "test_period"):
        if d.get(key) is not None:
            d[key] = SplitPeriod(**d[key])
    d["cost_model"] = cm
    return BacktestResult(**d)


def never_claim_profitable(metrics: dict[str, Any]) -> str:
    """Return a guardrail message reminding that results are not proof of edge."""
    return (
        "These results are exploratory research output. A single backtest, win "
        "rate, or risk-adjusted metric is not proof of a strategy's edge and "
        "must not be presented as evidence of profitability."
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m eti.backtesting <STRATEGY> <CSV> [OUT_JSON]"""
    import sys

    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if len(argv) < 2:
        print(
            "Usage: python -m eti.backtesting <STRATEGY> <CSV> [OUT_JSON]",
            file=sys.stderr,
        )
        print("Strategies: " + ", ".join(sorted(list_strategies())), file=sys.stderr)
        return 2
    strategy_name, csv_path = argv[0], argv[1]
    out_path = argv[2] if len(argv) > 2 else None

    try:
        from .data import load_bars

        bars = load_bars(csv_path)
    except FileNotFoundError as exc:
        print(f"Error: source file not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"Error loading {csv_path}: {exc}", file=sys.stderr)
        return 1

    try:
        result = run_split_backtest(
            bars,
            strategy_name,
            data_source=csv_path,
            data_path=csv_path,
            timeframe="source",
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if out_path:
        save_result(result, out_path)
        print(f"Wrote {out_path}")
    else:
        print(json.dumps(result.to_dict(), indent=2))
    print()
    print(never_claim_profitable(result.metrics["test"]))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
