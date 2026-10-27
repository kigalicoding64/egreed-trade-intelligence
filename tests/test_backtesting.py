import json
import math
import pandas as pd
import pytest

from eti.backtesting import (
    CostModel,
    BacktestResult,
    chronological_split,
    data_version,
    load_result,
    never_claim_profitable,
    run_backtest,
    run_split_backtest,
    save_result,
)
from eti.strategies import (
    momentum_signals,
    no_trade_signals,
    rsi_mean_reversion_signals,
    sma_crossover_signals,
)


def _make_bars(n=120, start=1.10, step=0.0001):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    close = [start + i * step for i in range(n)]
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": [c + 0.0002 for c in close],
            "low": [c - 0.0002 for c in close],
            "close": close,
            "volume": [100] * n,
        }
    )


# ---------------------------------------------------------------------------
# run_backtest basics
# ---------------------------------------------------------------------------
def test_run_backtest_returns_expected_keys():
    bars = _make_bars(100)
    res = run_backtest(bars, sma_crossover_signals, {"fast_window": 5, "slow_window": 15})
    for key in (
        "total_return_pct",
        "buy_hold_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "exposure_pct",
        "trade_count",
        "cost_model",
        "look_ahead_prevention",
        "warning",
        "equity_curve",
    ):
        assert key in res, key
    assert len(res["equity_curve"]) == len(bars)


def test_run_backtest_look_ahead_prevention():
    """Position at bar t+1 must equal signal at bar t (no look-ahead)."""
    bars = _make_bars(60)
    res = run_backtest(bars, sma_crossover_signals, {"fast_window": 5, "slow_window": 15})
    ec = res["equity_curve"]
    # First equity row must equal 1.0 (no position before first bar)
    assert ec[0]["equity"] == pytest.approx(1.0)
    assert ec[0]["buy_hold"] == pytest.approx(1.0)


def test_run_backtest_costs_applied():
    bars = _make_bars(100)
    zero_cost = run_backtest(
        bars, sma_crossover_signals, {"fast_window": 5, "slow_window": 15},
        cost_model=CostModel(0, 0, 0),
    )
    costly = run_backtest(
        bars, sma_crossover_signals, {"fast_window": 5, "slow_window": 15},
        cost_model=CostModel(spread_bps=10, commission_bps=10, slippage_bps=10),
    )
    # With costs, total return should be <= zero-cost version (costs drag)
    assert costly["total_cost_bps"] > zero_cost["total_cost_bps"]
    assert costly["cost_return_impact_pct"] <= zero_cost["cost_return_impact_pct"]


def test_look_ahead_position_shifted_by_one_bar():
    """Signal at bar t must be applied as position at bar t+1, never t."""
    # close rises 1% every bar; a constant-long signal should not earn bar 0's return.
    close = pd.Series([1.0 * (1.001 ** i) for i in range(20)])
    bars = pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=20, freq="h"),
            "close": close,
            "open": close,
            "high": close,
            "low": close,
            "volume": [100] * 20,
        }
    )

    def always_long(df):
        return pd.Series(1, index=df.index, dtype=int)

    res = run_backtest(bars, always_long, {}, cost_model=CostModel(0, 0, 0))
    ec = res["equity_curve"]
    # Bar 0: position is 0 (signal shifted) -> equity unchanged at 1.0
    assert ec[0]["equity"] == pytest.approx(1.0)
    # Bar 1: position is 1 -> equity reflects bar-1 return only
    expected_equity_bar1 = 1.0 + close.pct_change().iloc[1]
    assert ec[1]["equity"] == pytest.approx(expected_equity_bar1, abs=1e-12)
    # The one-bar delay is proven: bar 0 earns nothing, bar 1 earns bar-1's return.


def test_run_backtest_insufficient_data_raises():
    with pytest.raises(ValueError):
        run_backtest(_make_bars(2), sma_crossover_signals)


def test_run_backtest_empty_raises():
    with pytest.raises(ValueError):
        run_backtest(_make_bars(0), sma_crossover_signals)


def test_no_trade_baseline_flat_return():
    bars = _make_bars(100)
    res = run_backtest(bars, no_trade_signals, {}, cost_model=CostModel(0, 0, 0))
    assert res["total_return_pct"] == pytest.approx(0.0)
    assert res["trade_count"] == 0
    assert res["exposure_pct"] == pytest.approx(0.0)
    assert res["max_drawdown_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------
def test_chronological_split_proportions():
    train, val, test = chronological_split(1000, 0.6, 0.2, 0.2)
    assert train.rows == 600
    assert val.rows == 200
    assert test.rows == 200
    # Chronological ordering: train < val < test indices
    assert int(train.end) < int(val.start)
    assert int(val.end) < int(test.start)


def test_chronological_split_defaults_to_remaining_test():
    train, val, test = chronological_split(100, 0.5, 0.25)
    assert train.rows == 50
    assert val.rows == 25
    assert test.rows == 25


def test_chronological_split_invalid_raises():
    with pytest.raises(ValueError):
        chronological_split(100, 0.5, 0.5, 0.2)  # sums to 1.2
    with pytest.raises(ValueError):
        chronological_split(3, 0.6, 0.2, 0.2)  # validation split would be 0 rows


# ---------------------------------------------------------------------------
# run_split_backtest
# ---------------------------------------------------------------------------
def test_run_split_backtest_metadata():
    bars = _make_bars(300)
    result = run_split_backtest(
        bars,
        "sma_crossover",
        {"fast_window": 10, "slow_window": 30},
        cost_model=CostModel(spread_bps=1, commission_bps=1, slippage_bps=0),
        train_frac=0.6,
        val_frac=0.2,
        data_source="data/raw/EURUSD1440.csv",
        data_path="data/raw/EURUSD1440.csv",
        timeframe="1d",
    )
    assert isinstance(result, BacktestResult)
    assert result.strategy == "sma_crossover"
    assert result.parameters == {"fast_window": 10, "slow_window": 30}
    assert result.data_source == "data/raw/EURUSD1440.csv"
    assert result.data_version is not None and len(result.data_version) == 64
    assert result.timeframe == "1d"
    assert result.train_period.rows + result.validation_period.rows + result.test_period.rows == len(bars)
    assert result.test_period is not None
    assert "test" in result.metrics
    assert "total_return_pct" in result.metrics["test"]
    assert result.cost_model.total_bps == 2.0


def test_run_split_backtest_unknown_strategy_raises():
    bars = _make_bars(100)
    with pytest.raises(ValueError):
        run_split_backtest(bars, "does_not_exist")


def test_run_split_backtest_no_trade_control():
    bars = _make_bars(200)
    result = run_split_backtest(bars, "no_trade", {}, cost_model=CostModel(0, 0, 0))
    test = result.metrics["test"]
    assert test["total_return_pct"] == pytest.approx(0.0)
    assert test["trade_count"] == 0
    assert test["exposure_pct"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# save / load reproducibility
# ---------------------------------------------------------------------------
def test_save_and_load_result_roundtrip(tmp_path):
    bars = _make_bars(150)
    result = run_split_backtest(bars, "momentum", {"lookback": 10})
    out = tmp_path / "bt.json"
    save_result(result, out)
    assert out.exists()
    loaded = load_result(out)
    assert loaded.strategy == result.strategy
    assert loaded.parameters == result.parameters
    assert loaded.data_version == result.data_version
    assert loaded.metrics["test"]["total_return_pct"] == pytest.approx(
        result.metrics["test"]["total_return_pct"]
    )
    assert loaded.cost_model.total_bps == result.cost_model.total_bps


def test_data_version_deterministic(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n", encoding="utf-8")
    assert data_version(f) == data_version(f)


# ---------------------------------------------------------------------------
# guardrail
# ---------------------------------------------------------------------------
def test_never_claim_profitable_message():
    msg = never_claim_profitable({})
    assert "not proof" in msg.lower() or "exploratory" in msg.lower()
    assert "profitab" in msg.lower()
