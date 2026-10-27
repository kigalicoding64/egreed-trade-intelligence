import pandas as pd
import pytest

from eti.strategies import (
    BASELINE_STRATEGIES,
    get_strategy,
    list_strategies,
    momentum_signals,
    no_trade_signals,
    rsi_mean_reversion_signals,
    sma_crossover_signals,
)


def test_list_strategies_returns_descriptions():
    descs = list_strategies()
    for name in ("sma_crossover", "rsi_mean_reversion", "momentum", "no_trade"):
        assert name in descs
        assert isinstance(descs[name], str) and len(descs[name]) > 0


def test_get_strategy_returns_fn_and_params():
    fn, params = get_strategy("sma_crossover")
    assert fn is sma_crossover_signals
    assert params == {"fast_window": 20, "slow_window": 50}

    fn, params = get_strategy("no_trade")
    assert fn is no_trade_signals
    assert params == {}


def test_get_strategy_unknown_raises():
    with pytest.raises(ValueError):
        get_strategy("does_not_exist")


def test_sma_crossover_signals_shift_applied_by_backtester():
    # Raw signal must not itself be shifted here; the backtester shifts.
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    df = pd.DataFrame({"close": close})
    sig = sma_crossover_signals(df, fast_window=2, slow_window=4)
    # With fast>slow at the end, signal = 1 there; first bars NaN->0 via fillna in backtester
    assert sig.dtype.kind in "bi"  # bool or int
    assert len(sig) == len(df)


def test_rsi_mean_reversion_signals_bounded():
    n = 100
    close = pd.Series([1.1 + (i % 10) * 0.001 for i in range(n)])
    df = pd.DataFrame({"close": close})
    sig = rsi_mean_reversion_signals(df)
    assert set(sig.dropna().unique()) <= {0, 1}
    assert len(sig) == n


def test_momentum_signals_hand_computed():
    close = pd.Series([1.0, 1.1, 1.2, 1.05, 1.3])  # bar4 > bar0 (1.3 > 1.0)
    df = pd.DataFrame({"close": close})
    sig = momentum_signals(df, lookback=4).tolist()
    assert sig[:4] == [0, 0, 0, 0]  # not enough lookback history -> NaN -> 0
    assert sig[4] == 1  # 1.3 > 1.0


def test_no_trade_signals_always_flat():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    sig = no_trade_signals(df)
    assert (sig == 0).all()


def test_baseline_strategies_registry_complete():
    expected = {"sma_crossover", "rsi_mean_reversion", "momentum", "no_trade"}
    assert set(BASELINE_STRATEGIES) == expected
