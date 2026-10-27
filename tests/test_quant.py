import numpy as np
import pandas as pd
import pytest

from eti.quant import (
    RegimeResult,
    all_within_tolerance,
    classify_regime,
    insufficient_history_warning,
    reference_atr,
    reference_returns,
    reference_rsi,
    reference_sma,
    reference_volatility,
    verify_indicators,
)


# ---------------------------------------------------------------------------
# SMA — hand-computable
# ---------------------------------------------------------------------------
def test_reference_sma_hand_computed():
    close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = reference_sma(close, 3).tolist()
    assert pd.isna(out[0]) and pd.isna(out[1])  # not enough samples
    assert out[2] == pytest.approx(2.0)
    assert out[3] == pytest.approx(3.0)
    assert out[4] == pytest.approx(4.0)


def test_reference_sma_insufficient_history():
    close = pd.Series([1.0, 2.0, 3.0])
    out = reference_sma(close, 5)
    assert out.notna().sum() == 0


def test_reference_sma_invalid_window():
    with pytest.raises(ValueError):
        reference_sma(pd.Series([1.0, 2.0]), 0)


# ---------------------------------------------------------------------------
# Returns — hand-computable
# ---------------------------------------------------------------------------
def test_reference_returns_hand_computed():
    close = pd.Series([100.0, 110.0, 121.0])
    out = reference_returns(close).tolist()
    assert out[0] is None or pd.isna(out[0])
    assert out[1] == pytest.approx(0.10)
    assert out[2] == pytest.approx(121.0 / 110.0 - 1.0)


# ---------------------------------------------------------------------------
# RSI — property-based (bounded, monotonic edge cases)
# ---------------------------------------------------------------------------
def test_reference_rsi_bounded():
    rng = np.random.default_rng(42)
    close = pd.Series(1.1 + rng.standard_normal(100) * 0.001).cumsum()
    out = reference_rsi(close, 14)
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 100).all()


def test_reference_rsi_monotonic_decreasing_is_zero():
    close = pd.Series([10.0 - i * 0.1 for i in range(30)])
    out = reference_rsi(close, 14).dropna()
    # After warmup, monotonic decline -> avg_gain = 0 -> RSI = 0
    assert (out == 0.0).all()


def test_reference_rsi_monotonic_increasing_is_nan():
    close = pd.Series([10.0 + i * 0.1 for i in range(30)])
    out = reference_rsi(close, 14).dropna()
    # No losses -> avg_loss = 0 -> RSI undefined (NaN)
    assert out.isna().all()


def test_reference_rsi_invalid_window():
    with pytest.raises(ValueError):
        reference_rsi(pd.Series([1.0, 2.0]), 1)


# ---------------------------------------------------------------------------
# ATR — true-range hand check
# ---------------------------------------------------------------------------
def test_reference_atr_true_range_hand_computed():
    df = pd.DataFrame(
        {
            "high": [2.0, 3.0, 4.0],
            "low": [1.0, 1.5, 2.0],
            "close": [1.5, 2.0, 3.0],
        }
    )
    out = reference_atr(df, 2)
    valid = out.dropna()
    # With ewm min_periods=2, indices 1 and 2 are valid
    assert len(valid) == 2
    assert np.all(np.isfinite(valid))
    # True range: index0=1.0, index1=1.5, index2=2.0
    # ewm(alpha=0.5, adjust=False, min_periods=2): y[1]=0.5*1.0+0.5*1.5=1.25, y[2]=0.5*1.25+0.5*2.0=1.625
    assert valid.iloc[0] == pytest.approx(1.25)
    assert valid.iloc[1] == pytest.approx(1.625)


def test_reference_atr_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        reference_atr(df, 14)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------
def test_reference_volatility_hand_computed():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03])
    out = reference_volatility(returns, 3).tolist()
    assert pd.isna(out[0]) and pd.isna(out[1])
    assert out[2] == pytest.approx(returns.iloc[:3].std(), abs=1e-12)
    assert out[3] == pytest.approx(returns.iloc[1:4].std(), abs=1e-12)
    assert out[4] == pytest.approx(returns.iloc[2:5].std(), abs=1e-12)


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def test_classify_regime_insufficient_history():
    df = pd.DataFrame(
        {
            "close": pd.Series([1.1 + i * 0.001 for i in range(50)]),
            "high": pd.Series([1.101 + i * 0.001 for i in range(50)]),
            "low": pd.Series([1.099 + i * 0.001 for i in range(50)]),
        }
    )
    res = classify_regime(df)
    assert (res.regime == "insufficient_data").all()
    assert isinstance(res, RegimeResult)
    assert res.min_samples_required == 200


def test_classify_regime_trending_up():
    n = 250
    close = 1.1 + np.arange(n) * 0.001  # steady uptrend
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.001,
            "low": close - 0.001,
        }
    )
    res = classify_regime(df)
    labels = res.regime.dropna()
    # After warmup, should trend up (not all ranging/insufficient)
    non_insuff = labels[labels != "insufficient_data"]
    assert len(non_insuff) > 0
    assert (non_insuff == "trending_up").any() or (non_insuff == "ranging").any()
    assert set(labels.unique()) <= {
        "trending_up",
        "trending_down",
        "ranging",
        "insufficient_data",
    }


def test_classify_regime_trending_down():
    n = 250
    close = 1.1 - np.arange(n) * 0.001  # steady downtrend
    df = pd.DataFrame(
        {
            "close": close,
            "high": close + 0.001,
            "low": close - 0.001,
        }
    )
    res = classify_regime(df)
    labels = res.regime.dropna()
    non_insuff = labels[labels != "insufficient_data"]
    assert len(non_insuff) > 0
    assert (non_insuff == "trending_down").any() or (non_insuff == "ranging").any()


def test_classify_regime_invalid_windows():
    df = pd.DataFrame(
        {
            "close": pd.Series([1.1] * 250),
            "high": pd.Series([1.101] * 250),
            "low": pd.Series([1.099] * 250),
        }
    )
    with pytest.raises(ValueError):
        classify_regime(df, sma_windows=(50, 20, 200))  # fast not < slow


# ---------------------------------------------------------------------------
# Verification: engine vs independent reference
# ---------------------------------------------------------------------------
def _make_long_series(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 1.1 + np.cumsum(rng.standard_normal(n) * 0.0005)
    return pd.DataFrame(
        {
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": close + rng.standard_normal(n) * 0.0001,
            "high": close + np.abs(rng.standard_normal(n) * 0.0002),
            "low": close - np.abs(rng.standard_normal(n) * 0.0002),
            "close": close,
            "volume": rng.integers(50, 200, size=n),
        }
    )


def test_verify_indicators_all_within_tolerance():
    df = _make_long_series()
    results = verify_indicators(df)
    assert set(results) >= {"sma20", "sma50", "sma200", "rsi14", "atr14", "volatility20_pct", "return_pct"}
    assert all_within_tolerance(results)
    for name, r in results.items():
        assert r.within_tolerance is True
        assert r.max_rel_diff is not None and r.max_rel_diff < 1e-6
        assert r.non_nan_count_engine == r.non_nan_count_reference


def test_verify_indicators_raises_on_missing_columns():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError):
        verify_indicators(df)


def test_insufficient_history_warning():
    df_short = pd.DataFrame({"close": [1.0] * 50})
    assert insufficient_history_warning(df_short, 200) is not None
    df_long = pd.DataFrame({"close": [1.0] * 300})
    assert insufficient_history_warning(df_long, 200) is None
