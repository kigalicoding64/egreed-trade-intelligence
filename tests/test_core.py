import pandas as pd
import pytest
from eti.data import validate_bars, add_indicators, resample_bars
from eti.backtest import sma_crossover_backtest


def sample_bars(n=120):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    close = [1.10 + i * 0.0001 for i in range(n)]
    return pd.DataFrame({
        "time": times,
        "open": close,
        "high": [x + 0.0002 for x in close],
        "low": [x - 0.0002 for x in close],
        "close": close,
        "volume": [100] * n,
    })


def test_validate_bars_passes_valid_data():
    result = validate_bars(sample_bars())
    assert result["status"] == "pass"
    assert result["rows"] == 120


def test_validate_bars_detects_invalid_ohlc():
    bars = sample_bars()
    bars.loc[3, "high"] = 0.5
    assert validate_bars(bars)["invalid_ohlc_rows"] == 1


def test_indicators_create_expected_columns():
    result = add_indicators(sample_bars())
    for col in ("sma20", "sma50", "sma200", "rsi14", "atr14"):
        assert col in result.columns
    assert result["sma20"].notna().sum() > 0


def test_resample_aggregates_candles():
    bars = sample_bars(4)
    out = resample_bars(bars, "2h")
    assert len(out) == 2
    assert out.iloc[0]["open"] == bars.iloc[0]["open"]
    assert out.iloc[0]["close"] == bars.iloc[1]["close"]


def test_backtest_rejects_invalid_windows():
    with pytest.raises(ValueError):
        sma_crossover_backtest(sample_bars(), 20, 20)


def test_backtest_returns_metrics():
    result = sma_crossover_backtest(sample_bars(), 5, 15, 1)
    assert "total_return_pct" in result
    assert "max_drawdown_pct" in result
    assert result["bars"] == 120
