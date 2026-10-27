from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .data import add_indicators as engine_add_indicators

Regime = Literal["trending_up", "trending_down", "ranging", "insufficient_data"]

# ---------------------------------------------------------------------------
# Independent reference implementations
# ---------------------------------------------------------------------------
# These are deliberately simple, transparently correct implementations that can
# be checked by hand. They exist to verify the engine (eti.data.add_indicators)
# and to guarantee reproducible results. They never impute or invent values.


def reference_sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average. Returns NaN until ``window`` real samples exist."""
    if window < 1:
        raise ValueError("window must be >= 1")
    return close.rolling(window, min_periods=window).mean()


def reference_returns(close: pd.Series) -> pd.Series:
    """Bar-to-bar simple returns (fraction). First bar is NaN."""
    return close.pct_change()


def reference_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI.

    Uses Wilder's smoothing (alpha = 1/window) with min_periods=window so the
    first RSI value appears only after ``window`` real periods. Returns NaN
    when average loss is zero (no declining bars in the window).
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def reference_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Wilder's ATR using the standard true-range definition."""
    if window < 2:
        raise ValueError("window must be >= 2")
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ATR requires columns: {sorted(missing)}")
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def reference_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation of returns. NaN until ``window`` samples."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return returns.rolling(window, min_periods=window).std()


# ---------------------------------------------------------------------------
# Trend / regime classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeResult:
    regime: pd.Series  # Regime label per bar
    sma20: pd.Series
    sma50: pd.Series
    sma200: pd.Series
    atr14: pd.Series
    min_samples_required: int = 200
    note: str = (
        "Regime is 'insufficient_data' until 200 real samples are available. "
        "Labels are descriptive, not predictive."
    )


def classify_regime(
    df: pd.DataFrame,
    sma_windows: tuple[int, int, int] = (20, 50, 200),
    atr_window: int = 14,
    regime_return_threshold_pct: float = 0.10,
) -> RegimeResult:
    """Classify each bar into a trend/regime label.

    Labels (provisional, descriptive):
      - trending_up:   close above SMA20 above SMA50 and SMA20 rising
      - trending_down: close below SMA20 below SMA50 and SMA20 falling
      - ranging:       otherwise (price oscillating around SMAs, low trend)
      - insufficient_data: fewer than 200 real samples available

    Requires at least ``max(sma_windows)`` real samples before a non-insufficient
    label is assigned. Never fabricates a regime for short histories.
    """
    fast, slow, trend_window = sma_windows
    if not (fast < slow < trend_window):
        raise ValueError("Require fast < slow < trend_window")
    if len(df) < trend_window:
        n = len(df)
        regimes = pd.Series(["insufficient_data"] * n, index=df.index, dtype=object)
        empty = pd.Series(dtype="float64", index=df.index)
        return RegimeResult(
            regime=regimes,
            sma20=empty,
            sma50=empty,
            sma200=empty,
            atr14=empty,
            min_samples_required=trend_window,
        )

    close = df["close"]
    sma20 = reference_sma(close, fast)
    sma50 = reference_sma(close, slow)
    sma200 = reference_sma(close, trend_window)
    atr14 = reference_atr(df, atr_window)

    sma20_rising = sma20.diff() > 0
    sma20_falling = sma20.diff() < 0

    price_above_stack = (close > sma20) & (sma20 > sma50) & (sma50 > sma200)
    price_below_stack = (close < sma20) & (sma20 < sma50) & (sma50 < sma200)

    regime = pd.Series("ranging", index=df.index, dtype=object)
    regime[price_above_stack & sma20_rising] = "trending_up"
    regime[price_below_stack & sma20_falling] = "trending_down"

    # Before the trend window is fully populated, label as insufficient_data.
    warmup = sma200.isna()
    regime[warmup] = "insufficient_data"

    return RegimeResult(
        regime=regime,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        atr14=atr14,
        min_samples_required=trend_window,
    )


# ---------------------------------------------------------------------------
# Verification: engine vs independent reference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VerificationResult:
    indicator: str
    max_abs_diff: float | None
    max_rel_diff: float | None
    within_tolerance: bool
    non_nan_count_engine: int
    non_nan_count_reference: int
    note: str


def _max_abs_diff(a: pd.Series, b: pd.Series) -> float | None:
    combined = pd.concat([a, b], axis=1).dropna()
    if combined.empty:
        return None
    return float((combined.iloc[:, 0] - combined.iloc[:, 1]).abs().max())


def _max_rel_diff(a: pd.Series, b: pd.Series) -> float | None:
    combined = pd.concat([a, b], axis=1).dropna()
    if combined.empty:
        return None
    denom = combined.iloc[:, 1].abs().replace(0, np.nan)
    rel = (combined.iloc[:, 0] - combined.iloc[:, 1]).abs() / denom
    return float(rel.max()) if len(rel) else None


def verify_indicators(
    df: pd.DataFrame, tolerance: float = 1e-6
) -> dict[str, VerificationResult]:
    """Cross-check ``eti.data.add_indicators`` against independent references.

    Returns a dict keyed by indicator name. A result is within tolerance when
    the maximum relative difference between engine and reference is below
    ``tolerance`` (default 1e-6, i.e. ~0.0001%).
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for verification: {sorted(missing)}")

    engine = engine_add_indicators(df)
    close = df["close"]
    returns = reference_returns(close)

    checks: dict[str, VerificationResult] = {}

    # SMA 20 / 50 / 200
    for n in (20, 50, 200):
        col = f"sma{n}"
        ref = reference_sma(close, n)
        checks[col] = VerificationResult(
            indicator=col,
            max_abs_diff=_max_abs_diff(engine[col], ref),
            max_rel_diff=_max_rel_diff(engine[col], ref),
            within_tolerance=_max_rel_diff(engine[col], ref) is not None
            and _max_rel_diff(engine[col], ref) < tolerance,
            non_nan_count_engine=int(engine[col].notna().sum()),
            non_nan_count_reference=int(ref.notna().sum()),
            note="Simple moving average; engine and reference must agree.",
        )

    # RSI 14
    ref_rsi = reference_rsi(close, 14)
    checks["rsi14"] = VerificationResult(
        indicator="rsi14",
        max_abs_diff=_max_abs_diff(engine["rsi14"], ref_rsi),
        max_rel_diff=_max_rel_diff(engine["rsi14"], ref_rsi),
        within_tolerance=_max_rel_diff(engine["rsi14"], ref_rsi) is not None
        and _max_rel_diff(engine["rsi14"], ref_rsi) < tolerance,
        non_nan_count_engine=int(engine["rsi14"].notna().sum()),
        non_nan_count_reference=int(ref_rsi.notna().sum()),
        note="Wilder's RSI; NaN where average loss is zero (no declines).",
    )

    # ATR 14
    ref_atr = reference_atr(df, 14)
    checks["atr14"] = VerificationResult(
        indicator="atr14",
        max_abs_diff=_max_abs_diff(engine["atr14"], ref_atr),
        max_rel_diff=_max_rel_diff(engine["atr14"], ref_atr),
        within_tolerance=_max_rel_diff(engine["atr14"], ref_atr) is not None
        and _max_rel_diff(engine["atr14"], ref_atr) < tolerance,
        non_nan_count_engine=int(engine["atr14"].notna().sum()),
        non_nan_count_reference=int(ref_atr.notna().sum()),
        note="Wilder's ATR over true range.",
    )

    # Volatility 20 (engine stores return_pct = pct_change()*100, so scale ref by 100)
    ref_vol = reference_volatility(returns * 100.0, 20)
    checks["volatility20_pct"] = VerificationResult(
        indicator="volatility20_pct",
        max_abs_diff=_max_abs_diff(engine["volatility20_pct"], ref_vol),
        max_rel_diff=_max_rel_diff(engine["volatility20_pct"], ref_vol),
        within_tolerance=_max_rel_diff(engine["volatility20_pct"], ref_vol) is not None
        and _max_rel_diff(engine["volatility20_pct"], ref_vol) < tolerance,
        non_nan_count_engine=int(engine["volatility20_pct"].notna().sum()),
        non_nan_count_reference=int(ref_vol.notna().sum()),
        note="Rolling std of bar-to-bar returns (engine in %, reference scaled to %).",
    )

    # Returns
    ref_ret = reference_returns(close)
    checks["return_pct"] = VerificationResult(
        indicator="return_pct",
        max_abs_diff=_max_abs_diff(engine["return_pct"], ref_ret * 100.0),
        max_rel_diff=_max_rel_diff(engine["return_pct"], ref_ret * 100.0),
        within_tolerance=_max_rel_diff(engine["return_pct"], ref_ret * 100.0) is not None
        and _max_rel_diff(engine["return_pct"], ref_ret * 100.0) < tolerance,
        non_nan_count_engine=int(engine["return_pct"].notna().sum()),
        non_nan_count_reference=int(ref_ret.notna().sum()),
        note="Bar-to-bar simple returns.",
    )

    return checks


def all_within_tolerance(results: dict[str, VerificationResult]) -> bool:
    """True if every indicator verification is within tolerance."""
    return all(r.within_tolerance for r in results.values())


def insufficient_history_warning(df: pd.DataFrame, min_samples: int = 200) -> str | None:
    """Return a warning string if the dataset is too small for regime analysis."""
    if len(df) < min_samples:
        return (
            f"Insufficient history: {len(df)} real samples < {min_samples} required. "
            "Regime labels and long-window indicators (SMA200) are unreliable."
        )
    return None
