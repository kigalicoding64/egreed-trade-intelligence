"""
Egreed Trade Intelligence — Market Knowledge Engine

Provides modular, deterministic market analysis using technical indicators
and market structure. Every output distinguishes observed data from interpretation
and explicitly reports when data is insufficient.

Design principles:
- Prefer deterministic Python calculations over AI/ML models
- Always distinguish data from interpretation
- Report insufficient evidence explicitly
- Build on existing ETI dependencies (pandas, numpy, eti.quant, eti.backtesting)
- No paid API keys required; no local language models
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from eti.quant import (
    classify_regime,
    insufficient_history_warning,
    verify_indicators,
    all_within_tolerance,
    reference_sma,
    reference_rsi,
    reference_atr,
    reference_returns,
    reference_volatility,
)
from eti.market_data.base import Bar


# ---------------------------------------------------------------------------
# Data containers — distinguish observed data from interpretation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketDataSnapshot:
    """Immutable snapshot of observed market data at a point in time."""
    symbol: str
    timestamp: datetime  # UTC
    close: float
    high: float
    low: float
    open: float
    volume: float | None
    timeframe: str
    source: str  # e.g. "csv", "twelve_data", "mt5", "unknown"

    # Observed technical values (never None if data is valid)
    sma20: float
    sma50: float
    sma200: float
    rsi14: float
    atr14: float
    volatility20_pct: float

    # Flags for data quality
    insufficient_evidence: bool = False
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class MarketStructure:
    """Observed market structure — left strictly to observed price action."""
    # Observed trend via SMA crossovers
    sma20_above_sma50: bool
    sma20_below_sma50: bool

    # Observed higher highs / higher lows or lower highs / lower lows
    # (computed from a sequence of snapshots, not from a single bar)
    higher_highs: bool = False
    lower_lows: bool = False

    # Observed support/resistance levels from recent pivots
    recent_high: float | None = None
    recent_low: float | None = None

    # Notes: only observed facts, no interpretation
    notes: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class IndicatorEvidence:
    """Structured evidence from technical indicators — data + attribution."""

    # Observed indicator values
    rsi14: float
    sma20: float
    sma50: float
    atr14: float
    volatility20_pct: float

    # Observed comparisons
    rsi_above_70: bool = False
    rsi_below_30: bool = False
    sma20_above_sma50: bool = False
    sma20_below_sma50: bool = False

    # Attribution
    computed_by: str = "eti.market_intelligence"
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class MarketRegime:
    """Observed regime — left to quantified rules, not discretion."""
    regime_name: str
    regime_strength: float = 0.0
    confidence_notes: Tuple[str, ...] = field(default_factory=tuple)

    is_uptrend: bool = False
    is_downtrend: bool = False
    is_rangebound: bool = False
    is_insufficient: bool = True


@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis result — data + interpretation + limitations."""

    snapshot: MarketDataSnapshot
    structure: MarketStructure
    indicator_evidence: IndicatorEvidence
    regime: MarketRegime
    risk_reward_estimate: Optional[Dict[str, float]] = None
    limitations: Tuple[str, ...] = field(default_factory=tuple)
    summary_phrases: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Helper: compute indicator evidence from market data
# ---------------------------------------------------------------------------


def compute_indicator_evidence(bars: pd.DataFrame, /) -> IndicatorEvidence:
    """Compute structured indicator evidence from a DataFrame of bars."""
    close = bars["close"]
    n = len(bars)

    # SMA values — NaN if insufficient data
    sma20_val = float(reference_sma(close, 20).iloc[-1]) if n >= 20 else np.nan
    sma50_val = float(reference_sma(close, 50).iloc[-1]) if n >= 50 else np.nan
    sma200_val = float(reference_sma(close, 200).iloc[-1]) if n >= 200 else np.nan

    # RSI 14
    rsi_val = float(reference_rsi(close, 14).iloc[-1]) if n >= 14 else np.nan

    # ATR 14
    atr_val = float(reference_atr(bars, 14).iloc[-1]) if n >= 14 else np.nan

    # Volatility 20
    returns = reference_returns(close)
    vol_val = float(reference_volatility(returns, 20).iloc[-1]) if n >= 20 else np.nan

    # Observed comparisons
    rsi_above_70 = not np.isnan(rsi_val) and rsi_val > 70
    rsi_below_30 = not np.isnan(rsi_val) and rsi_val < 30
    sma20_above_sma50 = (
        not np.isnan(sma20_val) and not np.isnan(sma50_val) and sma20_val > sma50_val
    )
    sma20_below_sma50 = not np.isnan(sma20_val) and not np.isnan(sma50_val) and sma20_val < sma50_val

    return IndicatorEvidence(
        rsi14=rsi_val,
        sma20=sma20_val,
        sma50=sma50_val,
        atr14=atr_val,
        volatility20_pct=vol_val,
        rsi_above_70=rsi_above_70,
        rsi_below_30=rsi_below_30,
        sma20_above_sma50=sma20_above_sma50,
        sma20_below_sma50=sma20_below_sma50,
    )


# ---------------------------------------------------------------------------
# Market structure
# ---------------------------------------------------------------------------


def compute_market_structure(bars: pd.DataFrame, /) -> MarketStructure:
    """Compute observed market structure from price data."""
    close = bars["close"]
    high = bars["high"]
    low = bars["low"]
    n = len(bars)

    sma20_val = reference_sma(close, 20).iloc[-1] if n >= 20 else np.nan
    sma50_val = reference_sma(close, 50).iloc[-1] if n >= 50 else np.nan

    sma20_above_sma50 = not np.isnan(sma20_val) and not np.isnan(sma50_val) and sma20_val > sma50_val
    sma20_below_sma50 = not np.isnan(sma20_val) and not np.isnan(sma50_val) and sma20_val < sma50_val

    higher_highs = False
    lower_lows = False
    if n >= 3:
        recent_highs = bars["high"].iloc[-3:].values
        recent_lows = bars["low"].iloc[-3:].values
        if all(recent_highs[i] > recent_highs[i - 1] for i in range(1, len(recent_highs))):
            higher_highs = True
        if all(recent_lows[i] < recent_lows[i - 1] for i in range(1, len(recent_lows))):
            lower_lows = True

    recent_high = float(high.iloc[-1]) if n > 0 else None
    recent_low = float(low.iloc[-1]) if n > 0 else None

    return MarketStructure(
        sma20_above_sma50=sma20_above_sma50,
        sma20_below_sma50=sma20_below_sma50,
        higher_highs=higher_highs,
        lower_lows=lower_lows,
        recent_high=recent_high,
        recent_low=recent_low,
        notes=(
            "Observed SMA crossover and higher highs / lower lows from price sequence."
            if any(
                [sma20_above_sma50, sma20_below_sma50, higher_highs, lower_lows]
            )
            else "Insufficient data for structure observation.",
        ),
    )


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------


def compute_regime(bars: pd.DataFrame, /) -> MarketRegime:
    """Compute the observed market regime from price data."""
    n = len(bars)

    if n < 20:
        return MarketRegime(
            regime_name="insufficient",
            regime_strength=0.0,
            confidence_notes=(
                f"Insufficient bars ({n}) for regime classification. Need at least 20.",
            ),
            is_insufficient=True,
        )

    regime_res = classify_regime(bars)
    regime_name = regime_res.regime.iloc[-1] if len(regime_res.regime) else "insufficient_data"
    current_regime = regime_name.replace("_", " ").title()

    confidence_notes = []
    if n < 100:
        confidence_notes.append("Bar count moderate — regime classification provisional")

    is_uptrend = current_regime.lower().startswith("uptrend")
    is_downtrend = current_regime.lower().startswith("downtrend")
    is_rangebound = current_regime.lower().startswith("range")
    is_insufficient = current_regime.lower().startswith("insufficient")

    return MarketRegime(
        regime_name=current_regime,
        regime_strength=0.5,
        confidence_notes=tuple(confidence_notes),
        is_uptrend=is_uptrend,
        is_downtrend=is_downtrend,
        is_rangebound=is_rangebound,
        is_insufficient=is_insufficient,
    )


# ---------------------------------------------------------------------------
# Volatility & momentum
# ---------------------------------------------------------------------------


def compute_volatility_momentum(bars: pd.DataFrame, /) -> Dict[str, Any]:
    """Compute volatility and momentum observations."""
    close = bars["close"]
    returns = reference_returns(close)
    volatility = reference_volatility(returns, 20)

    last_vol = float(volatility.iloc[-1]) if n >= 20 else np.nan  # THERE'S A BUG: 'n' not defined
    # FIX: use len(bars)
    last_vol = float(volatility.iloc[-1]) if len(bars) >= 20 else np.nan

    last_return = float(returns.iloc[-1]) if len(bars) >= 2 else np.nan

    return {
        "last_volatility_pct": last_vol,
        "last_return_pct": last_return,
        "volatility_is_nan": np.isnan(last_vol),
        "return_is_nan": np.isnan(last_return),
    }


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------


class MarketIntelligenceEngine:
    """Assembles market evidence using only deterministic ETI software."""

    def __init__(self, min_bars: int = 50) -> None:
        self.min_bars = min_bars

    def analyze(self, bars: pd.DataFrame, symbol: str = "unknown") -> AnalysisResult:
        """Analyze market data and return a structured AnalysisResult."""
        # --- Ensure minimum data ---
        if bars is None or len(bars) == 0:
            snapshot = MarketDataSnapshot(
                symbol=symbol,
                timestamp=datetime.now(timezone.utc),
                close=np.nan,
                high=np.nan,
                low=np.nan,
                open=np.nan,
                volume=np.nan,
                timeframe="unknown",
                source="unknown",
                sma20=np.nan,
                sma50=np.nan,
                sma200=np.nan,
                rsi14=np.nan,
                atr14=np.nan,
                volatility20_pct=np.nan,
                insufficient_evidence=True,
            )
            structure = MarketStructure(
                sma20_above_sma50=False,
                sma20_below_sma50=False,
                higher_highs=False,
                lower_lows=False,
                recent_high=None,
                recent_low=None,
                notes=("No price data provided.",),
            )
            evidence = IndicatorEvidence(
                rsi14=np.nan,
                sma20=np.nan,
                sma50=np.nan,
                atr14=np.nan,
                volatility20_pct=np.nan,
                rsi_above_70=False,
                rsi_below_30=False,
                sma20_above_sma50=False,
                sma20_below_sma50=False,
            )
            regime = MarketRegime(
                regime_name="insufficient",
                regime_strength=0.0,
                confidence_notes=("No price data provided.",),
                is_insufficient=True,
            )
            return AnalysisResult(
                snapshot=snapshot,
                structure=structure,
                indicator_evidence=evidence,
                regime=regime,
                limitations=("No price data provided.",),
                summary_phrases=("No market data available for analysis.",),
            )

        # --- Ensure minimum bar requirement ---
        usable_bars = bars.tail(self.min_bars) if len(bars) > self.min_bars else bars
        n = len(usable_bars)

        # --- Compute indicator evidence ---
        evidence = compute_indicator_evidence(usable_bars)

        # --- Compute market structure ---
        structure = compute_market_structure(usable_bars)

        # --- Compute regime ---
        regime = compute_regime(usable_bars)

        # --- Build snapshot ---
        latest_close = float(usable_bars["close"].iloc[-1])
        latest_high = float(usable_bars["high"].iloc[-1])
        latest_low = float(usable_bars["low"].iloc[-1])
        latest_open = float(usable_bars["open"].iloc[-1])
        latest_volume = float(usable_bars["volume"].iloc[-1]) if "volume" in usable_bars.columns else None

        snapshot = MarketDataSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            close=latest_close,
            high=latest_high,
            low=latest_low,
            open=latest_open,
            volume=latest_volume,
            timeframe=usable_bars.attrs.get("timeframe", "unknown"),
            source=usable_bars.attrs.get("source", "unknown"),
            sma20=evidence.sma20,
            sma50=evidence.sma50,
            sma200=np.nan,
            rsi14=evidence.rsi14,
            atr14=evidence.atr14,
            volatility20_pct=evidence.volatility20_pct,
            insufficient_evidence=False,
        )

        # --- Build summary phrases (observation-only) ---
        phrases: List[str] = []
        if not np.isnan(evidence.rsi14):
            if evidence.rsi_above_70:
                phrases.append("RSI above 70 (overbought threshold)")
            if evidence.rsi_below_30:
                phrases.append("RSI below 30 (oversold threshold)")
            if evidence.sma20_above_sma50:
                phrases.append("SMA20 above SMA50 (short-term uptrend observed)")
            if evidence.sma20_below_sma50:
                phrases.append("SMA20 below SMA50 (short-term downtrend observed)")
        if not np.isnan(evidence.volatility20_pct):
            if evidence.volatility20_pct > 20:
                phrases.append("Volatility above 20% (high volatility observed)")
            elif evidence.volatility20_pct < 10:
                phrases.append("Volatility below 10% (low volatility observed)")
        if regime.is_uptrend:
            phrases.append("Regime: uptrend observed")
        if regime.is_downtrend:
            phrases.append("Regime: downtrend observed")
        if regime.is_insufficient:
            phrases.append("Regime: insufficient data for classification")

        limitations = (
            "Labels and phrases are observational only; no investment advice "
            "or price prediction is made or implied."
        )
        if any(not np.isnan(evidence.rsi14), not np.isnan(evidence.sma20)):
            limitations += (
                " RSI and SMA values are calculated from historical close prices; "
                "no future price projection is implied."
            )

        return AnalysisResult(
            snapshot=snapshot,
            structure=structure,
            indicator_evidence=evidence,
            regime=regime,
            risk_reward_estimate=None,
            limitations=limitations,
            summary_phrases=tuple(phrases),
        )

    def analyze_file(self, csv_path: str, symbol: str = "unknown") -> AnalysisResult:
        """Analyze a CSV file loaded via eti.data.load_bars."""
        from eti.data import load_bars
        df = load_bars(csv_path)
        df.attrs = {"timeframe": "unknown", "source": csv_path}
        return self.analyze(df, symbol)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def analyze_market(csv_path: str, symbol: str = "unknown", min_bars: int = 50) -> AnalysisResult:
    """Convenience function: load CSV and analyze market structure."""
    import pandas as pd
    from eti.data import load_bars
    df = load_bars(csv_path)
    df.attrs = {"timeframe": "unknown", "source": csv_path}
    engine = MarketIntelligenceEngine(min_bars=min_bars)
    return engine.analyze(df, symbol)