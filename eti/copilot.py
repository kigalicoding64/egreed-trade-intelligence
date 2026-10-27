"""
Research copilot — Milestone 6.

Design principles (per ETI engineering standards):
- The copilot is added ONLY after the quantitative and data layers are reliable.
- ALL calculations (indicators, exposure, risk, backtest metrics) are performed
  by deterministic software (eti.quant, eti.backtesting). The language model
  may explain results but MUST NOT fabricate measurements or override controls.
- Responses are grounded in actual system data; every cited value identifies its
  source and timestamp.
- An explicit insufficient-evidence response is returned when data is missing,
  stale, or insufficient — never a hallucinated answer.
- Risk controls cannot be overridden by the AI copilot.
- No LLM is connected by default. An optional LLM adapter is wired through an
  environment variable (ETI_LLM_API_KEY) and a registered adapter; credentials
  are never hardcoded.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

import pandas as pd

from eti.quant import (
    reference_sma,
    reference_rsi,
    reference_atr,
    reference_returns,
    reference_volatility,
    classify_regime,
    verify_indicators,
    all_within_tolerance,
    insufficient_history_warning,
)
from eti.backtesting import (
    CostModel,
    run_split_backtest,
    never_claim_profitable,
)


# ---------------------------------------------------------------------------
# Evidence model — every cited value carries source + timestamp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Evidence:
    """A single piece of verified evidence used to ground a copilot response.

    ``computed_by`` is always "deterministic" for values calculated by ETI
    software; it is never "llm" for measurements.
    """
    label: str                      # human-readable label, e.g. "SMA 20"
    value: object                   # the computed value
    source: str                     # module/function that produced it
    timestamp: datetime             # when the evidence was computed (UTC)
    computed_by: str = "deterministic"
    unit: str | None = None         # e.g. "%", "bars", "bps"
    notes: str | None = None


@dataclass(frozen=True)
class EvidenceBundle:
    """A collection of evidence items supporting one copilot response."""
    instrument: str
    as_of: datetime                 # time the bundle was assembled (UTC)
    items: list[Evidence] = field(default_factory=list)
    sufficient: bool = True         # whether evidence is sufficient to answer
    insufficiency_reason: str | None = None

    def find(self, label: str) -> Evidence | None:
        for item in self.items:
            if item.label == label:
                return item
        return None


# ---------------------------------------------------------------------------
# LLM adapter protocol — optional, never connected by default
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMAdapter(Protocol):
    """Protocol for an optional language-model adapter.

    The copilot never depends on a concrete LLM implementation. A null adapter
    is used by default, so ETI functions with zero external AI calls.
    """

    name: str
    enabled: bool

    def complete(self, system_prompt: str, user_prompt: str) -> str: ...


class NullLLMAdapter:
    """Default adapter: no external AI. Returns a deterministic not-available
    marker so the copilot can produce a grounded, insufficient-evidence
    response without calling any model."""
    name = "null"
    enabled = False

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return "__NULL_LLM__"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopilotConfig:
    max_quote_staleness_seconds: float = 300.0
    min_bars_for_indicators: int = 50
    min_bars_for_regime: int = 200
    cost_model: CostModel = field(default_factory=CostModel)
    periods_per_year: float | None = 252
    llm_api_key_env: str = "ETI_LLM_API_KEY"
    llm_adapter: LLMAdapter | None = None  # injected, never auto-created

    def llm_api_key(self) -> str | None:
        return os.environ.get(self.llm_api_key_env)


# ---------------------------------------------------------------------------
# Deterministic evidence assembly (no LLM involved in calculations)
# ---------------------------------------------------------------------------


class DeterministicEngine:
    """Assembles evidence using only deterministic quantitative software.

    Never fabricates values. Returns NaN/None for unavailable data and marks
    the bundle insufficient when required history is missing.
    """

    def __init__(self, config: CopilotConfig) -> None:
        self.config = config

    def indicator_evidence(self, df: pd.DataFrame, instrument: str) -> EvidenceBundle:
        now = datetime.now(timezone.utc)
        items: list[Evidence] = []
        sufficient = True
        reasons: list[str] = []

        if df is None or len(df) == 0:
            return EvidenceBundle(
                instrument=instrument,
                as_of=now,
                items=[],
                sufficient=False,
                insufficiency_reason="No price data available.",
            )

        close = df["close"]

        # SMA 20 / 50 / 200
        for n in (20, 50, 200):
            sma = reference_sma(close, n)
            last = float(sma.iloc[-1]) if pd.notna(sma.iloc[-1]) else None
            if last is None:
                sufficient = False
                reasons.append(f"SMA {n} not available (insufficient history).")
            items.append(
                Evidence(
                    label=f"SMA {n}",
                    value=last,
                    source="eti.quant.reference_sma",
                    timestamp=now,
                    unit="price",
                )
            )

        # RSI 14
        rsi = reference_rsi(close, 14)
        last_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else None
        if last_rsi is None:
            sufficient = False
            reasons.append("RSI 14 not available (insufficient history / no declining bars).")
        items.append(
            Evidence(label="RSI 14", value=last_rsi, source="eti.quant.reference_rsi",
                     timestamp=now, unit="index (0-100)")
        )

        # ATR 14
        atr = reference_atr(df, 14)
        last_atr = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else None
        if last_atr is None:
            sufficient = False
            reasons.append("ATR 14 not available (insufficient history).")
        items.append(
            Evidence(label="ATR 14", value=last_atr, source="eti.quant.reference_atr",
                     timestamp=now, unit="price")
        )

        # Volatility 20
        rets = reference_returns(close)
        vol = reference_volatility(rets, 20)
        last_vol = float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else None
        items.append(
            Evidence(label="Volatility 20", value=last_vol, source="eti.quant.reference_volatility",
                     timestamp=now, unit="% per bar")
        )

        # Last close
        last_close = float(close.iloc[-1])
        items.append(
            Evidence(label="Last close", value=last_close, source="eti.data.load_bars",
                     timestamp=now, unit="price")
        )

        # Regime
        try:
            regime_res = classify_regime(df)
            current_regime = regime_res.regime.iloc[-1] if len(regime_res.regime) else "insufficient_data"
            items.append(
                Evidence(label="Regime", value=str(current_regime),
                         source="eti.quant.classify_regime", timestamp=now)
            )
            if str(current_regime) == "insufficient_data":
                sufficient = False
                reasons.append("Regime not available (insufficient history < 200 bars).")
        except Exception as exc:  # noqa: BLE001 - surface as insufficient evidence
            sufficient = False
            reasons.append(f"Regime calculation unavailable: {exc}")

        # Engine self-verification
        try:
            checks = verify_indicators(df)
            engine_ok = all_within_tolerance(checks)
            items.append(
                Evidence(label="Engine self-check", value=engine_ok,
                         source="eti.quant.verify_indicators", timestamp=now)
            )
            if not engine_ok:
                sufficient = False
                reasons.append("Indicator engine failed self-verification.")
        except Exception as exc:  # noqa: BLE001
            sufficient = False
            reasons.append(f"Indicator verification unavailable: {exc}")

        history_warn = insufficient_history_warning(df, self.config.min_bars_for_indicators)
        if history_warn:
            sufficient = False
            reasons.append(history_warn)

        return EvidenceBundle(
            instrument=instrument,
            as_of=now,
            items=items,
            sufficient=sufficient,
            insufficiency_reason="; ".join(reasons) if reasons else None,
        )

    def backtest_evidence(
        self,
        bars: pd.DataFrame,
        strategy_name: str,
        params: dict | None = None,
    ) -> EvidenceBundle:
        now = datetime.now(timezone.utc)
        items: list[Evidence] = []
        reasons: list[str] = []
        sufficient = True

        try:
            result = run_split_backtest(
                bars,
                strategy_name,
                params=params or {},
                cost_model=self.config.cost_model,
                periods_per_year=self.config.periods_per_year,
            )
        except ValueError as exc:
            return EvidenceBundle(
                instrument=strategy_name,
                as_of=now,
                items=[],
                sufficient=False,
                insufficiency_reason=str(exc),
            )

        test = result.metrics["test"]
        train = result.metrics["train"]
        val = result.metrics["validation"]

        items.append(
            Evidence(label="Strategy", value=result.strategy,
                     source="eti.backtesting.run_split_backtest", timestamp=now)
        )
        items.append(
            Evidence(label="Parameters", value=result.parameters,
                     source="eti.backtesting.run_split_backtest", timestamp=now)
        )
        items.append(
            Evidence(label="Data version", value=result.data_version,
                     source="eti.backtesting.data_version", timestamp=now)
        )
        items.append(
            Evidence(label="Cost model (total bps)", value=result.cost_model.total_bps,
                     source="eti.backtesting.CostModel", timestamp=now, unit="bps")
        )
        items.append(
            Evidence(label="Test total return", value=test["total_return_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Test buy & hold return", value=test["buy_hold_return_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Test max drawdown", value=test["max_drawdown_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Test Sharpe", value=test["sharpe"],
                     source="eti.backtesting.run_split_backtest", timestamp=now)
        )
        items.append(
            Evidence(label="Test trade count", value=test["trade_count"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="trades")
        )
        items.append(
            Evidence(label="Test exposure", value=test["exposure_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Test period bars", value=result.test_period.rows if result.test_period else None,
                     source="eti.backtesting.chronological_split", timestamp=now, unit="bars")
        )
        items.append(
            Evidence(label="Train total return", value=train["total_return_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Validation total return", value=val["total_return_pct"],
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )

        # Insufficient-evidence checks
        if test["trade_count"] == 0:
            sufficient = False
            reasons.append("No trades executed in test period; results not meaningful.")
        if result.test_period and result.test_period.rows < 50:
            sufficient = False
            reasons.append(f"Test period too short ({result.test_period.rows} bars) for reliable evaluation.")
        # Train/test divergence warning (not a hard failure, but noted)
        if train["total_return_pct"] is not None and test["total_return_pct"] is not None:
            divergence = abs(train["total_return_pct"] - test["total_return_pct"])
            items.append(
                Evidence(label="Train/test return divergence", value=round(divergence, 2),
                         source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
            )

        return EvidenceBundle(
            instrument=strategy_name,
            as_of=now,
            items=items,
            sufficient=sufficient,
            insufficiency_reason="; ".join(reasons) if reasons else None,
        )

    def risk_evidence(self, test_metrics: dict) -> EvidenceBundle:
        """Assemble risk evidence; risk controls cannot be overridden by the LLM."""
        now = datetime.now(timezone.utc)
        items: list[Evidence] = []
        reasons: list[str] = []
        sufficient = True

        items.append(
            Evidence(label="Max drawdown", value=test_metrics.get("max_drawdown_pct"),
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Sharpe", value=test_metrics.get("sharpe"),
                     source="eti.backtesting.run_split_backtest", timestamp=now)
        )
        items.append(
            Evidence(label="Sortino", value=test_metrics.get("sortino"),
                     source="eti.backtesting.run_split_backtest", timestamp=now)
        )
        items.append(
            Evidence(label="Calmar", value=test_metrics.get("calmar_pct"),
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Win rate", value=test_metrics.get("win_rate_pct"),
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="%")
        )
        items.append(
            Evidence(label="Trade count", value=test_metrics.get("trade_count"),
                     source="eti.backtesting.run_split_backtest", timestamp=now, unit="trades")
        )

        dd = test_metrics.get("max_drawdown_pct")
        if dd is not None and dd < -20.0:
            sufficient = False
            reasons.append(f"Max drawdown {dd:.2f}% exceeds the -20% risk threshold.")
        trades = test_metrics.get("trade_count")
        if trades is not None and trades < 5:
            sufficient = False
            reasons.append(f"Only {trades} trades; risk statistics not statistically reliable.")

        return EvidenceBundle(
            instrument="risk",
            as_of=now,
            items=items,
            sufficient=sufficient,
            insufficiency_reason="; ".join(reasons) if reasons else None,
        )


# ---------------------------------------------------------------------------
# Copilot — deterministic core, optional LLM for explanation only
# ---------------------------------------------------------------------------


class ResearchCopilot:
    """Research copilot grounded in deterministic ETI calculations.

    The LLM (if configured) is used ONLY to translate deterministic evidence
    into plain language. It cannot:
      - compute indicators, exposure, risk, or backtest metrics
      - override risk controls
      - fabricate measurements
    When evidence is insufficient, an explicit insufficient-evidence response
    is returned.
    """

    def __init__(self, config: CopilotConfig | None = None) -> None:
        self.config = config or CopilotConfig()
        self.engine = DeterministicEngine(self.config)
        self.llm: LLMAdapter = self.config.llm_adapter or NullLLMAdapter()

    # -- configuration ----------------------------------------------------
    def set_llm_adapter(self, adapter: LLMAdapter) -> None:
        self.llm = adapter

    def llm_enabled(self) -> bool:
        return bool(self.llm and self.llm.enabled and self.config.llm_api_key())

    # -- public queries ---------------------------------------------------
    def explain_indicators(self, df: pd.DataFrame, instrument: str = "EUR/USD") -> str:
        bundle = self.engine.indicator_evidence(df, instrument)
        return self._render_bundle(bundle, "indicators")

    def explain_backtest(
        self,
        bars: pd.DataFrame,
        strategy_name: str,
        params: dict | None = None,
    ) -> str:
        bundle = self.engine.backtest_evidence(bars, strategy_name, params)
        return self._render_bundle(bundle, "backtest")

    def assess_risk(self, test_metrics: dict) -> str:
        bundle = self.engine.risk_evidence(test_metrics)
        return self._render_bundle(bundle, "risk")

    # -- rendering --------------------------------------------------------
    def _render_bundle(self, bundle: EvidenceBundle, kind: str) -> str:
        now_str = bundle.as_of.strftime("%Y-%m-%d %H:%M:%S UTC")
        lines: list[str] = []

        # Insufficient-evidence handling — explicit, never fabricated.
        if not bundle.sufficient:
            lines.append("## Insufficient evidence")
            lines.append(
                f"The copilot cannot answer this {kind} question because: "
                f"{bundle.insufficiency_reason}"
            )
            lines.append(
                "No measurements are fabricated. Please provide more data or "
                "relax constraints and try again."
            )
            if bundle.items:
                lines.append("")
                lines.append("### Available (partial) evidence")
                lines.append(self._evidence_list(bundle.items))
            lines.append("")
            lines.append(
                f"_Evidence assembled at {now_str}. All values computed by "
                "deterministic ETI software._"
            )
            return "\n".join(lines)

        lines.append(f"## {kind.title()} — grounded in deterministic evidence")
        lines.append(f"Instrument: {bundle.instrument}  |  As of: {now_str}")
        lines.append("")
        lines.append(self._evidence_list(bundle.items))

        # Optional LLM explanation — strictly layered on top of evidence.
        if self.llm_enabled():
            try:
                explanation = self._llm_explain(kind, bundle)
                lines.append("")
                lines.append("### Plain-language explanation (AI-generated)")
                lines.append(
                    "_The language model only describes the evidence above; it "
                    "does not compute or override any measurement._"
                )
                lines.append("")
                lines.append(explanation)
            except Exception as exc:  # noqa: BLE001 - never fail the whole response
                lines.append("")
                lines.append(
                    f"_AI explanation unavailable: {exc}. Using deterministic evidence only._"
                )
        else:
            lines.append("")
            lines.append(
                "_No language model configured. Response is deterministic evidence only._"
            )

        lines.append("")
        lines.append(
            "_Every value above is computed by deterministic ETI software "
            "(eti.quant / eti.backtesting) and tagged with its source and timestamp. "
            "The AI copilot does not fabricate measurements or override controls._"
        )
        return "\n".join(lines)

    @staticmethod
    def _evidence_list(items: list[Evidence]) -> str:
        out = []
        for item in items:
            val = item.value
            if isinstance(val, float):
                val = round(val, 6)
            unit = f" {item.unit}" if item.unit else ""
            out.append(
                f"- **{item.label}**: {val}{unit}  _(source: {item.source}, "
                f"computed at {item.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')})_"
            )
        return "\n".join(out)

    def _llm_explain(self, kind: str, bundle: EvidenceBundle) -> str:
        """Ask the LLM to describe the deterministic evidence in plain language.

        The system prompt strictly forbids the model from computing or overriding
        any value; it may only restate what is in the evidence bundle.
        """
        evidence_text = self._evidence_list(bundle.items)
        system_prompt = (
            "You are a research assistant for ETI (Egreed Trade Intelligence), "
            "a quantitative market-research platform. You must NEVER compute, "
            "invent, or override any numerical measurement. All measurements are "
            "provided to you in the evidence bundle and were computed by "
            "deterministic ETI software. Your only job is to restate and explain "
            "those measurements in plain language, clearly identifying each value's "
            "source and timestamp. Do not provide investment advice, do not predict "
            "future prices, and do not claim any strategy is profitable. If the "
            "evidence bundle is insufficient, say so explicitly."
        )
        user_prompt = (
            f"Context: {kind} research for instrument {bundle.instrument}. "
            f"Here is the deterministic evidence bundle (each item includes source "
            f"and computation timestamp):\n\n{evidence_text}\n\n"
            "Please restate these findings in plain language for a researcher. "
            "Identify each value's source and timestamp. Do not compute anything. "
            "Do not give investment advice or predict future prices."
        )
        return self.llm.complete(system_prompt, user_prompt)


__all__ = [
    "Evidence",
    "EvidenceBundle",
    "LLMAdapter",
    "NullLLMAdapter",
    "CopilotConfig",
    "DeterministicEngine",
    "ResearchCopilot",
]
