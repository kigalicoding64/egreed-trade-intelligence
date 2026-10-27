"""
Tests for Milestone 6 — Research copilot.

Key guarantees tested:
- No LLM connected by default (NullLLMAdapter, enabled=False).
- All calculations delegated to deterministic software (eti.quant, eti.backtesting).
- Explicit insufficient-evidence response when data is missing/insufficient.
- Evidence bundles carry source + timestamp on every item.
- Risk controls cannot be overridden by the copilot.
- LLM (if configured) only explains; never computes.
"""
import os
from datetime import datetime, timezone

import pandas as pd
import pytest

from eti.copilot import (
    Evidence,
    EvidenceBundle,
    LLMAdapter,
    NullLLMAdapter,
    CopilotConfig,
    DeterministicEngine,
    ResearchCopilot,
)
from eti.quant import reference_sma, reference_rsi, reference_atr


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Null LLM adapter — default, never connected
# ---------------------------------------------------------------------------
def test_null_llm_adapter_not_enabled():
    adapter = NullLLMAdapter()
    assert adapter.name == "null"
    assert adapter.enabled is False
    assert adapter.complete("sys", "user") == "__NULL_LLM__"


def test_copilot_default_has_no_llm():
    copilot = ResearchCopilot()
    assert copilot.llm_enabled() is False
    assert isinstance(copilot.llm, NullLLMAdapter)


def test_llm_not_enabled_without_api_key(monkeypatch):
    monkeypatch.delenv("ETI_LLM_API_KEY", raising=False)
    copilot = ResearchCopilot()
    assert copilot.llm_enabled() is False


# ---------------------------------------------------------------------------
# Evidence model
# ---------------------------------------------------------------------------
def test_evidence_has_source_and_timestamp():
    now = datetime.now(UTC)
    e = Evidence(label="SMA 20", value=1.12, source="eti.quant.reference_sma", timestamp=now)
    assert e.label == "SMA 20"
    assert e.value == 1.12
    assert e.source == "eti.quant.reference_sma"
    assert e.timestamp == now
    assert e.computed_by == "deterministic"


# ---------------------------------------------------------------------------
# Deterministic engine — indicator evidence
# ---------------------------------------------------------------------------
def _make_bars(n=300, start=1.10, step=0.0001, oscillate=True):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    if oscillate:
        # Oscillating price so RSI has both gains and losses and SMA crossover
        # produces position changes (trades) in the test period.
        import numpy as np
        close = start + np.cumsum(np.random.default_rng(7).standard_normal(n) * 0.0005)
    else:
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


def test_indicator_evidence_grounded_in_references():
    engine = DeterministicEngine(CopilotConfig())
    df = _make_bars(300)
    bundle = engine.indicator_evidence(df, "EUR/USD")

    assert bundle.sufficient is True
    assert bundle.instrument == "EUR/USD"
    assert bundle.as_of.tzinfo is not None  # timezone-aware

    # Every item has source + timestamp + computed_by=deterministic
    for item in bundle.items:
        assert item.source
        assert item.timestamp.tzinfo is not None
        assert item.computed_by == "deterministic"

    # Values match independent references
    sma20 = bundle.find("SMA 20")
    assert sma20 is not None
    assert sma20.value == pytest.approx(float(reference_sma(df["close"], 20).iloc[-1]))

    rsi14 = bundle.find("RSI 14")
    assert rsi14 is not None
    assert rsi14.value == pytest.approx(float(reference_rsi(df["close"], 14).iloc[-1]))

    atr14 = bundle.find("ATR 14")
    assert atr14 is not None
    assert atr14.value == pytest.approx(float(reference_atr(df, 14).iloc[-1]))


def test_indicator_evidence_insufficient_history():
    engine = DeterministicEngine(CopilotConfig(min_bars_for_indicators=200))
    df = _make_bars(50)  # fewer than 200
    bundle = engine.indicator_evidence(df, "EUR/USD")

    assert bundle.sufficient is False
    assert bundle.insufficiency_reason is not None
    assert "insufficient" in bundle.insufficiency_reason.lower()


def test_indicator_evidence_empty_data():
    engine = DeterministicEngine(CopilotConfig())
    bundle = engine.indicator_evidence(pd.DataFrame(), "EUR/USD")
    assert bundle.sufficient is False
    assert bundle.insufficiency_reason == "No price data available."


# ---------------------------------------------------------------------------
# Copilot explain_indicators
# ---------------------------------------------------------------------------
def test_copilot_explain_indicators_returns_grounded_text():
    copilot = ResearchCopilot()
    df = _make_bars(300)
    text = copilot.explain_indicators(df, "EUR/USD")

    assert "Insufficient evidence" not in text
    assert "SMA 20" in text
    assert "RSI 14" in text
    assert "eti.quant" in text
    assert "UTC" in text
    assert "deterministic" in text.lower() or "deterministic ETI" in text


def test_copilot_explain_indicators_insufficient():
    copilot = ResearchCopilot()
    df = _make_bars(10)  # too few bars
    text = copilot.explain_indicators(df, "EUR/USD")
    assert "Insufficient evidence" in text
    assert "No measurements are fabricated" in text


# ---------------------------------------------------------------------------
# Deterministic engine — backtest evidence
# ---------------------------------------------------------------------------
def test_backtest_evidence_chronological_split():
    engine = DeterministicEngine(CopilotConfig())
    df = _make_bars(300)
    bundle = engine.backtest_evidence(df, "sma_crossover", {"fast_window": 10, "slow_window": 30})

    assert bundle.sufficient is True
    assert bundle.find("Strategy") is not None
    assert bundle.find("Parameters") is not None
    assert bundle.find("Data version") is not None
    assert bundle.find("Test period bars") is not None
    assert bundle.find("Train total return") is not None
    assert bundle.find("Validation total return") is not None
    assert bundle.find("Test total return") is not None

    # Parameters echoed exactly
    params_item = bundle.find("Parameters")
    assert params_item.value == {"fast_window": 10, "slow_window": 30}


def test_backtest_evidence_no_trades_insufficient():
    engine = DeterministicEngine(CopilotConfig())
    df = _make_bars(120)
    bundle = engine.backtest_evidence(df, "no_trade")
    # no_trade executes zero trades -> insufficient
    assert bundle.sufficient is False
    assert "No trades executed" in bundle.insufficiency_reason


# ---------------------------------------------------------------------------
# Risk controls cannot be overridden
# ---------------------------------------------------------------------------
def test_risk_evidence_drawdown_threshold():
    engine = DeterministicEngine(CopilotConfig())
    # Max drawdown worse than -20% -> insufficient
    bundle = engine.risk_evidence({"max_drawdown_pct": -25.0, "trade_count": 50})
    assert bundle.sufficient is False
    assert "exceeds the -20% risk threshold" in bundle.insufficiency_reason

    # Acceptable drawdown -> sufficient
    bundle2 = engine.risk_evidence({"max_drawdown_pct": -5.0, "trade_count": 50})
    assert bundle2.sufficient is True


def test_risk_evidence_few_trades():
    engine = DeterministicEngine(CopilotConfig())
    bundle = engine.risk_evidence({"max_drawdown_pct": -5.0, "trade_count": 2})
    assert bundle.sufficient is False
    assert "not statistically reliable" in bundle.insufficiency_reason


# ---------------------------------------------------------------------------
# LLM adapter injection — never auto-created, never hardcoded
# ---------------------------------------------------------------------------
class RecordingAdapter:
    """Test double: records prompts, returns a fixed explanation."""
    def __init__(self) -> None:
        self.name = "recording"
        self.enabled = True
        self.calls = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return "AI explanation of the evidence."


def test_llm_adapter_injection_and_explanation():
    adapter = RecordingAdapter()
    copilot = ResearchCopilot(CopilotConfig(llm_adapter=adapter))
    copilot.set_llm_adapter(adapter)

    # Without API key, llm_enabled is False even though adapter.enabled=True
    assert copilot.llm_enabled() is False

    # With API key, LLM explains on top of evidence
    os.environ["ETI_LLM_API_KEY"] = "test-key-123"
    try:
        assert copilot.llm_enabled() is True
        df = _make_bars(300)
        text = copilot.explain_indicators(df, "EUR/USD")
        assert "AI explanation of the evidence." in text
        assert len(adapter.calls) >= 1
        # System prompt forbids computing/overriding
        sys_prompt = adapter.calls[-1][0]
        assert "NEVER compute" in sys_prompt or "never compute" in sys_prompt.lower()
    finally:
        del os.environ["ETI_LLM_API_KEY"]


def test_llm_only_explains_never_computes():
    adapter = RecordingAdapter()
    copilot = ResearchCopilot(CopilotConfig(llm_adapter=adapter))
    os.environ["ETI_LLM_API_KEY"] = "k"
    try:
        copilot.explain_indicators(_make_bars(300), "EUR/USD")
        user_prompt = adapter.calls[-1][1]
        # The prompt must not ask the LLM to compute anything
        assert "compute" not in user_prompt.lower() or "do not compute" in user_prompt.lower()
    finally:
        del os.environ["ETI_LLM_API_KEY"]


# ---------------------------------------------------------------------------
# ResearchCopilot assemble API surface
# ---------------------------------------------------------------------------
def test_copilot_has_assess_risk():
    copilot = ResearchCopilot()
    text = copilot.assess_risk({"max_drawdown_pct": -5.0, "sharpe": 1.0, "trade_count": 50})
    assert "Max drawdown" in text
    assert "Sharpe" in text
