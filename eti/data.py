from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

COLUMNS = ["time", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def load_bars(path: str | Path) -> pd.DataFrame:
    """Load a headerless tab/comma-delimited OHLCV file.

    Source timestamps remain timezone-naive until the data provider's timezone
    and candle conventions have been verified. This is intentional.
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None, names=COLUMNS)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    for col in PRICE_COLUMNS + ["volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=COLUMNS).sort_values("time").reset_index(drop=True)
    return df


def validate_bars(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0, "start": None, "end": None,
            "duplicate_timestamps": 0, "missing_values": 0,
            "invalid_ohlc_rows": 0, "negative_volume_rows": 0,
            "largest_time_gap_seconds": None, "status": "empty",
        }
    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df[PRICE_COLUMNS] <= 0).any(axis=1)
    )
    gaps = df["time"].diff().dropna().dt.total_seconds()
    dupes = int(df["time"].duplicated().sum())
    missing = int(df[COLUMNS].isna().sum().sum())
    invalid = int(invalid_ohlc.sum())
    negative_volume = int((df["volume"] < 0).sum())
    status = "pass" if not any([dupes, missing, invalid, negative_volume]) else "fail"
    return {
        "rows": int(len(df)),
        "start": str(df["time"].min()),
        "end": str(df["time"].max()),
        "duplicate_timestamps": dupes,
        "missing_values": missing,
        "invalid_ohlc_rows": invalid,
        "negative_volume_rows": negative_volume,
        "largest_time_gap_seconds": float(gaps.max()) if len(gaps) else None,
        "status": status,
        "timezone": "unverified_naive",
        "note": "Largest time gaps need market-session-aware classification; not every gap is a data outage.",
    }


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    close = x["close"]
    x["return_pct"] = close.pct_change() * 100
    for n in (20, 50, 200):
        x[f"sma{n}"] = close.rolling(n, min_periods=n).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi14"] = 100 - (100 / (1 + rs))
    prev_close = close.shift(1)
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev_close).abs(),
        (x["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    x["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    x["volatility20_pct"] = x["return_pct"].rolling(20).std()
    return x


def resample_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate bars using source-local naive time boundaries.

    Do not use for production signals until source timezone and session
    conventions have been confirmed.
    """
    if df.empty:
        return pd.DataFrame(columns=COLUMNS)
    x = df.set_index("time").sort_index()
    out = x.resample(rule, label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=PRICE_COLUMNS).reset_index()
    return out


def summary(df: pd.DataFrame, label: str) -> dict:
    if df.empty:
        raise ValueError("Cannot summarize an empty dataset")
    x = add_indicators(df)
    latest = x.iloc[-1]
    first_close = float(x["close"].iloc[0])
    latest_close = float(latest["close"])
    def number(v):
        return float(v) if pd.notna(v) and np.isfinite(v) else None
    return {
        "instrument": "EUR/USD",
        "dataset": label,
        "rows": int(len(x)),
        "start": str(x["time"].min()),
        "end": str(x["time"].max()),
        "first_close": first_close,
        "latest_close": latest_close,
        "change_from_first_pct": (latest_close / first_close - 1) * 100,
        "latest_sma20": number(latest.get("sma20")),
        "latest_sma50": number(latest.get("sma50")),
        "latest_sma200": number(latest.get("sma200")),
        "latest_rsi14": number(latest.get("rsi14")),
        "latest_atr14": number(latest.get("atr14")),
        "data_quality": validate_bars(df),
        "interpretation": "Descriptive historical analysis only; not a live quote or trade recommendation.",
    }
