from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import COLUMNS, PRICE_COLUMNS, load_bars, validate_bars

# Provisional FX session assumptions (timezone unverified).
# FX spot market generally pauses over the weekend. These durations are
# heuristics and MUST be verified against the source timezone and calendar
# before any gap is labelled an outage.
# The window (36-84 h) covers both intraday weekend closures (~48 h) and
# daily Friday-close-to-Monday-open gaps (~72 h calendar).
PROVISIONAL_WEEKEND_GAP_HOURS = (36.0, 84.0)
SUSPICIOUS_RETURN_THRESHOLD = 0.05  # 5 % bar-to-bar close change (implausible for FX)


def compute_file_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 hash of a source file for provenance tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def check_source_time_order(path: str | Path) -> dict[str, Any]:
    """Check whether timestamps in the raw source file are non-decreasing.

    Reads only the timestamp column so the full frame is not re-parsed.
    Does not modify load_bars behaviour.
    """
    raw = pd.read_csv(path, sep=None, engine="python", header=None, usecols=[0])
    times = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    diffs = times.diff().dropna()
    out_of_order = int((diffs < pd.Timedelta(0)).sum())
    return {
        "rows_read": int(len(times)),
        "out_of_order_rows": out_of_order,
        "is_monotonic_non_decreasing": out_of_order == 0,
        "first_raw_time": str(times.iloc[0]) if len(times) else None,
        "last_raw_time": str(times.iloc[-1]) if len(times) else None,
    }


def detect_suspicious_price_changes(
    df: pd.DataFrame, threshold: float = SUSPICIOUS_RETURN_THRESHOLD
) -> dict[str, Any]:
    """Flag bar-to-bar close changes exceeding threshold (fraction, e.g. 0.05 = 5 %).

    Never silently replaces flagged values; only reports them for review.
    """
    if df.empty or len(df) < 2:
        return {
            "suspicious_return_rows": 0,
            "max_abs_return_pct": None,
            "threshold_pct": threshold * 100,
            "first_suspicious_index": None,
        }
    rets = df["close"].pct_change().abs()
    mask = rets > threshold
    idx = int(mask.idxmax()) if mask.any() else -1
    return {
        "suspicious_return_rows": int(mask.sum()),
        "max_abs_return_pct": float(rets.max() * 100) if len(rets) else None,
        "threshold_pct": threshold * 100,
        "first_suspicious_index": idx if idx >= 0 else None,
    }


def _classify_single_gap(
    gap_seconds: float,
    expected_interval_seconds: float | None,
    weekend_gap_hours: tuple[float, float],
) -> dict[str, Any]:
    lo_h, hi_h = weekend_gap_hours
    weekend_lo = lo_h * 3600.0
    weekend_hi = hi_h * 3600.0

    if (
        expected_interval_seconds
        and gap_seconds <= expected_interval_seconds * 1.5
    ):
        return {
            "classification": "normal_interval",
            "confidence": "high",
            "reason": "Gap matches expected bar interval.",
        }

    if weekend_lo <= gap_seconds <= weekend_hi:
        return {
            "classification": "likely_weekend_or_scheduled_closure_provisional",
            "confidence": "medium",
            "reason": (
                f"Gap duration ({gap_seconds / 3600:.1f} h) is consistent with a "
                "weekend or scheduled FX market closure. Timezone unverified."
            ),
        }

    return {
        "classification": "potential_data_gap_review",
        "confidence": "low",
        "reason": (
            "Gap larger than expected interval and not consistent with a "
            "provisional weekend closure. Requires manual review against the "
            "market calendar and source timezone."
        ),
    }


def analyze_gaps(
    df: pd.DataFrame,
    expected_interval_seconds: float | None = None,
    weekend_gap_hours: tuple[float, float] = PROVISIONAL_WEEKEND_GAP_HOURS,
) -> dict[str, Any]:
    """Analyze timestamp gaps with provisional session-aware classification.

    Does NOT treat weekends or normal FX closures as feed failures. Large gaps
    are classified provisionally; the source timezone and market calendar must
    be verified before any gap is labelled an outage.
    """
    if df.empty or len(df) < 2:
        return {
            "gaps": [],
            "total_gaps": 0,
            "largest_gap_seconds": None,
            "expected_interval_seconds": expected_interval_seconds,
            "weekend_gap_hours": list(weekend_gap_hours),
            "classification_note": (
                "Insufficient data for gap analysis. Timezone and session "
                "conventions are unverified."
            ),
        }

    times = df["time"].reset_index(drop=True)
    diffs_seconds = times.diff().dropna().dt.total_seconds().reset_index(drop=True)
    positive = diffs_seconds[diffs_seconds > 0]
    if len(positive) == 0:
        return {
            "gaps": [],
            "total_gaps": 0,
            "largest_gap_seconds": None,
            "expected_interval_seconds": None,
            "weekend_gap_hours": list(weekend_gap_hours),
            "classification_note": (
                "No positive time intervals found (empty or all-duplicate data). "
                "Timezone and session conventions are unverified."
            ),
        }
    # Base granularity = smallest positive interval present. Gaps are intervals
    # clearly larger than the base; they are never assumed to be outages.
    expected_interval_seconds = float(positive.min())

    gaps: list[dict[str, Any]] = []
    for pos, gap in enumerate(diffs_seconds, start=1):
        if gap <= 0:
            continue  # duplicate / zero-length gap
        # Only intervals clearly larger than the base granularity count as gaps.
        if expected_interval_seconds and gap <= expected_interval_seconds * 1.5:
            continue
        classification = _classify_single_gap(
            float(gap), expected_interval_seconds, weekend_gap_hours
        )
        gaps.append(
            {
                "later_index": int(pos),
                "earlier_time": str(times.iloc[pos - 1]),
                "later_time": str(times.iloc[pos]),
                "gap_seconds": float(gap),
                "gap_hours": round(float(gap) / 3600.0, 3),
                "classification": classification["classification"],
                "confidence": classification["confidence"],
                "reason": classification["reason"],
            }
        )

    gaps.sort(key=lambda g: g["gap_seconds"], reverse=True)
    return {
        "gaps": gaps,
        "total_gaps": len(gaps),
        "largest_gap_seconds": float(diffs_seconds.max()) if len(diffs_seconds) else None,
        "expected_interval_seconds": expected_interval_seconds,
        "weekend_gap_hours": list(weekend_gap_hours),
        "classification_note": (
            "Provisional classification only. Source timezone, candle boundary, "
            "and FX session calendar are unverified. Weekends and scheduled "
            "closures are NOT automatically treated as feed failures."
        ),
    }


@dataclass
class DataQualityReport:
    source: dict[str, Any]
    validation: dict[str, Any]
    gaps: dict[str, Any]
    suspicious_prices: dict[str, Any]
    order_check: dict[str, Any]
    documentation: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_data_quality_report(
    path: str | Path,
    df: pd.DataFrame,
    suspicious_threshold: float = SUSPICIOUS_RETURN_THRESHOLD,
) -> DataQualityReport:
    """Build a complete machine-readable data-quality report for one dataset."""
    path = Path(path)
    file_hash = compute_file_hash(path)
    order = check_source_time_order(path)
    validation = validate_bars(df)
    gaps = analyze_gaps(df)
    suspicious = detect_suspicious_price_changes(
        df, threshold=suspicious_threshold
    )

    source: dict[str, Any] = {
        "path": str(path.resolve()),
        "name": path.name,
        "sha256": file_hash,
        "rows": int(len(df)),
        "first_timestamp": str(df["time"].min()) if len(df) else None,
        "last_timestamp": str(df["time"].max()) if len(df) else None,
        "timezone": "unverified_naive",
        "candle_boundary": "unverified",
        "volume_semantics": "unverified",
        "source_identity": "unverified",
    }

    documentation: dict[str, Any] = {
        "verified": [],
        "unknown": [
            "source_timestamp_timezone",
            "candle_boundary_convention",
            "volume_semantics_tick_vs_actual",
            "source_provider_identity",
        ],
        "assumed": [
            "headerless_tab_delimited_ohlcv_format",
            "columns_time_open_high_low_close_volume",
            "daily_candle_boundary_at_00_00_source_local_provisional",
            "fx_weekend_closure_duration_provisional",
        ],
        "notes": [
            "Original source file is unchanged; this report is a derived artifact.",
            "Timestamps are parsed as timezone-naive until the provider timezone "
            "and candle conventions are verified.",
            "Gap classification is provisional; weekends/scheduled closures are "
            "not automatically treated as feed failures.",
        ],
    }

    return DataQualityReport(
        source=source,
        validation=validation,
        gaps=gaps,
        suspicious_prices=suspicious,
        order_check=order,
        documentation=documentation,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def format_human_report(report: DataQualityReport) -> str:
    """Human-readable rendering of a DataQualityReport."""
    s = report.source
    v = report.validation
    g = report.gaps
    p = report.suspicious_prices
    o = report.order_check

    lines: list[str] = [
        "=" * 70,
        "ETI DATA QUALITY REPORT",
        "=" * 70,
        f"Generated: {report.generated_at}",
        f"Source: {s['name']}",
        f"Path: {s['path']}",
        f"SHA-256: {s['sha256']}",
        f"Rows: {s['rows']}",
        f"First timestamp: {s['first_timestamp']}",
        f"Last timestamp: {s['last_timestamp']}",
        f"Timezone: {s['timezone']}",
        "",
        "--- SOURCE IDENTITY / CONVENTIONS ---",
        f"Source identity: {s['source_identity']}",
        f"Candle boundary: {s['candle_boundary']}",
        f"Volume semantics: {s['volume_semantics']}",
        "",
        "--- DOCUMENTATION ---",
        "Verified: " + (", ".join(report.documentation["verified"]) or "none"),
        "Unknown: " + ", ".join(report.documentation["unknown"]),
        "Assumed: " + ", ".join(report.documentation["assumed"]),
        "",
        "--- VALIDATION ---",
        f"Status: {v['status']}",
        f"Duplicate timestamps: {v['duplicate_timestamps']}",
        f"Missing values: {v['missing_values']}",
        f"Invalid OHLC rows: {v['invalid_ohlc_rows']}",
        f"Negative volume rows: {v['negative_volume_rows']}",
        f"Largest time gap (s): {v['largest_time_gap_seconds']}",
        "",
        "--- SOURCE ORDER CHECK ---",
        f"Rows read: {o['rows_read']}",
        f"Out-of-order rows: {o['out_of_order_rows']}",
        f"Monotonic non-decreasing: {o['is_monotonic_non_decreasing']}",
        "",
        "--- SUSPICIOUS PRICE CHANGES ---",
        f"Threshold: {p['threshold_pct']}%",
        f"Suspicious rows: {p['suspicious_return_rows']}",
        f"Max abs return: {p['max_abs_return_pct']}%",
        "",
        "--- GAP ANALYSIS ---",
        f"Total gaps: {g['total_gaps']}",
        f"Largest gap (s): {g['largest_gap_seconds']}",
        f"Expected interval (s): {g['expected_interval_seconds']}",
    ]
    if g["gaps"]:
        lines.append("")
        lines.append("Top gaps:")
        for gap in g["gaps"][:10]:
            lines.append(
                f"  {gap['gap_hours']:.2f} h | {gap['classification']} "
                f"({gap['confidence']}) | {gap['reason'][:80]}"
            )
    lines.append("")
    lines.append(g["classification_note"])
    lines.append("")
    for note in report.documentation["notes"]:
        lines.append(f"  - {note}")
    lines.append("=" * 70)
    return "\n".join(lines)


def write_report(report: DataQualityReport, out_dir: str | Path) -> Path:
    """Write JSON and human-readable report files. Returns the JSON path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(report.source["name"]).stem
    json_path = out_dir / f"{stem}_data_quality.json"
    json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    txt_path = out_dir / f"{stem}_data_quality.txt"
    txt_path.write_text(format_human_report(report), encoding="utf-8")
    return json_path


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m eti.data_quality <CSV_PATH> <OUT_DIR>"""
    import sys

    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if len(argv) != 2:
        print("Usage: python -m eti.data_quality <CSV_PATH> <OUT_DIR>", file=sys.stderr)
        return 2
    csv_path, out_dir = argv
    try:
        df = load_bars(csv_path)
    except FileNotFoundError as exc:
        print(f"Error: source file not found: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface load errors to the operator
        print(f"Error loading {csv_path}: {exc}", file=sys.stderr)
        return 1

    report = build_data_quality_report(csv_path, df)
    json_path = write_report(report, out_dir)
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
