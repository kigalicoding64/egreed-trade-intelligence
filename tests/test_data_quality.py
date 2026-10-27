import pandas as pd
import pytest

from eti.data_quality import (
    SUSPICIOUS_RETURN_THRESHOLD,
    analyze_gaps,
    build_data_quality_report,
    check_source_time_order,
    compute_file_hash,
    detect_suspicious_price_changes,
    format_human_report,
    write_report,
)


def _sample_bars(n=120):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    close = [1.10 + i * 0.0001 for i in range(n)]
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": [x + 0.0002 for x in close],
            "low": [x - 0.0002 for x in close],
            "close": close,
            "volume": [100] * n,
        }
    )


def test_compute_file_hash_is_deterministic(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("x\n", encoding="utf-8")
    h1 = compute_file_hash(f)
    h2 = compute_file_hash(f)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_compute_file_hash_changes_with_content(tmp_path):
    f1 = tmp_path / "a.csv"; f1.write_text("a\n", encoding="utf-8")
    f2 = tmp_path / "b.csv"; f2.write_text("b\n", encoding="utf-8")
    assert compute_file_hash(f1) != compute_file_hash(f2)


def test_check_source_time_order_detects_out_of_order(tmp_path):
    # Out of order: 3rd row earlier than 2nd.
    content = (
        "2024-01-01 00:00\t1.1\t1.1\t1.1\t1.1\t100\n"
        "2024-01-01 01:00\t1.1\t1.1\t1.1\t1.1\t100\n"
        "2024-01-01 00:30\t1.1\t1.1\t1.1\t1.1\t100\n"
    )
    f = tmp_path / "ooo.csv"
    f.write_text(content, encoding="utf-8")
    res = check_source_time_order(f)
    assert res["rows_read"] == 3
    assert res["out_of_order_rows"] == 1
    assert res["is_monotonic_non_decreasing"] is False


def test_check_source_time_order_monotonic(tmp_path):
    content = (
        "2024-01-01 00:00\t1.1\t1.1\t1.1\t1.1\t100\n"
        "2024-01-01 01:00\t1.1\t1.1\t1.1\t1.1\t100\n"
        "2024-01-01 02:00\t1.1\t1.1\t1.1\t1.1\t100\n"
    )
    f = tmp_path / "mono.csv"
    f.write_text(content, encoding="utf-8")
    res = check_source_time_order(f)
    assert res["out_of_order_rows"] == 0
    assert res["is_monotonic_non_decreasing"] is True


def test_detect_suspicious_price_changes_none():
    df = _sample_bars(50)
    res = detect_suspicious_price_changes(df)
    assert res["suspicious_return_rows"] == 0
    assert res["threshold_pct"] == SUSPICIOUS_RETURN_THRESHOLD * 100


def test_detect_suspicious_price_changes_flags_jump():
    df = _sample_bars(50)
    df.loc[25, "close"] = 2.0  # ~80% jump, far above 5% threshold
    res = detect_suspicious_price_changes(df)
    assert res["suspicious_return_rows"] >= 1
    assert res["max_abs_return_pct"] is not None
    assert res["max_abs_return_pct"] > 5.0


def test_detect_suspicious_price_changes_empty():
    res = detect_suspicious_price_changes(pd.DataFrame())
    assert res["suspicious_return_rows"] == 0


def test_analyze_gaps_normal_interval():
    df = _sample_bars(24)  # hourly, 1 h intervals
    g = analyze_gaps(df)
    assert g["total_gaps"] == 0
    assert g["largest_gap_seconds"] is None or g["largest_gap_seconds"] == 3600.0
    assert all(gap["classification"] == "normal_interval" for gap in g["gaps"])


def test_analyze_gaps_weekend_provisional():
    # Hourly bars through Friday 16:00, then Monday 16:00 (~72 h gap).
    # The repeated 1 h spacing establishes the base interval so the gap is
    # detectable and provisionally classified as a weekend/scheduled closure.
    times = pd.to_datetime([
        "2024-01-05 12:00", "2024-01-05 13:00", "2024-01-05 14:00",
        "2024-01-05 15:00", "2024-01-05 16:00",  # Friday
        "2024-01-08 16:00",                        # Monday (~72 h gap)
    ])
    close = [1.10] * len(times)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(times),
        }
    )
    g = analyze_gaps(df)
    assert g["total_gaps"] == 1
    gap = g["gaps"][0]
    assert gap["classification"] == "likely_weekend_or_scheduled_closure_provisional"
    assert gap["confidence"] == "medium"


def test_analyze_gaps_potential_outage_review():
    # Hourly bars at 00:00, 01:00, then an 10 h gap to 11:00.
    # Base interval = 1 h; the 10 h gap is larger than base but shorter than
    # a provisional weekend, so it is flagged for review, not an outage.
    times = pd.to_datetime([
        "2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 11:00",
    ])
    close = [1.10] * len(times)
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100] * len(times),
        }
    )
    g = analyze_gaps(df)
    assert g["total_gaps"] == 1
    gap = g["gaps"][0]
    assert gap["classification"] == "potential_data_gap_review"
    assert gap["confidence"] == "low"


def test_analyze_gaps_duplicates_not_counted_as_gaps():
    times = pd.to_datetime([
        "2024-01-01 00:00",
        "2024-01-01 00:00",  # duplicate
        "2024-01-01 01:00",
    ])
    close = [1.10, 1.10, 1.10]
    df = pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [100, 100, 100],
        }
    )
    g = analyze_gaps(df)
    assert g["total_gaps"] == 0  # zero/negative diffs skipped


def test_build_data_quality_report_structure(tmp_path):
    f = tmp_path / "EURUSD1440.csv"
    f.write_text(
        "2010-10-12 00:00\t1.38120\t1.39331\t1.37961\t1.39191\t42040\n"
        "2010-10-13 00:00\t1.39196\t1.40007\t1.39100\t1.39820\t79675\n",
        encoding="utf-8",
    )
    from eti.data import load_bars

    df = load_bars(f)
    report = build_data_quality_report(f, df)
    d = report.to_dict()
    assert d["source"]["rows"] == 2
    assert d["source"]["sha256"]
    assert d["source"]["timezone"] == "unverified_naive"
    assert d["validation"]["status"] == "pass"
    assert d["documentation"]["unknown"]
    assert d["documentation"]["assumed"]
    assert d["generated_at"]


def test_write_report_creates_json_and_txt(tmp_path):
    f = tmp_path / "EURUSD1.csv"
    f.write_text(
        "2026-07-20 13:33\t1.14198\t1.14198\t1.14189\t1.14190\t119\n"
        "2026-07-20 13:34\t1.14191\t1.14199\t1.14184\t1.14185\t162\n",
        encoding="utf-8",
    )
    from eti.data import load_bars

    df = load_bars(f)
    report = build_data_quality_report(f, df)
    json_path = write_report(report, tmp_path)
    assert json_path.exists()
    assert json_path.suffix == ".json"
    txt_path = tmp_path / "EURUSD1_data_quality.txt"
    assert txt_path.exists()
    txt = txt_path.read_text(encoding="utf-8")
    assert "ETI DATA QUALITY REPORT" in txt
    assert "unverified_naive" in txt


def test_format_human_report_contains_key_sections(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text(
        "2024-01-01 00:00\t1.10\t1.1002\t1.0998\t1.10\t100\n"
        "2024-01-01 01:00\t1.1001\t1.1003\t1.10\t1.1001\t100\n",
        encoding="utf-8",
    )
    from eti.data import load_bars

    df = load_bars(f)
    report = build_data_quality_report(f, df)
    text = format_human_report(report)
    for section in [
        "ETI DATA QUALITY REPORT",
        "VALIDATION",
        "GAP ANALYSIS",
        "SOURCE IDENTITY",
        "DOCUMENTATION",
    ]:
        assert section in text
