# Ensure timeframe and path have default values before data loading
timeframe = "Source timeframe"
path = str(MINUTE_PATH)

raw = _load(str(path))
if timeframe != "Source timeframe":
    rule = {"15 minutes": "15min", "1 hour": "1h", "4 hours": "4h"}[timeframe]
    bars = resample_bars(raw, rule)
else:
    bars = raw

ind = _indicators(bars)
stats = summary(bars, source)

# Data quality report (computed on the resampled/selected bars)
quality_report = _quality(bars, str(path))

# ---------------------------------------------------------------------------
# Header metrics
# ---------------------------------------------------------------------------
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Last historical close", f"{bars.close.iloc[-1]:.5f}")
m2.metric("RSI (14)", "—" if pd.isna(ind.rsi14.iloc[-1]) else f"{ind.rsi14.iloc[-1]:.2f}")
m3.metric("SMA (20)", "—" if pd.isna(ind.sma20.iloc[-1]) else f"{ind.sma20.iloc[-1]:.5f}")
m4.metric("ATR (14)", "—" if pd.isna(ind.atr14.iloc[-1]) else f"{ind.atr14.iloc[-1]:.5f}")
m5.metric("Bars", f"{len(bars):,}")