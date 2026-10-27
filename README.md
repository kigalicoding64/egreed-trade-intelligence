# Egreed Trade Intelligence (ETI) — Working Core v0.1

A local historical EUR/USD research workspace. This is an early working core, not a finished production financial system.

## Included now

- Load the supplied ForexSB-style, headerless OHLCV CSV files.
- Basic OHLC/volume/timestamp validation.
- SMA 20/50/200, RSI 14, ATR 14 and rolling volatility.
- Exploratory SMA crossover backtest with next-bar position timing and a configurable simplified transaction cost.
- Streamlit research dashboard with candlestick chart, data quality tab, backtest tab and recent bars.
- FastAPI endpoints for health, summary, bars and backtest.
- Unit tests for the core functions.

## Explicit limitations

- CSV source identity, timezone, candle boundary conventions and volume meaning are not yet verified.
- Resampled candle boundaries use the source's timezone-naive timestamps.
- Gaps are reported, but not automatically classified as market closures versus provider outages.
- The backtest is a simple long/flat research baseline, not a validated strategy. Cost is simplified; spread, slippage, financing, execution, and walk-forward validation are not fully modeled.
- No live market feed, live news integration, authentication, subscription system, broker integration, or real-money execution.
- Never interpret the last historical close as a live price.

## Run locally (Windows PowerShell)

Install Python 3.11 or newer, then from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Run the API in another terminal

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn eti.api:app --reload
```

- `GET http://127.0.0.1:8000/health`
- `GET http://127.0.0.1:8000/api/summary?timeframe=1d`
- `GET http://127.0.0.1:8000/api/bars?timeframe=1d&limit=100`
- `GET http://127.0.0.1:8000/api/backtest?timeframe=1d&fast=20&slow=50&cost_bps=1`
- API docs: `http://127.0.0.1:8000/docs`

## Data files

The provided `EURUSD1.csv` and `EURUSD1440.csv` are included under `data/raw/` for local testing. Keep original files unchanged and track their provenance separately.

## Next engineering milestone

**ETI-001 hardening:** record provider/source metadata, verify timestamp timezone and candle conventions, produce a session-aware gap report, add fixtures for known edge cases, and test against independently calculated reference values before trusting any strategy research.
