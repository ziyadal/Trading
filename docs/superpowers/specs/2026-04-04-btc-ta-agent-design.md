# BTC Technical Analysis Agent — Design Spec

**Date:** 2026-04-04  
**Status:** Approved

---

## Overview

Two-file system: `data.py` handles all data ingestion and persistence; `app.py` defines the LangChain agent and runs it with streaming. The agent queries a local SQLite database to produce a structured technical analysis report with a 24h BTC price target.

---

## File 1: `data.py`

### Responsibilities
- Fetch last 2 days of BTC/USDT 5m OHLCV candles from Binance via CCXT (no API key required)
- Calculate 4 technical indicators using `pandas-ta`
- Upsert all rows into a local SQLite database (`trading.db`)
- Expose `load_df()` to return the full dataset as a pandas DataFrame

### Public API
```python
def refresh_db() -> None: ...   # fetch → compute indicators → upsert to SQLite
def load_df() -> pd.DataFrame: ...  # read full btc_ohlcv table from SQLite
```

### SQLite Table: `btc_ohlcv`
| Column | Type | Notes |
|---|---|---|
| timestamp | TEXT (PK) | ISO format string, e.g. `"2026-04-03 14:35:00"` |
| open | REAL | |
| high | REAL | |
| low | REAL | |
| close | REAL | |
| volume | REAL | |
| rsi_14 | REAL | RSI with period 14 |
| macd | REAL | MACD line (12/26) |
| macd_signal | REAL | Signal line (9) |
| macd_hist | REAL | MACD histogram |
| bb_upper | REAL | Bollinger upper band (20, 2σ) |
| bb_mid | REAL | Bollinger middle band |
| bb_lower | REAL | Bollinger lower band |
| ema_50 | REAL | EMA with period 50 |

### Indicators
- **RSI(14):** Momentum oscillator — values below 30 = oversold, above 70 = overbought
- **MACD(12, 26, 9):** Trend-following — `macd > macd_signal` signals bullish momentum
- **Bollinger Bands(20, 2):** Volatility bands — price touching lower band = potential reversal
- **EMA(50):** Trend direction — price above EMA50 = uptrend

### Refresh Strategy
On each run, fetch the last 2 days of candles and `INSERT OR REPLACE` by timestamp PK. Data persists between runs; only stale/new candles are overwritten.

---

## File 2: `app.py`

### Startup Sequence
1. Call `data.refresh_db()` to fetch fresh candles and upsert to SQLite
2. Call `data.load_df()` to load the full dataset into memory as `df`
3. Define tool(s), agent, and invoke with streaming

### Tool: `query_market_data(query: str) -> str`

The tool's docstring acts as the agent's query reference. It documents:
- All available columns with types and example values
- Concrete example queries:
  - `"close > 80000 and rsi_14 < 30"` — oversold at high price
  - `"macd > macd_signal"` — bullish MACD crossover rows
  - `"close < bb_lower"` — price below lower Bollinger Band
  - `"close > ema_50"` — price above EMA50 (uptrend)
  - `"timestamp >= '2026-04-03 00:00:00'"` — recent candles only
  - `"rsi_14 > 70 and close > bb_upper"` — overbought + extended

Returns **all matching rows** as a formatted string (no row cap). Returns a clear message string if no rows match. All exceptions caught and returned as a string — never raises.

### Agent
- **Framework:** `create_react_agent` from `langgraph.prebuilt`
- **Model:** `gpt-4.1` via `ChatOpenAI`
- **Tools:** `[query_market_data]`

### System Prompt
Instructs the agent to:
1. Query recent price action (last 12–24 candles)
2. Check RSI for overbought/oversold conditions
3. Check MACD for trend direction and crossovers
4. Check Bollinger Bands for volatility and price extremes
5. Check EMA(50) for overall trend direction
6. Synthesise signals into a structured report with:
   - Current market condition summary
   - Bullish signals observed
   - Bearish signals observed
   - 24h price target with reasoning
   - Confidence level (Low / Medium / High)

System prompt also includes explicit query syntax guidance:
- Use `df.query()` string syntax (pandas-style, not SQL)
- Column names, operators, and string quoting rules
- Example queries to adapt

### Streaming Invocation
```python
for chunk in agent.stream({"messages": [("user", message)]}):
    print(chunk, flush=True)
```

---

## Dependencies to Add
- `pandas-ta` — technical indicator calculations

## Files to Modify
- `app.py` — full rewrite of current skeleton
- `pyproject.toml` — add `pandas-ta`

## Files to Create
- `data.py` — new data ingestion module
