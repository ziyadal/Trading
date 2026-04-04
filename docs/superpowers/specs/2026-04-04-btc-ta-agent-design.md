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

## Unit Testing (`tests/test_data.py`, `tests/test_tools.py`)

### `tests/test_data.py` — data module
- **`test_refresh_db_creates_table`** — after `refresh_db()`, SQLite `btc_ohlcv` table exists and has rows
- **`test_load_df_returns_dataframe`** — `load_df()` returns a pandas DataFrame with all expected columns
- **`test_load_df_has_correct_columns`** — asserts all 14 columns present (timestamp, ohlcv + 8 indicator cols)
- **`test_indicators_not_all_null`** — RSI, MACD, BB, EMA columns are not entirely NaN (enough rows to compute)
- **`test_timestamp_format`** — timestamp column values match `YYYY-MM-DD HH:MM:SS` format
- **`test_upsert_does_not_duplicate`** — calling `refresh_db()` twice leaves no duplicate timestamps

### `tests/test_tools.py` — agent tool
- **`test_valid_query_returns_string`** — a valid query like `"close > 0"` returns a non-empty string
- **`test_no_match_returns_message`** — a query that matches nothing returns the "No rows match" string
- **`test_invalid_query_returns_error_string`** — a malformed query (e.g. `"invalid =="`) returns an error string, does not raise
- **`test_returns_all_rows`** — result string row count matches the DataFrame query result count

Tests use a real (temporary) SQLite DB fixture — no mocking of the data layer. CCXT calls are mocked with synthetic OHLCV data so tests run offline.

---

## Dependencies to Add
- `pandas-ta` — technical indicator calculations
- `pytest` — test runner

## Files to Modify
- `app.py` — full rewrite of current skeleton
- `pyproject.toml` — add `pandas-ta`, `pytest`

## Files to Create
- `data.py` — new data ingestion module
- `tests/test_data.py` — unit tests for data module
- `tests/test_tools.py` — unit tests for agent tool
