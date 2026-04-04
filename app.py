"""
app.py — BTC/USDT Technical Analysis Agent.

On each run:
  1. Refreshes the local SQLite database with the latest 5m candles.
  2. Loads the full dataset into memory.
  3. Invokes a LangGraph ReAct agent that queries the data and produces a
     structured technical analysis report with a 24h BTC price target.
"""

import data
import pandas as pd
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

load_dotenv(override=True)

SYSTEM_PROMPT = """You are a professional cryptocurrency technical analyst specialising in BTC/USDT.

You have access to a tool called `query_market_data` that lets you query a pandas
DataFrame of the last 2 days of BTC/USDT 5-minute OHLCV candles with computed
technical indicators.  The tool accepts pandas query strings (NOT SQL) and returns
all matching rows as a formatted table.

ANALYSIS WORKFLOW
-----------------
Work through these steps in order, making one targeted tool call per step:

1. RECENT PRICE ACTION
   Query the last 24 candles (approx 2 hours) to establish current price level.
   Example: "timestamp >= '<recent_timestamp>'"

2. RSI — MOMENTUM
   Query for RSI extremes to identify overbought/oversold conditions.
   Example: "rsi_14 < 35" or "rsi_14 > 65"

3. MACD — TREND DIRECTION
   Query for MACD vs signal line relationship across all recent candles.
   Example: "macd > macd_signal" (bullish) or "macd < macd_signal" (bearish)

4. BOLLINGER BANDS — VOLATILITY
   Query for price position relative to bands.
   Example: "close < bb_lower" or "close > bb_upper"

5. EMA(50) — MACRO TREND
   Query for price vs EMA50 to determine dominant trend.
   Example: "close > ema_50" or "close < ema_50"

After collecting evidence from all 5 queries, write your report in this exact format:

---
## BTC/USDT Technical Analysis Report

**Current Price:** $X,XXX

### Market Condition
[1-2 sentence summary of current market state]

### Bullish Signals
- [signal 1]
- [signal 2]
...

### Bearish Signals
- [signal 1]
- [signal 2]
...

### 24-Hour Price Target
**Target:** $X,XXX – $X,XXX
**Reasoning:** [2-3 sentences explaining the target based on evidence gathered]
**Confidence:** Low / Medium / High
---

If a query returns no matching rows, note the absence of that signal in your
report — it is still useful information (e.g. "no RSI oversold readings in the
last 2 days").
"""


# ---------------------------------------------------------------------------
# Tool helper — module-level so tests can import it directly
# ---------------------------------------------------------------------------

def _query_df(df: pd.DataFrame, query: str) -> str:
    """Execute a pandas query string against *df* and return all results as a
    formatted string.  Never raises — all errors are returned as strings.
    """
    try:
        result = df.query(query, engine="python")
        if result.empty:
            return "No rows match the query."
        return result.to_string(index=False)
    except Exception as e:
        return f"Query error: {str(e)}"


# ---------------------------------------------------------------------------
# Main — all side-effectful code is gated here
# ---------------------------------------------------------------------------

def main() -> None:
    print("Refreshing market data...")
    data.refresh_db()
    df: pd.DataFrame = data.load_df()
    print(f"Loaded {len(df)} candles.  Latest: {df['timestamp'].iloc[-1]}\n")

    @tool
    def query_market_data(query: str) -> str:
        """Query the BTC/USDT 5-minute OHLCV dataset using a pandas query string.

        AVAILABLE COLUMNS
        -----------------
        timestamp   (str)  — e.g. "2026-04-03 14:35:00"  — use >= / <= for time ranges
        open        (float) — candle open price in USDT
        high        (float) — candle high price in USDT
        low         (float) — candle low price in USDT
        close       (float) — candle close price in USDT
        volume      (float) — traded volume in BTC
        rsi_14      (float) — RSI(14): <30 oversold, >70 overbought
        macd        (float) — MACD line (EMA12 - EMA26)
        macd_signal (float) — MACD signal line (EMA9 of macd)
        macd_hist   (float) — MACD histogram (macd - macd_signal)
        bb_upper    (float) — Bollinger upper band (SMA20 + 2σ)
        bb_mid      (float) — Bollinger middle band (SMA20)
        bb_lower    (float) — Bollinger lower band (SMA20 - 2σ)
        ema_50      (float) — EMA(50) trend line

        QUERY SYNTAX
        ------------
        Use pandas query strings (not SQL).  String values must use single quotes
        inside the query string.  Combine conditions with 'and' / 'or'.

        EXAMPLE QUERIES
        ---------------
        Recent candles:
            "timestamp >= '2026-04-03 00:00:00'"

        Oversold momentum:
            "rsi_14 < 30"

        Overbought momentum:
            "rsi_14 > 70"

        Bullish MACD crossover rows (macd crossed above signal):
            "macd > macd_signal"

        Bearish MACD crossover rows:
            "macd < macd_signal"

        Price below lower Bollinger Band (potential reversal / breakdown):
            "close < bb_lower"

        Price above upper Bollinger Band (extended / potential reversal):
            "close > bb_upper"

        Price in uptrend (above EMA50):
            "close > ema_50"

        Price in downtrend (below EMA50):
            "close < ema_50"

        Combined — oversold AND in downtrend:
            "rsi_14 < 30 and close < ema_50"

        Time-filtered — last 2 hours of data (adjust timestamp accordingly):
            "timestamp >= '2026-04-03 22:00:00' and close > 80000"

        RETURNS
        -------
        All matching rows as a formatted table string, or a plain message if no
        rows match.  Any query syntax error is returned as an error string (never
        raises an exception).
        """
        return _query_df(df, query)

    model = ChatOpenAI(model="gpt-4.1")
    agent = create_react_agent(model, tools=[query_market_data], prompt=SYSTEM_PROMPT)

    message = (
        "Analyse the BTC/USDT 5m data and produce a technical analysis report "
        "with a 24-hour price target."
    )

    print("=" * 60)
    print("BTC/USDT Technical Analysis Agent — streaming output")
    print("=" * 60)

    for chunk in agent.stream({"messages": [("user", message)]}):
        if "agent" in chunk:
            content = chunk["agent"]["messages"][-1].content
            if content:
                print(content, flush=True)
        elif "tools" in chunk:
            for msg in chunk["tools"]["messages"]:
                print(f"\n[Tool call: {msg.name}]\n{msg.content[:500]}\n", flush=True)


if __name__ == "__main__":
    main()
