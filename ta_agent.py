"""
ta_agent.py — BTC/USDT Technical Analysis agent (ReAct + SQL queries on trading.db).
"""

import sqlite3
import os

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from models import TAOutput

DB_PATH = "trading.db"

TA_SYSTEM_PROMPT = """You are a professional cryptocurrency technical analyst specialising in BTC/USDT.

You have access to a tool called `query_db` that lets you run read-only SQL
queries against a SQLite database containing BTC/USDT 5-minute OHLCV candles
with computed technical indicators.

TABLE: btc_ohlcv
COLUMNS: timestamp (TEXT, PK), open, high, low, close, volume,
         rsi_14, macd, macd_signal, macd_hist,
         bb_upper, bb_mid, bb_lower, ema_50

ANALYSIS WORKFLOW
-----------------
Work through these steps in order, making one targeted tool call per step.
Use LIMIT clauses to keep data concise — the most recent 100 candles (~8 hours)
is sufficient for recent action, while filtered queries naturally return fewer rows.

1. RECENT PRICE ACTION
   SELECT * FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 100

2. RSI — MOMENTUM
   SELECT timestamp, close, rsi_14 FROM btc_ohlcv
   WHERE rsi_14 < 35 OR rsi_14 > 65
   ORDER BY timestamp DESC LIMIT 50

3. MACD — TREND DIRECTION
   SELECT timestamp, macd, macd_signal, macd_hist
   FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 100

4. BOLLINGER BANDS — VOLATILITY
   SELECT timestamp, close, bb_lower, bb_upper
   FROM btc_ohlcv
   WHERE close < bb_lower OR close > bb_upper
   ORDER BY timestamp DESC LIMIT 50

5. EMA(50) — MACRO TREND
   SELECT timestamp, close, ema_50
   FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 100

6. VOLUME — CONFIRMATION
   SELECT timestamp, close, volume FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 100

After collecting evidence, output the report in this format:

---
## BTC/USDT Technical Analysis Report

**Current Price:** $X,XXX

### Market Condition
...

### Bullish Signals
...

### Bearish Signals
...

### Volume Analysis
...

### 1-Week Price Target
**Target:** $X,XXX – $X,XXX
**Reasoning:** ...
**Confidence:** Low / Medium / High
---
"""


# Tool factory — cutoff filters out future data for backtesting
def make_query_tool(cutoff: str | None = None):
    @tool
    def query_db(sql: str) -> str:
        """Run a read-only SQL query against trading.db."""

        if not sql.strip().upper().startswith("SELECT"):
            return "Error: only SELECT queries are allowed."

        # In backtest mode, silently replace the table with a filtered subquery
        # so the agent only sees data up to the cutoff timestamp.
        query = sql
        if cutoff:
            query = query.replace(
                "btc_ohlcv",
                f"(SELECT * FROM btc_ohlcv WHERE timestamp <= '{cutoff}')",
            )

        print(f"\n[SQL QUERY]\n{query}\n")

        try:
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return "No rows match the query."

            cols = rows[0].keys()
            lines = ["\t".join(cols)]

            for row in rows:
                vals = []
                for v in row:
                    if isinstance(v, float):
                        vals.append(f"{v:.2f}")
                    else:
                        vals.append(str(v))
                lines.append("\t".join(vals))

            return "\n".join(lines)

        except Exception as e:
            return f"Query error: {e}"

    return query_db


def run_ta_agent(cutoff: str | None = None) -> TAOutput:
    """Run the TA agent against trading.db.

    Args:
        cutoff: Optional timestamp (e.g. '2026-04-01 12:00:00').
                When set, the agent only sees data up to this time (for backtesting).

    Returns:
        TAOutput with full report + structured prediction fields.
    """

    model = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_agent(
        model=model,
        tools=[make_query_tool(cutoff)],
        system_prompt=TA_SYSTEM_PROMPT,
        response_format=TAOutput,
    )

    message = (
        "Analyse the BTC/USDT 5m data and produce a technical analysis report "
        "with a 1-week price target."
    )

    print("=" * 60)
    print("BTC/USDT Technical Analysis Agent")
    if cutoff:
        print(f"  (backtest mode — cutoff: {cutoff})")
    print("=" * 60)

    response = agent.invoke({
        "messages": [("user", message)]
    })

    return response["structured_response"]


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(override=True)
    result = run_ta_agent()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result.report)

    print(f"\nDirection: {result.direction}")
    print(f"Target: ${result.target_low:,.0f} - ${result.target_high:,.0f}")
    print(f"Confidence: {result.confidence:.0%}")