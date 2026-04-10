"""
ta_agent.py — BTC/USDT Technical Analysis agent (ReAct + SQL queries on trading.db).
"""

import sqlite3
import os

from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

# ✅ Load environment variables
load_dotenv()

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
Work through these steps in order, making one targeted tool call per step:

1. RECENT PRICE ACTION
   SELECT * FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 24

2. RSI — MOMENTUM
   SELECT timestamp, close, rsi_14 FROM btc_ohlcv 
   WHERE rsi_14 < 35 OR rsi_14 > 65 
   ORDER BY timestamp DESC LIMIT 50

3. MACD — TREND DIRECTION
   SELECT timestamp, macd, macd_signal, macd_hist 
   FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 24

4. BOLLINGER BANDS — VOLATILITY
   SELECT timestamp, close, bb_lower, bb_upper 
   FROM btc_ohlcv 
   WHERE close < bb_lower OR close > bb_upper 
   ORDER BY timestamp DESC LIMIT 50

5. EMA(50) — MACRO TREND
   SELECT timestamp, close, ema_50 
   FROM btc_ohlcv ORDER BY timestamp DESC LIMIT 24

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

### 24-Hour Price Target
**Target:** $X,XXX – $X,XXX  
**Reasoning:** ...  
**Confidence:** Low / Medium / High
---
"""


# ✅ TOOL
@tool
def query_db(sql: str) -> str:
    """Run a read-only SQL query against trading.db."""
    
    print(f"\n[SQL QUERY]\n{sql}\n")  # 🔍 debug visibility

    if not sql.strip().upper().startswith("SELECT"):
        return "Error: only SELECT queries are allowed."

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return "No rows match the query."

        cols = rows[0].keys()
        lines = ["\t".join(cols)]

        for row in rows:
            lines.append("\t".join(str(v) for v in row))

        return "\n".join(lines)

    except Exception as e:
        return f"Query error: {e}"


# ✅ MAIN AGENT RUNNER (NO STREAMING)
def run_ta_agent() -> str:
    """Run the TA agent against trading.db."""

    model = ChatOpenAI(
        model="gpt-4.1",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    agent = create_agent(
        model=model,
        tools=[query_db],
        system_prompt=TA_SYSTEM_PROMPT  # ✅ correct param
    )

    message = (
        "Analyse the BTC/USDT 5m data and produce a technical analysis report "
        "with a 24-hour price target."
    )

    print("=" * 60)
    print("BTC/USDT Technical Analysis Agent")
    print("=" * 60)

    # ✅ Simple invoke (no streaming)
    response = agent.invoke({
        "messages": [("user", message)]
    })

    # 🔍 Extract final output safely
    try:
        final_output = response["messages"][-1].content
    except Exception:
        final_output = str(response)

    return final_output


# ✅ ENTRY POINT
if __name__ == "__main__":
    report = run_ta_agent()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")

    print(report)