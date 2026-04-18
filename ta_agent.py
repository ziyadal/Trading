"""
ta_agent.py — BTC/USDT Technical Analysis agent (ReAct + SQL queries on trading.db).
"""

import sqlite3
import os

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from models import TAOutput
from prompts.ta import TA_SYSTEM_PROMPT, TA_USER_PROMPT

DB_PATH = "trading.db"


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


def run_ta_agent(
    cutoff: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    model_name: str | None = None,
) -> TAOutput:
    """Run the TA agent against trading.db.

    Args:
        cutoff: Optional timestamp (e.g. '2026-04-01 12:00:00').
                When set, the agent only sees data up to this time (for backtesting).
        system_prompt: Override the default TA system prompt.
        user_prompt: Override the default user message.
        model_name: Override the default model (gpt-4.1-mini).

    Returns:
        TAOutput with full report + structured prediction fields.
    """
    _prompt = system_prompt or TA_SYSTEM_PROMPT
    _user = user_prompt or TA_USER_PROMPT
    _model = model_name or "gpt-4.1-mini"

    model = ChatOpenAI(
        model=_model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    agent = create_agent(
        model=model,
        tools=[make_query_tool(cutoff)],
        system_prompt=_prompt,
        response_format=TAOutput,
    )

    print("=" * 60)
    print("BTC/USDT Technical Analysis Agent")
    if cutoff:
        print(f"  (backtest mode — cutoff: {cutoff})")
    print("=" * 60)

    response = agent.invoke({
        "messages": [("user", _user)]
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