"""
ta_agent.py — BTC/USDT Technical Analysis agent (multi-timeframe, with structured tools).

Tools exposed to the agent:
    get_market_snapshot(timeframe)  — structured snapshot per timeframe (5m / 1h / 4h)
    find_analogues(regime, rsi_bucket, bb_width_bucket, n)  — historical base rates
    query_db(sql)                   — raw SQL escape hatch

All tools respect `cutoff` when supplied (for backtesting) so the agent never
sees bars after the cutoff. find_analogues additionally requires that a row's
forward-return window ended at or before the cutoff, preventing lookahead bias.
"""

import json
import sqlite3
from datetime import datetime, timedelta

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.callbacks import get_openai_callback

from data import TABLE_5M, TABLE_1H, TABLE_4H
from llm import make_llm
from models import TAOutput
from prompts.ta import TA_SYSTEM_PROMPT, TA_USER_PROMPT
from usage import AgentUsage, count_tool_calls, usage_from_callback

DB_PATH = "trading.db"

VALID_TIMEFRAMES = {"5m": TABLE_5M, "1h": TABLE_1H, "4h": TABLE_4H}

# How many bars make up 7 days on each timeframe (for lookahead-bias filter).
FWD_BARS = {"5m": 7 * 24 * 12, "1h": 7 * 24, "4h": 7 * 6}

# How many minutes one bar covers (for computing cutoff offsets).
BAR_MINUTES = {"5m": 5, "1h": 60, "4h": 240}


# ---------------------------------------------------------------------------
# Snapshot tool
# ---------------------------------------------------------------------------

def _bucket_rsi(rsi: float | None) -> str:
    if rsi is None:
        return "unknown"
    if rsi < 30:
        return "oversold"
    if rsi < 45:
        return "low"
    if rsi < 55:
        return "mid"
    if rsi < 70:
        return "high"
    return "overbought"


def _bucket_bb_width(rank: float | None) -> str:
    if rank is None:
        return "unknown"
    if rank < 0.33:
        return "low"
    if rank < 0.66:
        return "mid"
    return "high"


def _snapshot(conn: sqlite3.Connection, timeframe: str, cutoff: str | None) -> dict:
    """Build a structured snapshot for *timeframe*. Respects cutoff if set."""
    table = VALID_TIMEFRAMES[timeframe]

    base_where = f"WHERE timestamp <= '{cutoff}'" if cutoff else ""
    latest = conn.execute(
        f"SELECT * FROM {table} {base_where} ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    if latest is None:
        return {"error": f"No data in {table}"}

    cols = [c["name"] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    row = dict(zip(cols, latest))

    # Most recent swing high / swing low before the latest bar.
    last_swing_high = conn.execute(
        f"SELECT timestamp, high FROM {table} "
        f"WHERE swing_high = 1 AND timestamp < '{row['timestamp']}' "
        f"{('AND timestamp <= ' + repr(cutoff)) if cutoff else ''} "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()
    last_swing_low = conn.execute(
        f"SELECT timestamp, low FROM {table} "
        f"WHERE swing_low = 1 AND timestamp < '{row['timestamp']}' "
        f"{('AND timestamp <= ' + repr(cutoff)) if cutoff else ''} "
        "ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    def _bars_ago(ts: str | None) -> int | None:
        if ts is None:
            return None
        delta = (
            datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            - datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        ).total_seconds() / 60
        return int(delta / BAR_MINUTES[timeframe])

    def _round(x, n=2):
        return None if x is None else round(x, n)

    return {
        "timeframe": timeframe,
        "timestamp": row["timestamp"],
        "current_price": _round(row["close"]),
        "open": _round(row["open"]),
        "high": _round(row["high"]),
        "low": _round(row["low"]),
        "rsi_14": _round(row["rsi_14"]),
        "rsi_bucket": _bucket_rsi(row["rsi_14"]),
        "rsi_slope_5": _round(row["rsi_slope_5"]),
        "macd": {
            "line": _round(row["macd"]),
            "signal": _round(row["macd_signal"]),
            "hist": _round(row["macd_hist"]),
            "hist_positive": (row["macd_hist"] or 0) > 0,
        },
        "bb": {
            "upper": _round(row["bb_upper"]),
            "mid": _round(row["bb_mid"]),
            "lower": _round(row["bb_lower"]),
            "width_pct": _round(row["bb_width"]),
            "width_rank_30d": _round(row["bb_width_rank"], 3),
            "width_bucket": _bucket_bb_width(row["bb_width_rank"]),
            "position": (
                "above_upper" if row["close"] and row["bb_upper"] and row["close"] > row["bb_upper"]
                else "below_lower" if row["close"] and row["bb_lower"] and row["close"] < row["bb_lower"]
                else "upper_half" if row["close"] and row["bb_mid"] and row["close"] > row["bb_mid"]
                else "lower_half"
            ),
        },
        "ema_50": _round(row["ema_50"]),
        "dist_from_ema50_pct": _round(row["dist_from_ema50_pct"]),
        "atr_14": _round(row["atr_14"]),
        "atr_pct": _round(row["atr_pct"]),
        "volume": _round(row["volume"]),
        "vol_ratio": _round(row["vol_ratio"]),
        "regime": row["regime"],
        "last_swing_high": (
            {"price": _round(last_swing_high[1]), "bars_ago": _bars_ago(last_swing_high[0])}
            if last_swing_high else None
        ),
        "last_swing_low": (
            {"price": _round(last_swing_low[1]), "bars_ago": _bars_ago(last_swing_low[0])}
            if last_swing_low else None
        ),
    }


# ---------------------------------------------------------------------------
# Analogue-finder tool
# ---------------------------------------------------------------------------

def _find_analogues(
    conn: sqlite3.Connection,
    regime: str,
    rsi_bucket: str,
    bb_width_bucket: str,
    n: int,
    cutoff: str | None,
) -> dict:
    """Find historical bars on the 4h timeframe matching the given conditions.

    Returns aggregate forward-return stats plus up to *n* individual matches.
    In backtest mode, the forward window must have ended at or before cutoff.
    """
    table = TABLE_4H

    # Condition filters — map buckets to RSI / bb_width_rank ranges.
    rsi_ranges = {
        "oversold": (0, 30),
        "low": (30, 45),
        "mid": (45, 55),
        "high": (55, 70),
        "overbought": (70, 100),
    }
    bb_ranges = {"low": (0.0, 0.33), "mid": (0.33, 0.66), "high": (0.66, 1.01)}

    if rsi_bucket not in rsi_ranges:
        return {"error": f"Unknown rsi_bucket '{rsi_bucket}'. Choose from: {list(rsi_ranges)}"}
    if bb_width_bucket not in bb_ranges:
        return {"error": f"Unknown bb_width_bucket '{bb_width_bucket}'. Choose from: {list(bb_ranges)}"}

    rsi_lo, rsi_hi = rsi_ranges[rsi_bucket]
    bb_lo, bb_hi = bb_ranges[bb_width_bucket]

    where = [
        f"regime = '{regime}'",
        f"rsi_14 >= {rsi_lo} AND rsi_14 < {rsi_hi}",
        f"bb_width_rank >= {bb_lo} AND bb_width_rank < {bb_hi}",
        "fwd_7d_return_pct IS NOT NULL",
    ]
    if cutoff:
        # Forward window must have closed by the cutoff — no lookahead.
        horizon_end = (
            datetime.strptime(cutoff, "%Y-%m-%d %H:%M:%S") - timedelta(days=7)
        ).strftime("%Y-%m-%d %H:%M:%S")
        where.append(f"timestamp <= '{horizon_end}'")

    where_sql = " AND ".join(where)

    agg = conn.execute(
        f"SELECT COUNT(*), AVG(fwd_7d_return_pct), AVG(fwd_7d_high_pct), "
        f"AVG(fwd_7d_low_pct), "
        f"SUM(CASE WHEN fwd_7d_return_pct > 0 THEN 1 ELSE 0 END) "
        f"FROM {table} WHERE {where_sql}"
    ).fetchone()

    count, avg_ret, avg_high, avg_low, positive = agg
    if not count:
        return {
            "conditions": {
                "regime": regime,
                "rsi_bucket": rsi_bucket,
                "bb_width_bucket": bb_width_bucket,
            },
            "matches": 0,
            "message": "No historical analogues found for these conditions.",
        }

    samples = conn.execute(
        f"SELECT timestamp, close, rsi_14, bb_width_rank, "
        f"fwd_7d_return_pct, fwd_7d_high_pct, fwd_7d_low_pct "
        f"FROM {table} WHERE {where_sql} "
        f"ORDER BY timestamp DESC LIMIT {int(n)}"
    ).fetchall()

    return {
        "conditions": {
            "regime": regime,
            "rsi_bucket": rsi_bucket,
            "bb_width_bucket": bb_width_bucket,
        },
        "matches": int(count),
        "aggregate": {
            "pct_positive_7d": round(100 * positive / count, 1),
            "avg_fwd_7d_return_pct": round(avg_ret, 2),
            "avg_fwd_7d_high_pct": round(avg_high, 2),
            "avg_fwd_7d_low_pct": round(avg_low, 2),
        },
        "recent_examples": [
            {
                "timestamp": r[0],
                "close": round(r[1], 2),
                "rsi_14": round(r[2], 2),
                "bb_width_rank": round(r[3], 3),
                "fwd_7d_return_pct": round(r[4], 2),
                "fwd_7d_high_pct": round(r[5], 2),
                "fwd_7d_low_pct": round(r[6], 2),
            }
            for r in samples
        ],
    }


# ---------------------------------------------------------------------------
# Tool factories — each returns LangChain @tool functions bound to `cutoff`
# ---------------------------------------------------------------------------

def _build_tools(cutoff: str | None):
    @tool
    def get_market_snapshot(timeframe: str) -> str:
        """Return a structured snapshot of current BTC/USDT conditions on one timeframe.

        Args:
            timeframe: One of "5m", "1h", "4h".

        Returns a JSON blob with: price, RSI (+ bucket and slope), MACD state,
        Bollinger Bands (width, rank, bucket, position), EMA-50 and distance,
        ATR, volume ratio, regime tag, and the most recent swing high/low.
        """
        if timeframe not in VALID_TIMEFRAMES:
            return json.dumps({"error": f"Invalid timeframe '{timeframe}'. Choose from: {list(VALID_TIMEFRAMES)}"})

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            snap = _snapshot(conn, timeframe, cutoff)
        finally:
            conn.close()

        print(f"\n[SNAPSHOT {timeframe}] regime={snap.get('regime')} rsi={snap.get('rsi_14')} price={snap.get('current_price')}")
        return json.dumps(snap, indent=2)

    @tool
    def find_analogues(
        regime: str,
        rsi_bucket: str,
        bb_width_bucket: str,
        n: int = 5,
    ) -> str:
        """Find historical 4h bars matching the given conditions and summarise their 7-day forward outcomes.

        Args:
            regime: One of "TRENDING_UP", "TRENDING_DOWN", "RANGE", "SQUEEZE".
            rsi_bucket: "oversold" | "low" | "mid" | "high" | "overbought".
            bb_width_bucket: "low" (squeezed) | "mid" | "high" (expanded).
            n: Number of recent examples to return (default 5).

        Returns match count, aggregate forward-return stats (% positive,
        avg return, avg high, avg low over the next 7 days), plus the most
        recent individual examples.
        """
        conn = sqlite3.connect(DB_PATH)
        try:
            result = _find_analogues(conn, regime, rsi_bucket, bb_width_bucket, n, cutoff)
        finally:
            conn.close()

        print(f"\n[ANALOGUES] regime={regime} rsi={rsi_bucket} bbw={bb_width_bucket} -> {result.get('matches', 0)} matches")
        return json.dumps(result, indent=2)

    @tool
    def query_db(sql: str) -> str:
        """Run a read-only SQL SELECT against trading.db.

        Tables: btc_ohlcv (5m), btc_ohlcv_1h, btc_ohlcv_4h. All share the same columns.
        Use this only for questions the snapshot and analogue tools can't answer —
        e.g. counting specific multi-bar patterns, inspecting a specific date range.
        """
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: only SELECT queries are allowed."

        query = sql
        if cutoff:
            # Rewrite every known table reference to filter by cutoff.
            for table in VALID_TIMEFRAMES.values():
                query = query.replace(
                    table,
                    f"(SELECT * FROM {table} WHERE timestamp <= '{cutoff}')",
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
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                lines.append("\t".join(vals))
            return "\n".join(lines)
        except Exception as e:
            return f"Query error: {e}"

    return [get_market_snapshot, find_analogues, query_db]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_ta_agent(
    cutoff: str | None = None,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    model_name: str | None = None,
) -> tuple[TAOutput, AgentUsage]:
    """Run the TA agent against trading.db.

    Args:
        cutoff: Optional timestamp (e.g. '2026-04-01 12:00:00').
                When set, the agent only sees data up to this time (for backtesting).
        system_prompt: Override the default TA system prompt.
        user_prompt: Override the default user message.
        model_name: Override the default model (gpt-4.1-mini).

    Returns:
        (TAOutput, AgentUsage) — structured prediction plus token/cost stats.
    """
    _prompt = system_prompt or TA_SYSTEM_PROMPT
    _user = user_prompt or TA_USER_PROMPT
    _model = model_name or "openai/gpt-4.1-mini"

    model = make_llm(_model)

    agent = create_agent(
        model=model,
        tools=_build_tools(cutoff),
        system_prompt=_prompt,
        response_format=TAOutput,
    )

    print("=" * 60)
    print("BTC/USDT Technical Analysis Agent")
    if cutoff:
        print(f"  (backtest mode — cutoff: {cutoff})")
    print("=" * 60)

    with get_openai_callback() as cb:
        response = agent.invoke({"messages": [("user", _user)]})

    usage = usage_from_callback(
        agent="ta",
        model=_model,
        cb=cb,
        tool_calls=count_tool_calls(response["messages"]),
    )

    return response["structured_response"], usage


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(override=True)
    result, usage = run_ta_agent()

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60 + "\n")
    print(result.report)

    print(f"\nDirection: {result.direction}")
    print(f"Regime:    {result.regime}")
    print(f"Target:    ${result.target_low:,.0f} - ${result.target_high:,.0f}")
    print(f"Invalid:   ${result.invalidation_level:,.0f}")
    print(f"Confidence: {result.confidence:.0%}")
    print(f"\nDevil's advocate: {result.devils_advocate}")
    print(
        f"\nUsage: tool_calls={usage.tool_calls} "
        f"in={usage.input_tokens} out={usage.output_tokens} "
        f"cost=${usage.cost:.4f}"
    )
