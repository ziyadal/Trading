"""
harness.py — Evaluation harness for the BTC trading pipeline.

Runs agents across historical cutoffs, scores predictions against actual
price movement, and logs results to eval.db. Two modes:
  - "ta":   TA agent only (fast, cheap)
  - "full": TA + debate + PM with fake news (no news agent)

Usage (notebook or script):

    from harness import run_harness, AgentConfig

    # TA only, 5 weekly tests starting Feb 1
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="ta",
        run_label="ta_v1",
    )

    # Test a new TA prompt
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="ta",
        ta=AgentConfig(system_prompt=MY_NEW_PROMPT),
        run_label="ta_v2",
    )

    # Full pipeline (TA + debate + PM, fake news)
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="full",
        run_label="full_v1",
    )

    # Custom date range with 4 evenly-spaced tests
    results = run_harness(
        start_date="2026-01-15 12:00:00",
        end_date="2026-04-01 12:00:00",
        num_weeks=4,
        mode="ta",
        run_label="spread_test",
    )
"""

import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

load_dotenv(override=True)

from graph import CHECKPOINT_DB, build_graph, make_saver, make_thread_id
from ta_agent import run_ta_agent
from usage import AgentUsage

EVAL_DB = "eval.db"

# Annualization factor for Sharpe — weekly cutoffs => sqrt(52).
WEEKS_PER_YEAR = 52


# ---------------------------------------------------------------------------
# Config / result types
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration overrides for an agent. None = use agent's default."""

    system_prompt: str | None = None
    user_prompt: str | None = None
    model: str | None = None


@dataclass
class TestResult:
    """Result from scoring a single agent prediction at one cutoff."""

    week: int
    cutoff: str
    agent: str
    direction: str
    target: float | None
    confidence: float
    price_at_cutoff: float | None = None
    actual_price: float | None = None
    direction_correct: bool | None = None
    target_error: float | None = None
    # Reasoning / trade levels captured from the agent's structured output
    report: str | None = None
    entry: float | None = None
    stop_loss: float | None = None
    key_point: str | None = None       # bull/bear key_argument or PM key_reason
    winning_side: str | None = None    # PM only
    # Realized 1-week return if you'd taken this trade (long for BULLISH,
    # short for BEARISH, flat 0 for NEUTRAL). None when can't be scored.
    realized_return: float | None = None
    # Per-agent LLM usage for this cutoff
    model: str | None = None
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_cutoffs(
    start_date: str, num_weeks: int, end_date: str | None = None
) -> list[str]:
    """Generate cutoff timestamps.

    If end_date is provided, cutoffs are evenly spaced between start and end.
    Otherwise, cutoffs are 1 week apart starting from start_date.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        if num_weeks <= 1:
            return [start_date]
        step = (end - start) / (num_weeks - 1)
        return [
            (start + step * i).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(num_weeks)
        ]

    return [
        (start + timedelta(weeks=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(num_weeks)
    ]


def _get_price(timestamp: str, direction: str = "before") -> float | None:
    """Get the closest BTC close price at/before or at/after a timestamp."""
    conn = sqlite3.connect("trading.db")
    try:
        if direction == "before":
            row = conn.execute(
                "SELECT close FROM btc_ohlcv WHERE timestamp <= ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (timestamp,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT close FROM btc_ohlcv WHERE timestamp >= ? "
                "ORDER BY timestamp ASC LIMIT 1",
                (timestamp,),
            ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _score_direction(
    predicted: str, price_before: float, price_after: float
) -> bool:
    """Check if the predicted direction matches actual price movement."""
    went_up = price_after > price_before
    if predicted in ("BULLISH", "BUY", "UP"):
        return went_up
    elif predicted in ("BEARISH", "SELL", "DOWN"):
        return not went_up
    else:  # NEUTRAL
        pct_change = abs(price_after - price_before) / price_before
        return pct_change < 0.02


def _realized_return(
    direction: str, price_before: float, price_after: float
) -> float:
    """Realized 1-week return if you'd traded this signal at full size.

    BULLISH = long: profit when price rises.
    BEARISH = short: profit when price falls.
    NEUTRAL = flat: 0% (capital sits in cash for the week).
    """
    weekly_change = (price_after - price_before) / price_before
    if direction in ("BULLISH", "BUY", "UP"):
        return weekly_change
    if direction in ("BEARISH", "SELL", "DOWN"):
        return -weekly_change
    return 0.0


# ---------------------------------------------------------------------------
# Risk/return metric computation
# ---------------------------------------------------------------------------

def _max_drawdown(returns: list[float]) -> float:
    """Largest peak-to-trough loss on the equity curve formed by *returns*.

    Equity starts at 1.0; we compound (1+r) and track running peak. Returned
    as a positive fraction (e.g. 0.25 means a 25% drawdown).
    """
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        equity *= (1 + r)
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_run_metrics(results: list[TestResult]) -> list[dict]:
    """Compute per-agent risk/return metrics for one harness run.

    Builds one row per real agent (ta/bull/bear/pm) plus a synthetic
    'buy_hold' row that simulates always being long over the same cutoffs —
    this is the baseline a PM-gated strategy must beat on a risk-adjusted basis.

    All returns assumed weekly; Sharpe is annualized by sqrt(52).
    """
    agents = sorted(set(r.agent for r in results))

    # Buy-and-hold uses any one agent's price observations (same prices per
    # cutoff across agents). Pick the first agent that has scoreable data.
    bh_returns: list[float] = []
    for a in agents:
        priced = [
            r for r in results
            if r.agent == a
            and r.price_at_cutoff is not None
            and r.actual_price is not None
        ]
        if priced:
            bh_returns = [
                (r.actual_price - r.price_at_cutoff) / r.price_at_cutoff  # type: ignore[operator]
                for r in priced
            ]
            break

    metric_rows: list[dict] = []
    for agent in agents + (["buy_hold"] if bh_returns else []):
        returns: list[float]
        committed_returns: list[float]
        if agent == "buy_hold":
            returns = bh_returns
            committed_returns = bh_returns  # always in-market
            wins = sum(1 for r in committed_returns if r > 0)
        else:
            scored = [
                r for r in results
                if r.agent == agent and r.realized_return is not None
            ]
            if not scored:
                continue
            returns = [r.realized_return for r in scored if r.realized_return is not None]
            committed_returns = [
                r.realized_return for r in scored
                if r.realized_return is not None
                and r.direction not in ("NEUTRAL", "HOLD", None, "")
            ]
            wins = sum(
                1 for r in scored
                if r.direction not in ("NEUTRAL", "HOLD", None, "")
                and r.direction_correct
            )

        if not returns:
            continue

        n = len(returns)
        mean_r = sum(returns) / n
        if n > 1:
            variance = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
            std_r = math.sqrt(variance)
        else:
            std_r = 0.0

        cumulative = 1.0
        for r in returns:
            cumulative *= (1 + r)
        cumulative -= 1

        sharpe = (mean_r / std_r) * math.sqrt(WEEKS_PER_YEAR) if std_r > 0 else None
        max_dd = _max_drawdown(returns)
        committed_n = len(committed_returns)
        hit_rate = (wins / committed_n) if committed_n else None

        metric_rows.append({
            "agent": agent,
            "total_weeks": n,
            "committed_weeks": committed_n,
            "committed_hit_rate": hit_rate,
            "cumulative_return": cumulative,
            "mean_return": mean_r,
            "std_return": std_r,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
        })

    return metric_rows


# ---------------------------------------------------------------------------
# Eval logging
# ---------------------------------------------------------------------------

_HARNESS_EXTRA_COLS = (
    ("report",          "TEXT"),
    ("entry",           "REAL"),
    ("stop_loss",       "REAL"),
    ("key_point",       "TEXT"),
    ("winning_side",    "TEXT"),
    ("model",           "TEXT"),
    ("tool_calls",      "INTEGER DEFAULT 0"),
    ("input_tokens",    "INTEGER DEFAULT 0"),
    ("output_tokens",   "INTEGER DEFAULT 0"),
    ("cost",            "REAL    DEFAULT 0"),
    ("realized_return", "REAL"),
)


def _ensure_columns(conn: sqlite3.Connection, table: str, cols: tuple) -> None:
    """Add any missing columns to an existing table (simple SQLite migration)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in cols:
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _init_harness_tables(db_path: str = EVAL_DB) -> None:
    """Create harness_results + run_totals + run_metrics (and migrate older schemas)."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS harness_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label       TEXT NOT NULL,
            run_timestamp   TEXT NOT NULL,
            week            INTEGER NOT NULL,
            cutoff          TEXT NOT NULL,
            agent           TEXT NOT NULL,
            direction       TEXT,
            target          REAL,
            confidence      REAL,
            price_at_cutoff REAL,
            actual_price    REAL,
            direction_correct INTEGER,
            target_error    REAL,
            report          TEXT,
            entry           REAL,
            stop_loss       REAL,
            key_point       TEXT,
            winning_side    TEXT,
            model           TEXT,
            tool_calls      INTEGER DEFAULT 0,
            input_tokens    INTEGER DEFAULT 0,
            output_tokens   INTEGER DEFAULT 0,
            cost            REAL    DEFAULT 0,
            realized_return REAL
        )
    """)
    _ensure_columns(conn, "harness_results", _HARNESS_EXTRA_COLS)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_totals (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label           TEXT NOT NULL,
            run_timestamp       TEXT NOT NULL,
            mode                TEXT NOT NULL,
            num_cutoffs         INTEGER NOT NULL,
            total_tool_calls    INTEGER NOT NULL,
            total_input_tokens  INTEGER NOT NULL,
            total_output_tokens INTEGER NOT NULL,
            total_cost          REAL    NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS run_metrics (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            run_label           TEXT NOT NULL,
            run_timestamp       TEXT NOT NULL,
            agent               TEXT NOT NULL,
            total_weeks         INTEGER NOT NULL,
            committed_weeks     INTEGER NOT NULL,
            committed_hit_rate  REAL,
            cumulative_return   REAL NOT NULL,
            mean_return         REAL,
            std_return          REAL,
            sharpe_ratio        REAL,
            max_drawdown        REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_harness_results(
    results: list[TestResult],
    run_label: str,
    mode: str,
    num_cutoffs: int,
    db_path: str = EVAL_DB,
) -> None:
    """Save per-agent results, a run_totals row, and per-agent run_metrics."""
    _init_harness_tables(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    for r in results:
        conn.execute(
            """INSERT INTO harness_results
               (run_label, run_timestamp, week, cutoff, agent,
                direction, target, confidence, price_at_cutoff,
                actual_price, direction_correct, target_error,
                report, entry, stop_loss, key_point, winning_side,
                model, tool_calls, input_tokens, output_tokens, cost,
                realized_return)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?,
                       ?, ?, ?, ?, ?, ?)""",
            (
                run_label, timestamp, r.week, r.cutoff, r.agent,
                r.direction, r.target, r.confidence, r.price_at_cutoff,
                r.actual_price,
                None if r.direction_correct is None else int(r.direction_correct),
                r.target_error,
                r.report, r.entry, r.stop_loss, r.key_point, r.winning_side,
                r.model, r.tool_calls, r.input_tokens, r.output_tokens, r.cost,
                r.realized_return,
            ),
        )

    conn.execute(
        """INSERT INTO run_totals
           (run_label, run_timestamp, mode, num_cutoffs,
            total_tool_calls, total_input_tokens, total_output_tokens, total_cost)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            run_label, timestamp, mode, num_cutoffs,
            sum(r.tool_calls for r in results),
            sum(r.input_tokens for r in results),
            sum(r.output_tokens for r in results),
            sum(r.cost for r in results),
        ),
    )

    for m in _compute_run_metrics(results):
        conn.execute(
            """INSERT INTO run_metrics
               (run_label, run_timestamp, agent, total_weeks, committed_weeks,
                committed_hit_rate, cumulative_return, mean_return, std_return,
                sharpe_ratio, max_drawdown)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_label, timestamp, m["agent"], m["total_weeks"],
                m["committed_weeks"], m["committed_hit_rate"],
                m["cumulative_return"], m["mean_return"], m["std_return"],
                m["sharpe_ratio"], m["max_drawdown"],
            ),
        )

    conn.commit()
    conn.close()
    print(f"\nSaved {len(results)} harness results to eval.db (label: {run_label})")


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(
    start_date: str,
    num_weeks: int = 4,
    end_date: str | None = None,
    mode: str = "ta",
    ta: AgentConfig | None = None,
    bull: AgentConfig | None = None,
    bear: AgentConfig | None = None,
    pm: AgentConfig | None = None,
    num_rebuttals: int = 2,
    run_label: str = "eval",
) -> list[TestResult]:
    """Run the evaluation harness across weekly cutoffs.

    Args:
        start_date: First cutoff timestamp (e.g. "2026-02-01 12:00:00").
        num_weeks:  Number of tests to run.
        end_date:   Optional end date — if provided, cutoffs are evenly spaced
                    between start_date and end_date instead of weekly.
        mode:       "ta" for TA agent only, "full" for TA + debate + PM.
        ta:         Config overrides for the TA agent.
        bull:       Config overrides for the bull debate agent.
        bear:       Config overrides for the bear debate agent.
        pm:         Config overrides for the PM agent.
        num_rebuttals: Debate rebuttal rounds (full mode only).
        run_label:  Label for this evaluation run.

    Returns:
        List of TestResult objects with scores for every agent at each cutoff.
    """
    ta = ta or AgentConfig()
    bull = bull or AgentConfig()
    bear = bear or AgentConfig()
    pm = pm or AgentConfig()

    cutoffs = generate_cutoffs(start_date, num_weeks, end_date)
    results: list[TestResult] = []

    print(f"\n{'=' * 60}")
    print("EVALUATION HARNESS")
    print(f"  Mode:     {mode}")
    print(f"  Tests:    {num_weeks}")
    print(f"  Range:    {cutoffs[0]}  ->  {cutoffs[-1]}")
    print(f"  Label:    {run_label}")
    print(f"{'=' * 60}")

    # Full-pipeline mode runs through the LangGraph (with_hitl=False so it
    # ends after PM without blocking). TA-only mode keeps the direct call —
    # no graph overhead for fast prompt iteration.
    graph = None
    saver_conn = None
    if mode == "full":
        saver_conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
        saver = make_saver(saver_conn)
        graph = build_graph(saver, with_hitl=False)

    try:
        for i, cutoff in enumerate(cutoffs):
            week = i + 1

            print(f"\n{'-' * 60}")
            print(f"TEST {week}/{num_weeks} | Cutoff: {cutoff}")
            print(f"{'-' * 60}")

            try:
                _run_one_cutoff(
                    week=week, cutoff=cutoff, mode=mode, results=results,
                    ta=ta, bull=bull, bear=bear, pm=pm,
                    num_rebuttals=num_rebuttals,
                    graph=graph, run_label=run_label,
                )
            except Exception as e:
                print(f"  [ERROR] cutoff {cutoff} failed: {type(e).__name__}: {e}")
                print("  Skipping this cutoff and continuing.")
    finally:
        if saver_conn is not None:
            saver_conn.close()

    # --- Save to database ---
    log_harness_results(results, run_label, mode=mode, num_cutoffs=len(cutoffs))

    # --- Summary ---
    print_summary(results, run_label)
    return results


def _run_one_cutoff(
    week: int,
    cutoff: str,
    mode: str,
    results: list[TestResult],
    ta: AgentConfig,
    bull: AgentConfig,
    bear: AgentConfig,
    pm: AgentConfig,
    num_rebuttals: int,
    graph=None,
    run_label: str = "",
) -> None:
    """Run one cutoff's worth of work, appending TestResults to `results`.

    For mode="ta" runs the TA agent directly. For mode="full", invokes the
    LangGraph pipeline (one fresh thread_id per cutoff so checkpoints don't
    collide and any one cutoff can be replayed/rolled-back independently).
    """
    future_ts = (
        datetime.strptime(cutoff, "%Y-%m-%d %H:%M:%S") + timedelta(weeks=1)
    ).strftime("%Y-%m-%d %H:%M:%S")
    price_at_cutoff = _get_price(cutoff, "before")
    actual_price = _get_price(future_ts, "after")
    can_score = price_at_cutoff is not None and actual_price is not None

    if mode == "ta":
        ta_output, ta_usage = run_ta_agent(
            cutoff=cutoff,
            system_prompt=ta.system_prompt,
            user_prompt=ta.user_prompt,
            model_name=ta.model,
        )
        _record_ta(results, week, cutoff, ta_output, ta_usage,
                   price_at_cutoff, actual_price, can_score)
        return

    # --- Full pipeline via LangGraph ---
    assert graph is not None, "graph must be provided for mode='full'"

    thread_id = make_thread_id("harness")
    config = {
        "configurable": {
            "thread_id": thread_id,
            "cutoff": cutoff,
            "fake_news": True,
            "ta_model": ta.model,
            "ta_prompt": ta.system_prompt,
            "ta_user_prompt": ta.user_prompt,
            "bull_model": bull.model,
            "bull_prompt": bull.system_prompt,
            "bear_model": bear.model,
            "bear_prompt": bear.system_prompt,
            "pm_model": pm.model,
            "pm_prompt": pm.system_prompt,
            "num_rebuttals": num_rebuttals,
        },
        "metadata": {
            "run_type": "harness",
            "run_label": run_label,
            "cutoff": cutoff,
            "week": week,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    final = graph.invoke({}, config=config)
    print(f"  [thread] {thread_id}")

    ta_output = final["ta_output"]
    ta_usage = final["ta_usage"]
    bull_output = final["bull_output"]
    bear_output = final["bear_output"]
    pm_output = final["pm_output"]
    bull_usage = final["bull_usage"]
    bear_usage = final["bear_usage"]
    pm_usage = final["pm_usage"]

    _record_ta(results, week, cutoff, ta_output, ta_usage,
               price_at_cutoff, actual_price, can_score)

    for agent_name, stance, side_output, side_usage in (
        ("bull", "BULLISH", bull_output, bull_usage),
        ("bear", "BEARISH", bear_output, bear_usage),
    ):
        side_correct = None
        side_error = None
        side_return = None
        if can_score:
            side_correct = _score_direction(
                stance, price_at_cutoff, actual_price
            )
            side_error = abs(side_output.target - actual_price)
            side_return = _realized_return(stance, price_at_cutoff, actual_price)

        results.append(
            TestResult(
                week=week,
                cutoff=cutoff,
                agent=agent_name,
                direction=stance,
                target=side_output.target,
                confidence=side_output.confidence,
                price_at_cutoff=price_at_cutoff,
                actual_price=actual_price,
                direction_correct=side_correct,
                target_error=side_error,
                realized_return=side_return,
                report=side_output.report,
                entry=side_output.entry,
                stop_loss=side_output.stop_loss,
                key_point=side_output.key_argument,
                model=side_usage.model,
                tool_calls=side_usage.tool_calls,
                input_tokens=side_usage.input_tokens,
                output_tokens=side_usage.output_tokens,
                cost=side_usage.cost,
            )
        )
        _print_agent_line(
            agent_name.upper(), stance, side_output.target,
            side_output.confidence, can_score, side_correct,
            actual_price, side_error,
        )

    pm_correct = None
    pm_error = None
    pm_return = None
    if can_score:
        pm_correct = _score_direction(
            pm_output.decision, price_at_cutoff, actual_price
        )
        if pm_output.target:
            pm_error = abs(pm_output.target - actual_price)
        pm_return = _realized_return(
            pm_output.decision, price_at_cutoff, actual_price
        )

    results.append(
        TestResult(
            week=week,
            cutoff=cutoff,
            agent="pm",
            direction=pm_output.decision,
            target=pm_output.target,
            confidence=pm_output.confidence,
            price_at_cutoff=price_at_cutoff,
            actual_price=actual_price,
            direction_correct=pm_correct,
            target_error=pm_error,
            realized_return=pm_return,
            report=pm_output.report,
            entry=pm_output.entry,
            stop_loss=pm_output.stop_loss,
            key_point=pm_output.key_reason,
            winning_side=pm_output.winning_side,
            model=pm_usage.model,
            tool_calls=pm_usage.tool_calls,
            input_tokens=pm_usage.input_tokens,
            output_tokens=pm_usage.output_tokens,
            cost=pm_usage.cost,
        )
    )

    _print_agent_line(
        "PM", pm_output.decision, pm_output.target,
        pm_output.confidence, can_score, pm_correct, actual_price,
        pm_error,
    )


def _record_ta(
    results: list[TestResult],
    week: int,
    cutoff: str,
    ta_output,
    ta_usage: AgentUsage,
    price_at_cutoff: float | None,
    actual_price: float | None,
    can_score: bool,
) -> None:
    """Append the TA TestResult and print its line."""
    ta_correct = None
    ta_error = None
    ta_return = None
    if can_score:
        ta_correct = _score_direction(
            ta_output.direction, price_at_cutoff, actual_price
        )
        ta_error = abs(ta_output.target_high - actual_price)
        ta_return = _realized_return(
            ta_output.direction, price_at_cutoff, actual_price
        )

    results.append(
        TestResult(
            week=week,
            cutoff=cutoff,
            agent="ta",
            direction=ta_output.direction,
            target=ta_output.target_high,
            confidence=ta_output.confidence,
            price_at_cutoff=price_at_cutoff,
            actual_price=actual_price,
            direction_correct=ta_correct,
            target_error=ta_error,
            realized_return=ta_return,
            report=ta_output.report,
            model=ta_usage.model,
            tool_calls=ta_usage.tool_calls,
            input_tokens=ta_usage.input_tokens,
            output_tokens=ta_usage.output_tokens,
            cost=ta_usage.cost,
        )
    )
    _print_agent_line(
        "TA", ta_output.direction, ta_output.target_high,
        ta_output.confidence, can_score, ta_correct, actual_price, ta_error,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_agent_line(
    agent: str,
    direction: str,
    target: float | None,
    confidence: float,
    can_score: bool,
    correct: bool | None,
    actual_price: float | None,
    error: float | None,
) -> None:
    """Print one agent's result for a single test."""
    target_str = f"${target:,.0f}" if target else "N/A"
    print(f"  {agent}: {direction} | target={target_str} | conf={confidence:.0%}")
    if can_score and correct is not None:
        mark = "YES" if correct else "NO"
        error_str = f" | error=${error:,.0f}" if error is not None else ""
        print(f"       correct={mark} | actual=${actual_price:,.0f}{error_str}")
    else:
        print("       (no future data to score)")


def print_summary(results: list[TestResult], run_label: str = "") -> None:
    """Print aggregate evaluation summary."""
    if not results:
        print("\nNo results to summarise.")
        return

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {run_label}" if run_label else "SUMMARY")
    print(f"{'=' * 60}")

    agents = sorted(set(r.agent for r in results))

    header = f"{'Agent':<8} {'Tests':<7} {'Accuracy':<14} {'Avg Error':<12} {'Avg Conf':<10}"
    print(f"\n{header}")
    print("-" * len(header))

    for agent in agents:
        agent_results = [r for r in results if r.agent == agent]
        total = len(agent_results)

        scored = [r for r in agent_results if r.direction_correct is not None]
        if scored:
            correct = sum(1 for r in scored if r.direction_correct)
            accuracy = f"{correct}/{len(scored)} ({correct / len(scored):.0%})"
        else:
            accuracy = "no data"

        errors = [
            r.target_error for r in agent_results if r.target_error is not None
        ]
        avg_error = f"${sum(errors) / len(errors):,.0f}" if errors else "N/A"

        avg_conf = sum(r.confidence for r in agent_results) / total

        print(f"{agent:<8} {total:<7} {accuracy:<14} {avg_error:<12} {avg_conf:.0%}")

    # --- Usage breakdown ---
    print("\nUsage:")
    usage_header = (
        f"{'Agent':<8} {'Tools':<7} {'Input':<10} {'Output':<10} "
        f"{'Total':<10} {'Cost':<10}"
    )
    print(usage_header)
    print("-" * len(usage_header))

    totals = {"tool_calls": 0, "input": 0, "output": 0, "cost": 0.0}
    for agent in agents:
        ar = [r for r in results if r.agent == agent]
        tc = sum(r.tool_calls for r in ar)
        ti = sum(r.input_tokens for r in ar)
        to = sum(r.output_tokens for r in ar)
        cost = sum(r.cost for r in ar)
        totals["tool_calls"] += tc
        totals["input"] += ti
        totals["output"] += to
        totals["cost"] += cost
        print(
            f"{agent:<8} {tc:<7} {ti:<10,} {to:<10,} {ti + to:<10,} ${cost:<9.4f}"
        )

    print("-" * len(usage_header))
    print(
        f"{'TOTAL':<8} {totals['tool_calls']:<7} {totals['input']:<10,} "
        f"{totals['output']:<10,} {totals['input'] + totals['output']:<10,} "
        f"${totals['cost']:<9.4f}"
    )

    # --- Risk/return metrics ---
    metric_rows = _compute_run_metrics(results)
    if metric_rows:
        print("\nRisk/Return:")
        m_header = (
            f"{'Agent':<10} {'Weeks':<6} {'Commit':<7} {'HitRate':<9} "
            f"{'CumRet':<10} {'Sharpe':<9} {'MaxDD':<8}"
        )
        print(m_header)
        print("-" * len(m_header))
        order = {"ta": 0, "bull": 1, "bear": 2, "pm": 3, "buy_hold": 4}
        for m in sorted(metric_rows, key=lambda x: order.get(x["agent"], 99)):
            hit = (
                f"{m['committed_hit_rate']:.0%}"
                if m['committed_hit_rate'] is not None else "—"
            )
            sharpe = (
                f"{m['sharpe_ratio']:+.2f}"
                if m['sharpe_ratio'] is not None else "—"
            )
            print(
                f"{m['agent']:<10} {m['total_weeks']:<6} "
                f"{m['committed_weeks']:<7} {hit:<9} "
                f"{m['cumulative_return']:+.1%}".ljust(56)
                + f" {sharpe:<9} {m['max_drawdown']:.1%}"
            )

    # --- Calibration (needs enough data) ---
    scored_all = [r for r in results if r.direction_correct is not None]
    if len(scored_all) >= 4:
        print("\nCalibration:")
        buckets: dict[float, dict[str, int]] = {}
        for r in scored_all:
            bucket = round(r.confidence * 5) / 5  # nearest 0.2
            if bucket not in buckets:
                buckets[bucket] = {"total": 0, "correct": 0}
            buckets[bucket]["total"] += 1
            if r.direction_correct:
                buckets[bucket]["correct"] += 1

        for conf in sorted(buckets):
            b = buckets[conf]
            actual = b["correct"] / b["total"]
            print(
                f"  Stated {conf:.0%} conf -> Actual {actual:.0%}  (n={b['total']})"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Quick smoke test: 2 weeks of TA-only evaluation
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-04-06 12:00:00"
    weeks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    label = sys.argv[3] if len(sys.argv) > 3 else "cli_test"

    run_harness(
        start_date=start,
        num_weeks=weeks,
        mode="ta",
        run_label=label,
    )
