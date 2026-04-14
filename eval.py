"""
eval.py — Prediction logging and scoring for the trading pipeline.

Stores every agent's structured prediction in SQLite so you can compare
performance across different versions/configurations of the system.
"""

import sqlite3
from datetime import datetime, timezone

from models import BearOutput, BullOutput, NewsOutput, PMOutput, TAOutput

EVAL_DB = "eval.db"


def init_eval_db(db_path: str = EVAL_DB) -> None:
    """Create the eval_log table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS eval_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      TEXT NOT NULL,
            run_label   TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            agent       TEXT NOT NULL,
            direction   TEXT,
            target_price REAL,
            stop_loss   REAL,
            confidence  REAL,
            raw_output  TEXT,
            outcome     TEXT DEFAULT 'PENDING',
            price_actual REAL,
            score       REAL
        )
    """)
    conn.commit()
    conn.close()


def log_run(
    run_label: str,
    ta: TAOutput,
    news: NewsOutput,
    bull: BullOutput,
    bear: BearOutput,
    pm: PMOutput,
    db_path: str = EVAL_DB,
) -> str:
    """Log all predictions from a pipeline run. Returns the run_id."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    timestamp = datetime.now(timezone.utc).isoformat()

    rows = [
        ("ta", ta.direction, ta.target_high, None, ta.confidence, ta.report),
        ("news", news.direction, news.price_prediction, None, news.confidence, news.report),
        ("bull", "BULLISH", bull.target, bull.stop_loss, bull.confidence, bull.report),
        ("bear", "BEARISH", bear.target, bear.stop_loss, bear.confidence, bear.report),
        ("pm", pm.decision, pm.target, pm.stop_loss, pm.confidence, pm.report),
    ]

    conn = sqlite3.connect(db_path)
    for agent, direction, target, stop, confidence, report in rows:
        conn.execute(
            """INSERT INTO eval_log
               (run_id, run_label, timestamp, agent, direction,
                target_price, stop_loss, confidence, raw_output)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (run_id, run_label, timestamp, agent, direction, target, stop, confidence, report),
        )
    conn.commit()
    conn.close()

    print(f"\nLogged predictions for run {run_id} ({run_label}) at {timestamp}")
    return run_id


def init_harness_table(db_path: str = EVAL_DB) -> None:
    """Create the harness_results table if it doesn't exist."""
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
            target_error    REAL
        )
    """)
    conn.commit()
    conn.close()


def log_harness_results(
    results: list,
    run_label: str,
    db_path: str = EVAL_DB,
) -> None:
    """Save harness TestResult objects to the harness_results table."""
    init_harness_table(db_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(db_path)
    for r in results:
        conn.execute(
            """INSERT INTO harness_results
               (run_label, run_timestamp, week, cutoff, agent,
                direction, target, confidence, price_at_cutoff,
                actual_price, direction_correct, target_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_label, timestamp, r.week, r.cutoff, r.agent,
                r.direction, r.target, r.confidence, r.price_at_cutoff,
                r.actual_price,
                None if r.direction_correct is None else int(r.direction_correct),
                r.target_error,
            ),
        )
    conn.commit()
    conn.close()
    print(f"\nSaved {len(results)} harness results to eval.db (label: {run_label})")


def print_comparison(db_path: str = EVAL_DB) -> None:
    """Print a comparison of scores across run labels."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT run_label, agent, run_id, timestamp,
               COUNT(*) as n,
               AVG(confidence) as avg_confidence,
               AVG(CASE WHEN score IS NOT NULL THEN score END) as avg_score
        FROM eval_log
        GROUP BY run_label, agent
        ORDER BY timestamp DESC, run_label, agent
    """).fetchall()
    conn.close()

    if not rows:
        print("No evaluation data yet.")
        return

    print(f"\n{'Label':<20} {'Agent':<8} {'Run Time':<22} {'Runs':<6} {'Avg Conf':<10} {'Avg Score':<10}")
    print("-" * 78)
    for row in rows:
        score = f"{row['avg_score']:.2f}" if row["avg_score"] is not None else "pending"
        # Show ISO timestamp truncated to minutes for readability
        run_time = row["timestamp"][:16] if row["timestamp"] else "N/A"
        print(
            f"{row['run_label']:<20} {row['agent']:<8} {run_time:<22} {row['n']:<6} "
            f"{row['avg_confidence']:.2f}      {score}"
        )
