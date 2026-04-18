# Simplify Codebase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove backtest mode from main.py, remove unused eval logging (eval_log table, log_run, print_comparison, init_eval_db), simplify main.py to live-only, update all docs, and verify with pytest + pyright.

**Architecture:** main.py becomes live-only (always refreshes data, no cutoff). eval.py keeps only harness-related functions. The harness remains the single place for backtesting and evaluation. CLAUDE.md and architecture.md are updated to match.

**Tech Stack:** Python 3.12, Pydantic, LangChain, pytest, pyright

---

### Task 1: Simplify main.py — remove backtest mode and eval logging

**Files:**
- Modify: `main.py`

- [ ] **Step 1: Simplify run_pipeline() — remove cutoff, run_label, and eval logging**

Remove the `cutoff` and `run_label` parameters. Remove the `fake_news` parameter (live mode always calls real news). Remove the `if not cutoff` guard around data refresh — it always refreshes now. Remove the eval import, `init_eval_db()`, `log_run()`, and `print_comparison()` calls.

New `main.py`:

```python
"""
main.py — BTC/USDT trading pipeline orchestrator.

Usage:
    uv run main.py                  # live run
    uv run main.py --fake-news      # live run with fake news (skip Perplexity)
"""

import sys

from dotenv import load_dotenv

from data import refresh_db
from debate import run_debate
from news_agent import run_news_agent
from ta_agent import run_ta_agent

load_dotenv(override=True)


def run_pipeline(fake_news: bool = False) -> None:
    """Run the full trading pipeline (live data only).

    Args:
        fake_news: If True, use hardcoded news data instead of calling Perplexity.
    """
    # 1. Fresh market data
    print("Refreshing market data...")
    refresh_db()

    # 2. Analysis agents
    ta_output = run_ta_agent()

    print(f"\n{'=' * 60}")
    print("TA STRUCTURED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Direction:   {ta_output.direction}")
    print(f"  Price:       ${ta_output.current_price:,.0f}")
    print(f"  Target:      ${ta_output.target_low:,.0f} – ${ta_output.target_high:,.0f}")
    print(f"  Support:     ${ta_output.key_support:,.0f}")
    print(f"  Resistance:  ${ta_output.key_resistance:,.0f}")
    print(f"  Confidence:  {ta_output.confidence:.0%}")

    print(f"\n{'=' * 60}")
    print(f"TA REPORT:  {ta_output.report}")

    news_output = run_news_agent(fake=fake_news)

    print(f"\n{'=' * 60}")
    print("NEWS STRUCTURED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Direction:      {news_output.direction}")
    print(f"  Price Target:   ${news_output.price_prediction:,.0f}")
    print(f"  Key Catalyst:   {news_output.key_catalyst}")
    print(f"  Confidence:     {news_output.confidence:.0%}")

    # 3. Debate + PM decision
    print("\n" + "=" * 60)
    print("BTC/USDT Bull vs Bear Debate")
    print("=" * 60)
    debate_result = run_debate(ta_output.report, news_output.report)

    # 4. Print final decision
    pm = debate_result["pm"]
    print(f"\n{'=' * 60}")
    print(f"FINAL DECISION: {pm.decision}")
    if pm.entry:
        print(f"  Entry:      ${pm.entry:,.0f}")
        print(f"  Stop Loss:  ${pm.stop_loss:,.0f}")
        print(f"  Target:     ${pm.target:,.0f}")
        print(f"  Size:       {pm.position_size}%")
    print(f"  Confidence: {pm.confidence:.0%}")
    print(f"  Winner:     {pm.winning_side}")
    print(f"  Reason:     {pm.key_reason}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    use_fake_news = "--fake-news" in sys.argv
    run_pipeline(fake_news=use_fake_news)
```

- [ ] **Step 2: Run pyright to verify**

Run: `uv run pyright main.py`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "simplify: remove backtest mode and eval logging from main.py"
```

### Task 2: Strip eval.py down to harness-only functions

**Files:**
- Modify: `eval.py`

- [ ] **Step 1: Remove init_eval_db, log_run, print_comparison — keep only harness functions**

New `eval.py`:

```python
"""
eval.py — Harness result logging for the trading pipeline.

Stores scored predictions from harness evaluation runs in SQLite
so you can compare performance across different prompts/models.
"""

import sqlite3
from datetime import datetime, timezone

EVAL_DB = "eval.db"


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
```

- [ ] **Step 2: Run pyright to verify**

Run: `uv run pyright eval.py`
Expected: 0 errors

- [ ] **Step 3: Commit**

```bash
git add eval.py
git commit -m "simplify: remove unused eval_log functions, keep harness logging only"
```

### Task 3: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove backtest CLI example, update eval description**

Remove the backtest command example. Remove mention of `run_label` CLI arg. Update the eval logging description to mention only harness results. Remove the "Backtest mode" key pattern.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md to reflect simplified pipeline"
```

### Task 4: Update architecture.md

**Files:**
- Modify: `docs/architecture.md`

- [ ] **Step 1: Update component descriptions and diagrams**

Update the main.py description (no cutoff, no eval logging). Update eval.py description (harness-only). Remove the "Backtest Mode" section under "Two Modes" — replace with a note that backtesting is done via the harness. Remove `eval_log` from the databases table.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: update architecture doc to reflect simplified pipeline"
```

### Task 5: Update backlog.md

**Files:**
- Modify: `docs/backlog.md`

- [ ] **Step 1: Check off completed simplification work**

- [ ] **Step 2: Commit**

```bash
git add docs/backlog.md
git commit -m "docs: update backlog progress"
```

### Task 6: Final verification

- [ ] **Step 1: Run pyright across the whole project**

Run: `uv run pyright`
Expected: 0 errors

- [ ] **Step 2: Run pytest**

Run: `uv run pytest`
Expected: passes (no test files yet, so 0 collected is fine)

- [ ] **Step 3: Smoke test — run TA agent standalone (no Perplexity)**

Run: `uv run ta_agent.py`
Expected: agent runs, queries trading.db, prints report + structured output
