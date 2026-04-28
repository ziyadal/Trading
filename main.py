"""
main.py — BTC/USDT trading pipeline orchestrator.

Builds the LangGraph pipeline with HITL enabled, runs it, and resumes after
the human approves/rejects the PM's decision.

Usage:
    uv run main.py                  # live run
    uv run main.py --fake-news      # live run with fake news (skip Perplexity)
    uv run main.py --thread <id>    # resume an existing thread (e.g. after crash)
"""

import sqlite3
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv
from langgraph.types import Command

from data import refresh_db
from graph import CHECKPOINT_DB, build_graph, make_saver, make_thread_id

# LLM reports contain non-ASCII chars (≈, –, etc.); Windows' cp1252 console
# crashes on them. Force UTF-8 for all stdout writes.
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

load_dotenv(override=True)


def _print_pm_summary(pm) -> None:
    print(f"\n{'=' * 60}")
    print(f"PM DECISION: {pm.decision}")
    if pm.entry:
        print(f"  Entry:      ${pm.entry:,.0f}")
        print(f"  Stop Loss:  ${pm.stop_loss:,.0f}")
        print(f"  Target:     ${pm.target:,.0f}")
        print(f"  Size:       {pm.position_size}%")
    print(f"  Confidence: {pm.confidence:.0%}")
    print(f"  Winner:     {pm.winning_side}")
    print(f"  Reason:     {pm.key_reason}")
    print(f"{'=' * 60}")


def _prompt_human(pm) -> dict:
    """Terminal-based HITL gate. Returns {decision, notes}."""
    print("\n" + "=" * 60)
    print("HUMAN-IN-THE-LOOP REVIEW")
    print("=" * 60)
    _print_pm_summary(pm)

    while True:
        choice = input(
            "\nApprove this trade? [a]pprove / [r]eject / [e]dit notes: "
        ).strip().lower()
        if choice in ("a", "approve"):
            return {"decision": "approve", "notes": None}
        if choice in ("r", "reject"):
            notes = input("Reason for rejection (optional): ").strip() or None
            return {"decision": "reject", "notes": notes}
        if choice in ("e", "edit"):
            notes = input("Notes/edit instructions: ").strip()
            return {"decision": "edit", "notes": notes}
        print("Pick a/r/e.")


def run_pipeline(
    fake_news: bool = False,
    thread_id: str | None = None,
    ta_model: str | None = None,
    bull_model: str | None = None,
    bear_model: str | None = None,
    pm_model: str | None = None,
) -> None:
    """Run the full trading pipeline (live data) with HITL gate.

    Args:
        fake_news: If True, use hardcoded news data instead of calling Perplexity.
        thread_id: If set, resume an existing thread (e.g. after a crash).
                   Otherwise generates a fresh `live_<uuidv7>` id.
        *_model: Override per-agent models (forwarded via configurable).
    """
    if not thread_id:
        print("Refreshing market data...")
        refresh_db()

    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    saver = make_saver(conn)

    graph = build_graph(saver, with_hitl=True)

    if thread_id:
        print(f"Resuming thread: {thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        # When resuming after the HITL interrupt, we need to feed the human's
        # decision via Command(resume=...). Otherwise, just continue.
        snapshot = graph.get_state(config)
        if snapshot.next == ("hitl",) or "hitl" in snapshot.next:
            # Was paused at HITL — fetch PM output and prompt human
            pm = snapshot.values.get("pm_output")
            decision = _prompt_human(pm)
            graph.invoke(Command(resume=decision), config=config)
        else:
            graph.invoke(None, config=config)
    else:
        thread_id = make_thread_id("live")
        print(f"Thread: {thread_id}")
        config = {
            "configurable": {
                "thread_id": thread_id,
                "fake_news": fake_news,
                "ta_model": ta_model,
                "bull_model": bull_model,
                "bear_model": bear_model,
                "pm_model": pm_model,
            },
            "metadata": {
                "run_type": "live",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "fake_news": fake_news,
            },
        }

        # Run until the HITL interrupt
        graph.invoke({}, config=config)

        # We're paused at HITL — read state and prompt human
        snapshot = graph.get_state(config)
        if snapshot.next:  # graph isn't done — must be at hitl interrupt
            pm = snapshot.values.get("pm_output")
            decision = _prompt_human(pm)
            graph.invoke(Command(resume=decision), config=config)

    final = graph.get_state(config).values
    pm = final.get("pm_output")
    if pm is not None:
        _print_pm_summary(pm)
    print(f"\nHuman decision: {final.get('human_decision')}")
    if final.get("human_notes"):
        print(f"Notes: {final.get('human_notes')}")
    print(f"\nThread: {config['configurable']['thread_id']}")
    print("(Use --thread <id> to resume or rollback this run.)")
    conn.close()


if __name__ == "__main__":
    fake = "--fake-news" in sys.argv
    tid = None
    if "--thread" in sys.argv:
        i = sys.argv.index("--thread")
        if i + 1 < len(sys.argv):
            tid = sys.argv[i + 1]
    run_pipeline(fake_news=fake, thread_id=tid)
