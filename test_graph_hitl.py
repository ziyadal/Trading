"""Smoke test: run the full pipeline with fake news + cheap models, verify
HITL interrupt + resume + checkpoint history (the 4 rollback points)."""

import os
import sqlite3
import sys

from dotenv import load_dotenv
from langgraph.types import Command

load_dotenv(override=True)
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

from data import refresh_db
from graph import build_graph, make_saver, make_thread_id

CHEAP = "openai/gpt-4.1-nano"


def main():
    # Use a fresh checkpoint DB for the test
    db_path = "checkpoints.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("Refreshing market data (skip if already current)...")
    try:
        refresh_db()
    except Exception as e:
        print(f"  refresh_db failed (continuing with stale data): {e}")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = make_saver(conn)

    graph = build_graph(saver, with_hitl=True)
    thread_id = make_thread_id("live")
    print(f"\nThread: {thread_id}")

    config = {
        "configurable": {
            "thread_id": thread_id,
            "fake_news": True,
            "ta_model": CHEAP,
            "bull_model": CHEAP,
            "bear_model": CHEAP,
            "pm_model": CHEAP,
        },
        "metadata": {
            "run_type": "live-test",
            "fake_news": True,
        },
    }

    # ---- 1. Run until HITL interrupt ----
    print("\n>>> Invoking graph (will pause at HITL gate)...")
    result = graph.invoke({}, config=config)

    snapshot = graph.get_state(config)
    print(f"\n>>> Paused at: {snapshot.next}")
    pm = snapshot.values.get("pm_output")
    assert pm is not None, "pm_output missing — graph didn't reach hitl"
    print(f">>> PM decision: {pm.decision} confidence={pm.confidence:.0%}")

    # ---- 2. Check checkpoint history (the rollback points) ----
    print("\n>>> Checkpoint history (= rollback points):")
    history = list(graph.get_state_history(config))
    # Most-recent first; we want a readable list
    for i, snap in enumerate(reversed(history)):
        nodes_done = list(snap.values.keys()) if snap.values else []
        next_nodes = list(snap.next) if snap.next else ["END"]
        print(f"  cp{i}: next={next_nodes}  state_keys={sorted(nodes_done)}")

    # ---- 3. Resume with human "approve" ----
    print("\n>>> Resuming with Command(resume=approve)...")
    graph.invoke(
        Command(resume={"decision": "approve", "notes": "auto-approved in test"}),
        config=config,
    )

    final = graph.get_state(config).values
    print(f"\n>>> Final state:")
    print(f"  ta_output:      {'OK' if final.get('ta_output') else 'MISSING'}")
    print(f"  news_output:    {'OK' if final.get('news_output') else 'MISSING'}")
    print(f"  bull_output:    {'OK' if final.get('bull_output') else 'MISSING'}")
    print(f"  bear_output:    {'OK' if final.get('bear_output') else 'MISSING'}")
    print(f"  pm_output:      {'OK' if final.get('pm_output') else 'MISSING'}")
    print(f"  human_decision: {final.get('human_decision')}")
    print(f"  human_notes:    {final.get('human_notes')}")

    # ---- 4. Demonstrate rollback: pick the post-debate checkpoint and
    # show we could re-run pm from there ----
    print("\n>>> Rollback demo: finding the post-debate checkpoint...")
    history = list(graph.get_state_history(config))
    # find first checkpoint where bull/bear are present but pm is not
    for snap in history:
        v = snap.values
        if v.get("bull_output") and v.get("bear_output") and not v.get("pm_output"):
            print(f"  Found cp at {snap.config['configurable']['checkpoint_id']}")
            print(f"  next from there would be: {list(snap.next)}")
            print(f"  -> would re-run PM only if invoked with this checkpoint_id")
            break
    else:
        print("  (no post-debate-pre-pm checkpoint found — graph may have advanced too quickly)")

    conn.close()
    print("\n>>> Test complete.")


if __name__ == "__main__":
    main()
