"""
main.py — Full BTC/USDT trading pipeline.

Orchestrates three stages as a LangGraph:
  1. Technical Analysis agent  (ta_agent.py)
  2. News Research agent       (news_agent.py)
  3. Bull vs Bear Debate       (debate.py)
"""

from typing import Annotated

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from debate import init_tracking_db, print_analytics, run_debate
from news_agent import run_news_agent
from ta_agent import run_ta_agent

load_dotenv(override=True)


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict):
    ta_report: str
    news_report: str
    debate_result: dict


# ---------------------------------------------------------------------------
# LangGraph nodes
# ---------------------------------------------------------------------------

def ta_node(state: PipelineState) -> dict:
    """Run the Technical Analysis agent."""
    report = run_ta_agent()
    return {"ta_report": report}


def news_node(state: PipelineState) -> dict:
    """Run the News Research agent."""
    report = run_news_agent()
    return {"news_report": report}


def debate_node(state: PipelineState) -> dict:
    """Run the Bull vs Bear debate with Judge decision."""
    print("\n" + "=" * 60)
    print("BTC/USDT Bull vs Bear Debate")
    print("=" * 60)

    final_state = run_debate(
        ta_report=state["ta_report"],
        news_report=state["news_report"],
    )
    return {"debate_result": final_state}


def summary_node(state: PipelineState) -> dict:
    """Print post-run summary and analytics."""
    final = state["debate_result"]

    print(f"\n{'='*60}")
    print("POST-RUN SUMMARY")
    print(f"{'='*60}")
    print(f"  Rounds completed: {final['current_round'] - 1}")
    print(f"  Messages: {len(final['messages'])}")

    for i, msg in enumerate(final["messages"], 1):
        speaker = getattr(msg, "name", "Unknown")
        preview = msg.content[:80].replace("\n", " ")
        print(f"    {i}. [{speaker}] {preview}...")

    print(f"\n{'='*60}")
    print("STRUCTURED PREDICTIONS")
    print(f"{'='*60}")
    for pred in final["predictions"]:
        stop = f"${pred.stop_loss:,.0f}" if pred.stop_loss else "N/A"
        print(
            f"  {pred.agent_name:>6}: {pred.direction:>7} @ "
            f"target=${pred.target_price:,.0f}  stop={stop}  "
            f"confidence={pred.confidence:.0%}  "
            f"timeframe={pred.timeframe_hours}h"
        )

    print(f"\n{'='*60}")
    print("HISTORICAL PERFORMANCE")
    print(f"{'='*60}")
    print_analytics()

    return {}


# ---------------------------------------------------------------------------
# Build and run the pipeline graph
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("technical_analysis", ta_node)
    graph.add_node("news_research", news_node)
    graph.add_node("debate", debate_node)
    graph.add_node("summary", summary_node)

    # TA and News run sequentially (TA first to refresh market data),
    # then debate consumes both reports, then summary.
    graph.add_edge(START, "technical_analysis")
    graph.add_edge("technical_analysis", "news_research")
    graph.add_edge("news_research", "debate")
    graph.add_edge("debate", "summary")
    graph.add_edge("summary", END)

    return graph.compile()


def main() -> None:
    init_tracking_db()
    pipeline = build_pipeline()
    pipeline.invoke({
        "ta_report": "",
        "news_report": "",
        "debate_result": {},
    })


if __name__ == "__main__":
    main()
