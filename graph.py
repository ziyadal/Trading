"""
graph.py — LangGraph pipeline for the BTC trading agents.

Wraps TA → News → Debate → PM → (optional HITL) as a StateGraph with a
SQLite checkpointer, giving us:

  - rollback at any node boundary (4 checkpoint points before HITL),
  - a human-in-the-loop gate before any future trade execution,
  - per-cutoff resumability for the harness via thread_id.

Per-run config (model names, prompts, cutoff, fake_news) flows through
LangGraph's native config["configurable"] channel, not state — so nodes
read it via their (state, config) signature.
"""

from __future__ import annotations

import os
import time
from typing import Literal, TypedDict

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt

from debate import run_debate
from models import BearOutput, BullOutput, NewsOutput, PMOutput, TAOutput
from news_agent import run_news_agent
from pm import run_pm
from prompts.news import FAKE_RESEARCH
from ta_agent import run_ta_agent
from usage import AgentUsage

CHECKPOINT_DB = "checkpoints.db"

# Custom Pydantic / dataclass types we put into checkpoint state. Listing them
# here silences LangGraph's "unregistered msgpack type" deserialization warnings
# and keeps us forward-compatible (the warning says strict mode is coming).
_ALLOWED_CHECKPOINT_TYPES = [
    ("models", "TAOutput"),
    ("models", "NewsOutput"),
    ("models", "BullOutput"),
    ("models", "BearOutput"),
    ("models", "PMOutput"),
    ("usage", "AgentUsage"),
]


def make_saver(conn) -> SqliteSaver:
    """Build a SqliteSaver with our project's checkpoint type allowlist."""
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_CHECKPOINT_TYPES)
    saver = SqliteSaver(conn, serde=serde)
    saver.setup()
    return saver


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class TradingState(TypedDict, total=False):
    """Shared state passed between graph nodes.

    `total=False` so each node can return a partial dict — LangGraph merges
    partials into the prior state automatically.
    """

    # Agent outputs
    ta_output: TAOutput | None
    news_output: NewsOutput | None
    debate_messages: list[BaseMessage]
    bull_output: BullOutput | None
    bear_output: BearOutput | None
    pm_output: PMOutput | None

    # Per-agent usage
    ta_usage: AgentUsage | None
    bull_usage: AgentUsage | None
    bear_usage: AgentUsage | None
    pm_usage: AgentUsage | None

    # HITL outcome (set by the hitl node when graph compiled with_hitl=True)
    human_decision: Literal["approve", "reject", "edit"] | None
    human_notes: str | None


# ---------------------------------------------------------------------------
# Thread IDs (UUIDv7 — RFC 9562; sortable + unique)
# ---------------------------------------------------------------------------

def uuid7() -> str:
    """Generate a UUIDv7 string. Python stdlib doesn't ship one until 3.14."""
    ts_ms = int(time.time() * 1000) & 0xFFFFFFFFFFFF      # 48 bits
    rand_a = int.from_bytes(os.urandom(2), "big") & 0x0FFF  # 12 bits
    rand_b = int.from_bytes(os.urandom(8), "big") & 0x3FFFFFFFFFFFFFFF  # 62 bits

    value = (ts_ms << 80) | (0x7 << 76) | (rand_a << 64) | (0b10 << 62) | rand_b
    h = f"{value:032x}"
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def make_thread_id(source: Literal["live", "harness"]) -> str:
    """Generate a `<source>_<uuidv7>` thread id (greppable + sortable)."""
    return f"{source}_{uuid7()}"


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def ta_node(state: TradingState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    output, usage = run_ta_agent(
        cutoff=cfg.get("cutoff"),
        system_prompt=cfg.get("ta_prompt"),
        user_prompt=cfg.get("ta_user_prompt"),
        model_name=cfg.get("ta_model"),
    )
    return {"ta_output": output, "ta_usage": usage}


def news_node(state: TradingState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    fake = cfg.get("fake_news", False)
    cutoff = cfg.get("cutoff")

    if fake:
        # When fake, use the harness-style FAKE_RESEARCH so the report carries
        # a date stamp the debate prompts can reference.
        date_str = (cutoff or "")[:10]
        report = FAKE_RESEARCH.format(date=date_str) if date_str else FAKE_RESEARCH
        output = NewsOutput(
            report=report,
            direction="NEUTRAL",
            price_prediction=0,
            confidence=0.0,
            key_catalyst="no comment",
        )
        return {"news_output": output}

    output = run_news_agent(fake=False, model_name=cfg.get("news_model"))
    return {"news_output": output}


def debate_node(state: TradingState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    ta_out = state["ta_output"]
    news_out = state["news_output"]
    assert ta_out is not None and news_out is not None

    result = run_debate(
        ta_report=ta_out.report,
        news_report=news_out.report,
        bull_prompt=cfg.get("bull_prompt"),
        bear_prompt=cfg.get("bear_prompt"),
        bull_model=cfg.get("bull_model"),
        bear_model=cfg.get("bear_model"),
        num_rebuttals=cfg.get("num_rebuttals", 2),
    )
    return {
        "debate_messages": result["messages"],
        "bull_output": result["bull"],
        "bear_output": result["bear"],
        "bull_usage": result["usage"]["bull"],
        "bear_usage": result["usage"]["bear"],
    }


def pm_node(state: TradingState, config: RunnableConfig) -> dict:
    cfg = config.get("configurable", {})
    transcript = state["debate_messages"]
    bull = state["bull_output"]
    bear = state["bear_output"]
    assert bull is not None and bear is not None

    output, usage = run_pm(
        transcript=transcript,
        bull_final=bull,
        bear_final=bear,
        pm_prompt=cfg.get("pm_prompt"),
        pm_model=cfg.get("pm_model"),
    )
    return {"pm_output": output, "pm_usage": usage}


def hitl_node(state: TradingState, config: RunnableConfig) -> dict:
    """Human-in-the-loop gate. Pauses graph until resumed with a decision.

    The graph is compiled with this node only when `with_hitl=True`. The caller
    resumes via `graph.invoke(Command(resume={"decision": ..., "notes": ...}), config)`.
    """
    pm = state["pm_output"]
    assert pm is not None

    response = interrupt({
        "type": "trade_approval",
        "decision": pm.decision,
        "entry": pm.entry,
        "stop_loss": pm.stop_loss,
        "target": pm.target,
        "position_size": pm.position_size,
        "confidence": pm.confidence,
        "winning_side": pm.winning_side,
        "key_reason": pm.key_reason,
    })

    if isinstance(response, dict):
        decision = response.get("decision", "reject")
        notes = response.get("notes")
    else:
        decision = str(response)
        notes = None

    return {"human_decision": decision, "human_notes": notes}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(
    checkpointer: SqliteSaver | None,
    *,
    with_hitl: bool = True,
) -> CompiledStateGraph:
    """Compile the trading pipeline graph.

    Args:
        checkpointer: SqliteSaver instance, or None to run without persistence.
        with_hitl: When True (default, for live runs), inserts a `hitl` node
                   between `pm` and END that calls interrupt(). When False
                   (harness mode), `pm` flows straight to END.

    Returns:
        A compiled LangGraph ready for .invoke()/.stream().
    """
    g: StateGraph = StateGraph(TradingState)
    g.add_node("ta", ta_node)
    g.add_node("news", news_node)
    g.add_node("debate", debate_node)
    g.add_node("pm", pm_node)

    g.add_edge(START, "ta")
    g.add_edge("ta", "news")
    g.add_edge("news", "debate")
    g.add_edge("debate", "pm")

    if with_hitl:
        g.add_node("hitl", hitl_node)
        g.add_edge("pm", "hitl")
        g.add_edge("hitl", END)
    else:
        g.add_edge("pm", END)

    return g.compile(checkpointer=checkpointer)
