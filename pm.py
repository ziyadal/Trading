"""
pm.py — Portfolio Manager agent.

Reviews the bull vs bear debate transcript plus both sides' final structured
numbers, then renders the final trading decision.

Split from debate.py so the PM step can be its own LangGraph node (separate
checkpoint + rollback boundary).
"""

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import BaseMessage, HumanMessage

from llm import make_llm
from models import BearOutput, BullOutput, PMOutput
from prompts.debate import PM_PROMPT
from usage import AgentUsage, add_callback


def run_pm(
    transcript: list[BaseMessage],
    bull_final: BullOutput,
    bear_final: BearOutput,
    pm_prompt: str | None = None,
    pm_model: str | None = None,
) -> tuple[PMOutput, AgentUsage]:
    """Render the PM's final trading decision from a completed bull/bear debate.

    Args:
        transcript: Full bull/bear debate transcript (openings + rebuttals).
        bull_final: Bull's final structured assessment.
        bear_final: Bear's final structured assessment.
        pm_prompt: Override PM system prompt (default: PM_PROMPT).
        pm_model: Override PM model (default: openai/gpt-4.1-mini).

    Returns:
        (PMOutput, AgentUsage)
    """
    _pm_prompt = pm_prompt or PM_PROMPT
    _pm_model = pm_model or "openai/gpt-4.1-mini"

    pm_llm = make_llm(_pm_model)
    pm_usage = AgentUsage(agent="pm", model=_pm_model)

    # ToolStrategy (instead of ProviderStrategy) because PMOutput has Optional[float]
    # fields that don't round-trip cleanly through OpenRouter's strict-schema path.
    pm_agent = create_agent(
        model=pm_llm,
        tools=[],
        system_prompt=_pm_prompt,
        response_format=ToolStrategy(PMOutput),
    )

    bull_summary = (
        f"[BULL FINAL NUMBERS] entry=${bull_final.entry:,.0f} "
        f"stop=${bull_final.stop_loss:,.0f} target=${bull_final.target:,.0f} "
        f"confidence={bull_final.confidence:.0%}"
    )
    bear_summary = (
        f"[BEAR FINAL NUMBERS] entry=${bear_final.entry:,.0f} "
        f"stop=${bear_final.stop_loss:,.0f} target=${bear_final.target:,.0f} "
        f"confidence={bear_final.confidence:.0%}"
    )

    with get_openai_callback() as cb:
        pm_final: PMOutput = pm_agent.invoke({
            "messages": transcript
            + [
                HumanMessage(
                    content=(
                        "You have reviewed the complete bull vs bear debate transcript above.\n\n"
                        f"{bull_summary}\n{bear_summary}\n\n"
                        "Now perform your full evaluation and render your final trading decision."
                    )
                )
            ]
        })["structured_response"]
    add_callback(pm_usage, cb)

    entry_str = f"${pm_final.entry:,.0f}" if pm_final.entry else "N/A"
    target_str = f"${pm_final.target:,.0f}" if pm_final.target else "N/A"
    stop_str = f"${pm_final.stop_loss:,.0f}" if pm_final.stop_loss else "N/A"
    print(
        f"\nPM DECISION: {pm_final.decision} entry={entry_str} "
        f"target={target_str} stop={stop_str} "
        f"confidence={pm_final.confidence:.0%} winner={pm_final.winning_side}"
    )

    return pm_final, pm_usage
