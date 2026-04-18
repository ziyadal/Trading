"""
debate.py — Bull, Bear, and Portfolio Manager agents for BTC/USDT trading debate.

Flow:
  1. Bull and Bear opening arguments (independent — neither sees the other's opening)
  2. 2 rounds of counter-arguments (both see full transcript)
  3. Bull structured final assessment
  4. Bear structured final assessment
  5. PM structured decision (sees entire transcript)
"""

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models import BearOutput, BullOutput, PMOutput
from prompts.debate import BEAR_PROMPT, BULL_PROMPT, NUM_REBUTTALS, PM_PROMPT


# ---------------------------------------------------------------------------
# Debate runner
# ---------------------------------------------------------------------------

def run_debate(
    ta_report: str,
    news_report: str,
    bull_prompt: str | None = None,
    bear_prompt: str | None = None,
    pm_prompt: str | None = None,
    bull_model: str | None = None,
    bear_model: str | None = None,
    pm_model: str | None = None,
    num_rebuttals: int = NUM_REBUTTALS,
) -> dict:
    """Run the full bull vs bear debate and return structured outputs.

    Args:
        ta_report: Technical analysis report text.
        news_report: News research report text.
        bull_prompt: Override bull system prompt (default: BULL_PROMPT).
        bear_prompt: Override bear system prompt (default: BEAR_PROMPT).
        pm_prompt: Override PM system prompt (default: PM_PROMPT).
        bull_model: Override bull model (default: gpt-4.1-mini).
        bear_model: Override bear model (default: gpt-4.1-mini).
        pm_model: Override PM model (default: gpt-4.1-mini).
        num_rebuttals: Number of rebuttal rounds (default: 2).

    Returns:
        Dict with keys: messages, bull, bear, pm
    """
    _bull_prompt = bull_prompt or BULL_PROMPT
    _bear_prompt = bear_prompt or BEAR_PROMPT
    _pm_prompt = pm_prompt or PM_PROMPT

    bull_llm = ChatOpenAI(
        model=bull_model or "gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")
    )
    bear_llm = ChatOpenAI(
        model=bear_model or "gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")
    )
    pm_llm = ChatOpenAI(
        model=pm_model or "gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")
    )

    # Market context from TA and News
    market_context = (
        f"[TECHNICAL ANALYSIS REPORT]\n{ta_report}\n\n"
        f"[NEWS RESEARCH REPORT]\n{news_report}"
    )

    # Shared transcript (both sides see everything after openings)
    messages: list = []

    # --- Round 1: Opening arguments (independent — neither sees the other) ---
    print("\n" + "=" * 60)
    print("ROUND 1: OPENING ARGUMENTS (independent)")
    print("=" * 60)

    open_prompt = (
        f"[MARKET DATA]\n{market_context}\n\n"
        "Present your opening argument for BTC/USDT based on the data above."
    )

    # Bull opens (sees only market data)
    bull_resp = bull_llm.invoke([
        SystemMessage(content=_bull_prompt),
        HumanMessage(content=open_prompt),
    ])
    print(f"\nBULL OPENING:\n{bull_resp.content}\n")

    # Bear opens independently (sees only market data, NOT bull's opening)
    bear_resp = bear_llm.invoke([
        SystemMessage(content=_bear_prompt),
        HumanMessage(content=open_prompt),
    ])
    print(f"\nBEAR OPENING:\n{bear_resp.content}\n")

    # Combine both openings into the shared transcript for rebuttals
    messages.append(HumanMessage(content=open_prompt))
    messages.append(AIMessage(content=f"[BULL OPENING]\n{bull_resp.content}"))
    messages.append(AIMessage(content=f"[BEAR OPENING]\n{bear_resp.content}"))

    # --- Rebuttal rounds ---
    for round_num in range(1, num_rebuttals + 1):
        print("=" * 60)
        print(f"ROUND {round_num + 1}: COUNTER-ARGUMENTS")
        print("=" * 60)

        # Bull counters
        bull_prompt_msg = HumanMessage(
            content=(
                f"Counter-argument round {round_num}: Bull, respond to the bear's "
                "latest argument. You MUST introduce at least one NEW argument, "
                "data point, or angle not raised in any previous round. "
                "Do not repeat prior points — only reference them briefly if needed "
                "to build on them. Keep your response under 200 words."
            )
        )
        bull_resp = bull_llm.invoke(
            [SystemMessage(content=_bull_prompt)] + messages + [bull_prompt_msg]
        )
        messages.append(bull_prompt_msg)
        messages.append(
            AIMessage(content=f"[BULL REBUTTAL {round_num}]\n{bull_resp.content}")
        )
        print(f"\nBULL REBUTTAL {round_num}:\n{bull_resp.content}\n")

        # Bear counters
        bear_prompt_msg = HumanMessage(
            content=(
                f"Counter-argument round {round_num}: Bear, respond to the bull's "
                "latest argument. You MUST introduce at least one NEW argument, "
                "data point, or angle not raised in any previous round. "
                "Do not repeat prior points — only reference them briefly if needed "
                "to build on them. Keep your response under 200 words."
            )
        )
        bear_resp = bear_llm.invoke(
            [SystemMessage(content=_bear_prompt)] + messages + [bear_prompt_msg]
        )
        messages.append(bear_prompt_msg)
        messages.append(
            AIMessage(content=f"[BEAR REBUTTAL {round_num}]\n{bear_resp.content}")
        )
        print(f"\nBEAR REBUTTAL {round_num}:\n{bear_resp.content}\n")

    # --- Structured final assessments ---
    print("=" * 60)
    print("FINAL STRUCTURED ASSESSMENTS")
    print("=" * 60)

    # Bull final — structured output via create_agent(response_format=)
    bull_agent = create_agent(
        model=bull_llm,
        tools=[],
        system_prompt=_bull_prompt,
        response_format=BullOutput,
    )
    bull_final: BullOutput = bull_agent.invoke({
        "messages": messages
        + [
            HumanMessage(
                content=(
                    "Based on the complete debate, provide your final structured "
                    "bullish assessment with entry, stop loss, target, and confidence."
                )
            )
        ]
    })["structured_response"]
    print(
        f"\nBULL FINAL: entry=${bull_final.entry:,.0f} "
        f"target=${bull_final.target:,.0f} stop=${bull_final.stop_loss:,.0f} "
        f"confidence={bull_final.confidence:.0%}"
    )
    print(f"  Key argument: {bull_final.key_argument}")

    # Bear final — structured output via create_agent(response_format=)
    bear_agent = create_agent(
        model=bear_llm,
        tools=[],
        system_prompt=_bear_prompt,
        response_format=BearOutput,
    )
    bear_final: BearOutput = bear_agent.invoke({
        "messages": messages
        + [
            HumanMessage(
                content=(
                    "Based on the complete debate, provide your final structured "
                    "bearish assessment with entry, stop loss, target, and confidence."
                )
            )
        ]
    })["structured_response"]
    print(
        f"\nBEAR FINAL: entry=${bear_final.entry:,.0f} "
        f"target=${bear_final.target:,.0f} stop=${bear_final.stop_loss:,.0f} "
        f"confidence={bear_final.confidence:.0%}"
    )
    print(f"  Key argument: {bear_final.key_argument}")

    # PM decision — structured output via create_agent(response_format=)
    pm_agent = create_agent(
        model=pm_llm,
        tools=[],
        system_prompt=_pm_prompt,
        response_format=PMOutput,
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
    pm_final: PMOutput = pm_agent.invoke({
        "messages": messages
        + [
            HumanMessage(
                content=(
                    "You have reviewed the complete bull vs bear debate transcript above.\n\n"
                    f"{bull_summary}\n{bear_summary}\n\n"
                    "Now perform your full evaluation and render your final trading decision. "
                    "Calculate the risk:reward ratio explicitly for any trade you consider."
                )
            )
        ]
    })["structured_response"]

    entry_str = f"${pm_final.entry:,.0f}" if pm_final.entry else "N/A"
    target_str = f"${pm_final.target:,.0f}" if pm_final.target else "N/A"
    stop_str = f"${pm_final.stop_loss:,.0f}" if pm_final.stop_loss else "N/A"
    print(
        f"\nPM DECISION: {pm_final.decision} entry={entry_str} "
        f"target={target_str} stop={stop_str} "
        f"confidence={pm_final.confidence:.0%} winner={pm_final.winning_side}"
    )

    return {
        "messages": messages,
        "bull": bull_final,
        "bear": bear_final,
        "pm": pm_final,
    }
