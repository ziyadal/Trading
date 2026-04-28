"""
debate.py — Bull and Bear debate (PM moved to pm.py).

Flow:
  1. Bull and Bear opening arguments (independent — neither sees the other's opening)
  2. 2 rounds of counter-arguments (both see full transcript)
  3. Bull structured final assessment
  4. Bear structured final assessment

PM evaluation is now a separate step (pm.py) so it gets its own checkpoint /
rollback boundary in the LangGraph pipeline.
"""

from langchain.agents import create_agent
from langchain_community.callbacks import get_openai_callback
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from llm import make_llm
from models import BearOutput, BullOutput
from prompts.debate import BEAR_PROMPT, BULL_PROMPT, NUM_REBUTTALS
from usage import AgentUsage, add_callback


def run_debate(
    ta_report: str,
    news_report: str,
    bull_prompt: str | None = None,
    bear_prompt: str | None = None,
    bull_model: str | None = None,
    bear_model: str | None = None,
    num_rebuttals: int = NUM_REBUTTALS,
) -> dict:
    """Run bull vs bear debate (openings + rebuttals + final structured assessments).

    Args:
        ta_report: Technical analysis report text.
        news_report: News research report text.
        bull_prompt: Override bull system prompt (default: BULL_PROMPT).
        bear_prompt: Override bear system prompt (default: BEAR_PROMPT).
        bull_model: Override bull model (default: gpt-4.1-mini).
        bear_model: Override bear model (default: gpt-4.1-mini).
        num_rebuttals: Number of rebuttal rounds (default: 2).

    Returns:
        Dict with keys: messages, bull, bear, usage (bull_usage + bear_usage).
    """
    _bull_prompt = bull_prompt or BULL_PROMPT
    _bear_prompt = bear_prompt or BEAR_PROMPT

    _bull_model = bull_model or "openai/gpt-4.1-mini"
    _bear_model = bear_model or "openai/gpt-4.1-mini"

    bull_llm = make_llm(_bull_model)
    bear_llm = make_llm(_bear_model)

    # Per-agent usage accumulators. Bull/bear never use tools in the debate.
    bull_usage = AgentUsage(agent="bull", model=_bull_model)
    bear_usage = AgentUsage(agent="bear", model=_bear_model)

    market_context = (
        f"<TA_Report>\n{ta_report}\n</TA_Report>\n\n"
        f"<News_Report>\n{news_report}\n</News_Report>"
    )

    messages: list[BaseMessage] = []

    # --- Round 1: Opening arguments (independent — neither sees the other) ---
    print("\n" + "=" * 60)
    print("ROUND 1: OPENING ARGUMENTS (independent)")
    print("=" * 60)

    open_prompt = (
        f"[MARKET DATA]\n{market_context}\n\n"
        "Present your opening argument for BTC/USDT based on the data above."
    )

    with get_openai_callback() as cb:
        bull_resp = bull_llm.invoke([
            SystemMessage(content=_bull_prompt),
            HumanMessage(content=open_prompt),
        ])
    add_callback(bull_usage, cb)
    print(f"\nBULL OPENING:\n{bull_resp.content}\n")

    with get_openai_callback() as cb:
        bear_resp = bear_llm.invoke([
            SystemMessage(content=_bear_prompt),
            HumanMessage(content=open_prompt),
        ])
    add_callback(bear_usage, cb)
    print(f"\nBEAR OPENING:\n{bear_resp.content}\n")

    messages.append(HumanMessage(content=open_prompt))
    messages.append(AIMessage(content=f"[BULL OPENING]\n{bull_resp.content}"))
    messages.append(AIMessage(content=f"[BEAR OPENING]\n{bear_resp.content}"))

    # --- Rebuttal rounds ---
    for round_num in range(1, num_rebuttals + 1):
        print("=" * 60)
        print(f"ROUND {round_num + 1}: COUNTER-ARGUMENTS")
        print("=" * 60)

        bull_prompt_msg = HumanMessage(
            content=(
                f"Counter-argument round {round_num}: Bull, respond to the bear's "
                "latest argument. You MUST introduce at least one NEW argument, "
                "data point, or angle not raised in any previous round. "
                "Do not repeat prior points — only reference them briefly if needed "
                "to build on them. Keep your response under 200 words."
            )
        )
        with get_openai_callback() as cb:
            bull_resp = bull_llm.invoke(
                [SystemMessage(content=_bull_prompt)] + messages + [bull_prompt_msg]
            )
        add_callback(bull_usage, cb)
        messages.append(bull_prompt_msg)
        messages.append(
            AIMessage(content=f"[BULL REBUTTAL {round_num}]\n{bull_resp.content}")
        )
        print(f"\nBULL REBUTTAL {round_num}:\n{bull_resp.content}\n")

        bear_prompt_msg = HumanMessage(
            content=(
                f"Counter-argument round {round_num}: Bear, respond to the bull's "
                "latest argument. You MUST introduce at least one NEW argument, "
                "data point, or angle not raised in any previous round. "
                "Do not repeat prior points — only reference them briefly if needed "
                "to build on them. Keep your response under 200 words."
            )
        )
        with get_openai_callback() as cb:
            bear_resp = bear_llm.invoke(
                [SystemMessage(content=_bear_prompt)] + messages + [bear_prompt_msg]
            )
        add_callback(bear_usage, cb)
        messages.append(bear_prompt_msg)
        messages.append(
            AIMessage(content=f"[BEAR REBUTTAL {round_num}]\n{bear_resp.content}")
        )
        print(f"\nBEAR REBUTTAL {round_num}:\n{bear_resp.content}\n")

    # --- Structured final assessments ---
    print("=" * 60)
    print("FINAL STRUCTURED ASSESSMENTS")
    print("=" * 60)

    bull_agent = create_agent(
        model=bull_llm,
        tools=[],
        system_prompt=_bull_prompt,
        response_format=BullOutput,
    )
    with get_openai_callback() as cb:
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
    add_callback(bull_usage, cb)
    print(
        f"\nBULL FINAL: entry=${bull_final.entry:,.0f} "
        f"target=${bull_final.target:,.0f} stop=${bull_final.stop_loss:,.0f} "
        f"confidence={bull_final.confidence:.0%}"
    )
    print(f"  Key argument: {bull_final.key_argument}")

    bear_agent = create_agent(
        model=bear_llm,
        tools=[],
        system_prompt=_bear_prompt,
        response_format=BearOutput,
    )
    with get_openai_callback() as cb:
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
    add_callback(bear_usage, cb)
    print(
        f"\nBEAR FINAL: entry=${bear_final.entry:,.0f} "
        f"target=${bear_final.target:,.0f} stop=${bear_final.stop_loss:,.0f} "
        f"confidence={bear_final.confidence:.0%}"
    )
    print(f"  Key argument: {bear_final.key_argument}")

    return {
        "messages": messages,
        "bull": bull_final,
        "bear": bear_final,
        "usage": {
            "bull": bull_usage,
            "bear": bear_usage,
        },
    }
