"""
debate.py — Bull, Bear, and Portfolio Manager agents for BTC/USDT trading debate.

Flow:
  1. Bull opening argument
  2. Bear opening argument
  3. 2 rounds of counter-arguments (bull rebuts bear, bear rebuts bull)
  4. Bull structured final assessment
  5. Bear structured final assessment
  6. PM structured decision (sees entire transcript)

All agents see the full message history, so each rebuttal directly addresses
what the other side said.
"""

import os

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models import BearOutput, BullOutput, PMOutput

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BULL_PROMPT = """You are a bullish BTC analyst. Argue FOR a long BTC/USDT trade
using only the provided data. Be specific about entry, stop loss, and target."""

BEAR_PROMPT = """You are a bearish BTC analyst. Argue AGAINST a long BTC/USDT trade
using only the provided data. Be specific about entry, stop loss, and target."""

PM_PROMPT = """You are a portfolio manager making the final BTC/USDT trade decision after reviewing a full bull vs bear debate (opening arguments + two rebuttal rounds). You have access to the complete transcript.

EVALUATION PROCESS:

## 1. DEBATE DYNAMICS ANALYSIS
- Which side made the stronger opening argument? Why?
- Which side's argument IMPROVED more in the rebuttal round?
- Which side was FORCED to adjust their levels or thesis under pressure? What does that reveal?
- Which side made more meaningful concessions vs. deflected valid criticism?
- Did either side ignore a critical data point the other raised?

## 2. DEEP CRITIQUE QUALITY CHECK
- Did the bull's critique of the bear actually land, or was it surface-level?
- Did the bear's critique of the bull actually land, or was it surface-level?
- Were the concessions genuine or token?
- Score each side's critique quality: /10

## 3. DATA MERIT (independent of debate performance)
- Ignoring rhetoric, which side's data interpretation is more sound?
- Are there data points NEITHER side addressed that change the picture?
- What is the base case probability: up / down / sideways?

## 4. RISK ANALYSIS
- Max portfolio risk: 2% per trade
- If going with the bull: does their stop loss respect the 2% constraint? Is the stop at real structure or arbitrary?
- Highest-probability adverse scenario and its estimated likelihood

## 5. DECISION
DECISION: BULLISH | BEARISH | NEUTRAL
ENTRY: [price]
STOP LOSS: [price]
TARGET: [price]
POSITION SIZE: [% of portfolio]
CONFIDENCE: [0.0 - 1.0]
WINNING SIDE: BULL | BEAR | NEITHER
KEY REASON: [Which specific argument or data point was decisive]
WHAT WOULD CHANGE MY MIND: [One condition that would flip this decision]

RULES:
- NEUTRAL is valid — use it when neither side is convincing or confidence < 0.5
- Never exceed 2% portfolio risk regardless of conviction
- HARD RULE: Never approve a trade where risk (entry-to-stop) exceeds reward (entry-to-target). If the risk:reward ratio is worse than 1:1, the decision MUST be NEUTRAL regardless of conviction. Calculate this explicitly before deciding.
- Your job is to find truth, not pick a winner — if both sides are weak, say so
- If you disagree with BOTH sides' levels, propose your own with justification"""

NUM_REBUTTALS = 2


# ---------------------------------------------------------------------------
# Debate runner
# ---------------------------------------------------------------------------

def run_debate(ta_report: str, news_report: str) -> dict:
    """Run the full bull vs bear debate and return structured outputs.

    Args:
        ta_report: Technical analysis report text.
        news_report: News research report text.

    Returns:
        Dict with keys: messages, bull, bear, pm
    """
    oai_mini = ChatOpenAI(
        model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY")
    )

    # Market context from TA and News
    market_context = (
        f"[TECHNICAL ANALYSIS REPORT]\n{ta_report}\n\n"
        f"[NEWS RESEARCH REPORT]\n{news_report}"
    )

    # Shared transcript (both sides see everything)
    messages = []

    # --- Round 1: Opening arguments ---
    print("\n" + "=" * 60)
    print("ROUND 1: OPENING ARGUMENTS")
    print("=" * 60)

    # Bull opens
    bull_messages = [
        SystemMessage(content=BULL_PROMPT),
        HumanMessage(
            content=(
                f"[MARKET DATA]\n{market_context}\n\n"
                "Present your opening bullish argument for a long BTC/USDT trade."
            )
        ),
    ]
    bull_resp = oai_mini.invoke(bull_messages)
    messages.append(HumanMessage(content=bull_messages[-1].content))
    messages.append(AIMessage(content=f"[BULL OPENING]\n{bull_resp.content}"))
    print(f"\nBULL OPENING:\n{bull_resp.content}\n")

    # Bear opens (sees bull's opening)
    bear_prompt_msg = HumanMessage(
        content="Now present your opening bearish argument against a long BTC/USDT trade."
    )
    bear_resp = oai_mini.invoke(
        [SystemMessage(content=BEAR_PROMPT)] + messages + [bear_prompt_msg]
    )
    messages.append(bear_prompt_msg)
    messages.append(AIMessage(content=f"[BEAR OPENING]\n{bear_resp.content}"))
    print(f"\nBEAR OPENING:\n{bear_resp.content}\n")

    # --- Rebuttal rounds ---
    for round_num in range(1, NUM_REBUTTALS + 1):
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
        bull_resp = oai_mini.invoke(
            [SystemMessage(content=BULL_PROMPT)] + messages + [bull_prompt_msg]
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
        bear_resp = oai_mini.invoke(
            [SystemMessage(content=BEAR_PROMPT)] + messages + [bear_prompt_msg]
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

    # Bull final — structured output
    bull_structured = oai_mini.with_structured_output(BullOutput)
    bull_final = bull_structured.invoke(
        [SystemMessage(content=BULL_PROMPT)]
        + messages
        + [
            HumanMessage(
                content=(
                    "Based on the complete debate, provide your final structured "
                    "bullish assessment with entry, stop loss, target, and confidence."
                )
            )
        ]
    )
    print(
        f"\nBULL FINAL: entry=${bull_final.entry:,.0f} "
        f"target=${bull_final.target:,.0f} stop=${bull_final.stop_loss:,.0f} "
        f"confidence={bull_final.confidence:.0%}"
    )
    print(f"  Key argument: {bull_final.key_argument}")

    # Bear final
    bear_structured = oai_mini.with_structured_output(BearOutput)
    bear_final = bear_structured.invoke(
        [SystemMessage(content=BEAR_PROMPT)]
        + messages
        + [
            HumanMessage(
                content=(
                    "Based on the complete debate, provide your final structured "
                    "bearish assessment with entry, stop loss, target, and confidence."
                )
            )
        ]
    )
    print(
        f"\nBEAR FINAL: entry=${bear_final.entry:,.0f} "
        f"target=${bear_final.target:,.0f} stop=${bear_final.stop_loss:,.0f} "
        f"confidence={bear_final.confidence:.0%}"
    )
    print(f"  Key argument: {bear_final.key_argument}")

    # PM decision — same model, structured output
    pm_structured = oai_mini.with_structured_output(PMOutput)
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
    pm_final = pm_structured.invoke(
        [SystemMessage(content=PM_PROMPT)]
        + messages
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
    )

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
