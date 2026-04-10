"""
debate.py — Bull, Bear, and Portfolio Manager agents for BTC/USDT trading debate.
"""

from langchain.agents import create_agent

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

BULL_PROMPT = """You are a bullish BTC analyst. Argue FOR a long BTC/USDT trade
using only the provided data. Be specific about entry, stop loss, and target."""

BEAR_PROMPT = """You are a bearish BTC analyst. Argue AGAINST a long BTC/USDT trade
using only the provided data. Be specific about risk levels."""

PM_PROMPT = """You are a portfolio manager making the final BTC/USDT trade decision.
Weigh bull and bear arguments on merit. Max 2% portfolio risk per trade.
Output: DECISION (BUY/SELL/NO TRADE), ENTRY, STOP LOSS, TARGET, CONFIDENCE (0-1)."""

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

bull_agent = create_agent("openai:gpt-4.1-mini", tools=[], prompt=BULL_PROMPT)
bear_agent = create_agent("openai:gpt-4.1-mini", tools=[], prompt=BEAR_PROMPT)
pm_agent = create_agent("openai:gpt-4.1-mini", tools=[], prompt=PM_PROMPT)



