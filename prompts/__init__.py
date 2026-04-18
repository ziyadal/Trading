"""Prompts package — one module per agent.

Keeps prompt strings separate from agent logic so tuning a prompt is a
single-file diff.
"""

from prompts.debate import BEAR_PROMPT, BULL_PROMPT, NUM_REBUTTALS, PM_PROMPT
from prompts.news import FAKE_RESEARCH, NEWS_SYSTEM_PROMPT, NEWS_USER_PROMPT
from prompts.ta import TA_SYSTEM_PROMPT, TA_USER_PROMPT

__all__ = [
    "TA_SYSTEM_PROMPT",
    "TA_USER_PROMPT",
    "NEWS_SYSTEM_PROMPT",
    "NEWS_USER_PROMPT",
    "FAKE_RESEARCH",
    "BULL_PROMPT",
    "BEAR_PROMPT",
    "PM_PROMPT",
    "NUM_REBUTTALS",
]
