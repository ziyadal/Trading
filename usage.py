"""
usage.py — Per-agent token / cost / tool-call tracking for the harness.

Cost and tokens come from LangChain's get_openai_callback() context manager,
which uses LangChain's internal price table (no hand-rolled pricing dict).
Tool calls are counted from the message trajectory.

Only covers OpenAI agents (TA, bull, bear, PM). The harness uses fake news,
so news_agent isn't tracked here.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AgentUsage:
    agent: str
    model: str
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0  # 0.0 when unknown (e.g. Perplexity)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def usage_from_callback(
    agent: str, model: str, cb: Any, tool_calls: int = 0
) -> AgentUsage:
    """Build an AgentUsage from a get_openai_callback() result."""
    return AgentUsage(
        agent=agent,
        model=model,
        tool_calls=tool_calls,
        input_tokens=cb.prompt_tokens,
        output_tokens=cb.completion_tokens,
        cost=cb.total_cost,
    )


def count_tool_calls(messages: list[Any]) -> int:
    """Count tool calls across all AIMessages in a trajectory."""
    return sum(len(getattr(m, "tool_calls", None) or []) for m in messages)


def add_callback(usage: AgentUsage, cb: Any) -> None:
    """Accumulate a get_openai_callback() result into an existing usage."""
    usage.input_tokens += cb.prompt_tokens
    usage.output_tokens += cb.completion_tokens
    usage.cost += cb.total_cost
