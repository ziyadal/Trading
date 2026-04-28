"""
news_agent.py — BTC news research agent.

Single-phase agent using Perplexity's structured output (via OpenRouter):
  perplexity/sonar-pro researches the latest BTC news and returns a structured
  NewsOutput directly via response_format (JSON Schema). No separate OpenAI call.
"""

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage

from llm import make_llm
from models import NewsOutput
from prompts.news import FAKE_RESEARCH, NEWS_SYSTEM_PROMPT, NEWS_USER_PROMPT


def run_news_agent(fake: bool = False, model_name: str | None = None) -> NewsOutput:
    """Run BTC news research via Perplexity (through OpenRouter) with structured output.

    Args:
        fake: If True, skip the Perplexity call and return hardcoded data.
        model_name: Override the default model (perplexity/sonar-pro).

    Returns:
        NewsOutput with full report + structured prediction fields.
    """
    today = date.today().isoformat()

    print("\n" + "=" * 60)

    if fake:
        print("BTC News Research Agent (FAKE RESEARCH)")
        print("=" * 60)
        report = FAKE_RESEARCH.format(date=today)
        print(report)
        return NewsOutput(
            report=report,
            direction="NEUTRAL",
            price_prediction=0,
            confidence=0.0,
            key_catalyst="no comment",
        )

    print("BTC News Research Agent")
    print("=" * 60)

    _model = model_name or "perplexity/sonar-pro"
    llm = make_llm(_model).with_structured_output(NewsOutput, method="json_schema")

    result: NewsOutput = llm.invoke([
        SystemMessage(content=NEWS_SYSTEM_PROMPT.format(date=today)),
        HumanMessage(content=NEWS_USER_PROMPT.format(today=today)),
    ])

    print(f"\n{result.report}\n")
    return result
