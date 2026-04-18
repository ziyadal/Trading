"""
news_agent.py — BTC news research agent.

Single-phase agent using Perplexity's structured output:
  Perplexity sonar-pro researches the latest BTC news and returns a structured
  NewsOutput directly via response_format (JSON Schema). No separate OpenAI call.
"""

import os
from datetime import date

from openai import OpenAI

from models import NewsOutput
from prompts.news import FAKE_RESEARCH, NEWS_SYSTEM_PROMPT, NEWS_USER_PROMPT


def run_news_agent(fake: bool = False) -> NewsOutput:
    """Run BTC news research via Perplexity with structured output.

    Args:
        fake: If True, skip the Perplexity call and return hardcoded data.

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

    client = OpenAI(
        api_key=os.getenv("PERPLEXITY_API_KEY"),
        base_url="https://api.perplexity.ai",
    )

    response = client.chat.completions.create(
        model="sonar-pro",
        messages=[
            {"role": "system", "content": NEWS_SYSTEM_PROMPT.format(date=today)},
            {"role": "user", "content": NEWS_USER_PROMPT.format(today=today)},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"schema": NewsOutput.model_json_schema()},
        },
    )

    content = response.choices[0].message.content
    assert content is not None, "Perplexity returned empty response"
    result = NewsOutput.model_validate_json(content)

    print(f"\n{result.report}\n")
    return result
