"""
news_agent.py — BTC News Research Agent powered by Perplexity.

Perplexity's built-in live web search means a single API call returns a
grounded, cited news report — no tool loop required.

Usage:
    uv run python news_agent.py
"""

import os
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a professional cryptocurrency news researcher specialising in Bitcoin.

Your job is to research and summarise the most important developments from the last 24 hours
that are relevant to BTC price action. Your primary focus is Bitcoin, but you must include
relevant macro context and broader crypto market sentiment where it affects BTC.

RULES
-----
- Ground every claim in recent, verifiable news (prioritise the last 24 hours)
- Be concise and factual — no speculation except in the Outlook section
- If a section has no significant news, write "No significant developments in this period"
- Output the report in EXACTLY the format specified — no extra sections, no deviations

REPORT FORMAT
-------------
## BTC News Research Report — {date}

### 1. Macro Environment
[Fed policy, DXY, risk-on/off sentiment, equity market context that affects crypto]

### 2. Crypto Market Sentiment
[Overall crypto mood, BTC dominance, notable altcoin moves, Fear & Greed Index]

### 3. BTC-Specific News
[ETF flows, whale activity, exchange news, regulatory updates, on-chain highlights]

### 4. Key Risk Events (Next 24–48h)
[Scheduled macro events, options expiries, protocol upgrades, known catalysts]

### 5. 24h Outlook
**Directional Bias:** Bullish / Neutral / Bearish
[2–3 sentences synthesising the above into a directional view for BTC over the next 24 hours]
"""

USER_PROMPT = """Research and write a BTC news report covering the last 24 hours.
Focus primarily on Bitcoin, with relevant macro and crypto context.
Today's date: {today}"""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

def _build_client() -> ChatOpenAI:
    return ChatOpenAI(
        model="sonar-pro",
        base_url="https://api.perplexity.ai",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Main agent function
# ---------------------------------------------------------------------------

def run_news_agent(client: ChatOpenAI | None = None) -> str:
    """Run the news research agent and return the full report as a string.

    Streams output to stdout as tokens arrive.  Accepts an optional *client*
    so tests can inject a mock without making network calls.
    """
    if client is None:
        client = _build_client()

    today = date.today().isoformat()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT.format(date=today)),
        HumanMessage(content=USER_PROMPT.format(today=today)),
    ]

    full_report = ""
    for chunk in client.stream(messages):
        content = chunk.content
        print(content, end="", flush=True)
        full_report += content

    print()  # final newline
    return full_report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("BTC News Research Agent — streaming output")
    print("=" * 60)
    run_news_agent()


if __name__ == "__main__":
    main()
