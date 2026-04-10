"""
news_agent.py — BTC news research agent powered by Perplexity deep research.
"""

import os
from datetime import date

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv(override=True)

NEWS_SYSTEM_PROMPT = """You are a professional cryptocurrency news researcher specialising in Bitcoin.

Your job is to research and summarise the most important developments from the last week
that are relevant to BTC price action. Your primary focus is Bitcoin, but you must include
relevant macro context and broader crypto market sentiment where it affects BTC.

RULES
-----
- Ground every claim in recent, verifiable news
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

### 4. Key Risk Events (week)
[Scheduled macro events, options expiries, protocol upgrades, known catalysts]

### 5. Outlook
**Directional Bias:** Bullish / Neutral / Bearish
[2-3 sentences synthesising the above into a directional view for BTC over the next week]

### 6. Price Prediction
[Predict the price of BTC in 1 week]
"""

NEWS_USER_PROMPT = """Research and write a BTC news report covering the last week.
Focus primarily on Bitcoin, with relevant macro and crypto context.
Today's date: {today}"""


def run_news_agent() -> str:
    """Run the Perplexity-powered news research agent and return the report."""
    client = ChatOpenAI(
        model="sonar-deep-research",
        base_url="https://api.perplexity.ai",
        api_key=os.getenv("PERPLEXITY_API_KEY"),
    )

    today = date.today().isoformat()
    messages = [
        SystemMessage(content=NEWS_SYSTEM_PROMPT.format(date=today)),
        HumanMessage(content=NEWS_USER_PROMPT.format(today=today)),
    ]

    print("\n" + "=" * 60)
    print("BTC News Research Agent — streaming output")
    print("=" * 60)

    full_report = ""
    for chunk in client.stream(messages):
        content = chunk.content
        print(content, end="", flush=True)
        full_report += content

    print()
    return full_report
