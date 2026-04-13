"""
news_agent.py — BTC news research agent.

Two-phase agent:
  1. Research  — Perplexity deep research gathers the latest BTC news.
  2. Analysis  — OpenAI reads the research and produces a structured assessment.

The structured output (direction, confidence, price prediction) comes from the
analysis model, not a separate extractor — it IS the agent's own assessment.
"""

import os
from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models import NewsOutput

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


FAKE_RESEARCH = (
    "## BTC News Research Report — {date} (FAKE / TEST DATA)\n\n"
    "### 1. Macro Environment\n"
    "Fed held rates steady at 4.25-4.50%. DXY weakened to 99.8, "
    "providing a mild tailwind for risk assets. US equities traded flat "
    "to slightly positive on the week.\n\n"
    "### 2. Crypto Market Sentiment\n"
    "Fear & Greed Index at 62 (Greed). BTC dominance rose to 54.3%. "
    "Altcoins mostly flat, with ETH underperforming. Stablecoin inflows "
    "picked up slightly on-chain.\n\n"
    "### 3. BTC-Specific News\n"
    "Spot BTC ETFs saw $320M net inflows over the past week, led by "
    "BlackRock's IBIT. A large whale moved 4,200 BTC to cold storage, "
    "reducing exchange supply. No major regulatory updates.\n\n"
    "### 4. Key Risk Events (week)\n"
    "CPI data release on Wednesday. Options expiry on Friday ($2.1B "
    "notional). No protocol upgrades scheduled.\n\n"
    "### 5. Outlook\n"
    "**Directional Bias:** Bullish\n"
    "ETF inflows remain strong and macro conditions are supportive. "
    "CPI could inject volatility mid-week, but the path of least "
    "resistance looks modestly higher.\n\n"
    "### 6. Price Prediction\n"
    "BTC likely trades in the $84,000-$89,000 range over the next week, "
    "with upside toward $89,000 if CPI comes in soft."
)


def run_news_agent(fake: bool = False) -> NewsOutput:
    """Run news research (Perplexity) then structured analysis (OpenAI).

    Args:
        fake: If True, skip the Perplexity call and use hardcoded research text.
              The OpenAI structured analysis still runs.

    Returns:
        NewsOutput with full report + structured prediction fields.
    """
    today = date.today().isoformat()

    print("\n" + "=" * 60)

    if fake:
        # --- Phase 1 SKIPPED: use hardcoded research ---
        print("BTC News Research Agent (FAKE RESEARCH)")
        print("=" * 60)
        raw_report = FAKE_RESEARCH.format(date=today)
        print(raw_report)
    else:
        # --- Phase 1: Research with Perplexity ---
        print("BTC News Research Agent")
        print("=" * 60)

        perplexity = ChatOpenAI(
            model="sonar-deep-research",
            base_url="https://api.perplexity.ai",
            api_key=os.getenv("PERPLEXITY_API_KEY"),
        )

        research_messages = [
            SystemMessage(content=NEWS_SYSTEM_PROMPT.format(date=today)),
            HumanMessage(content=NEWS_USER_PROMPT.format(today=today)),
        ]

        raw_report = ""
        for chunk in perplexity.stream(research_messages):
            content = chunk.content
            print(content, end="", flush=True)
            raw_report += content

        print()

    # --- Phase 2: Structured analysis with OpenAI ---
    print("Producing structured analysis...")
    analyst = ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    ).with_structured_output(NewsOutput)

    result = analyst.invoke([
        SystemMessage(content=NEWS_SYSTEM_PROMPT.format(date=today)),
        HumanMessage(
            content=(
                "Here is the latest BTC news research. Analyze it and provide "
                "your structured assessment:\n\n" + raw_report
            )
        ),
    ])

    return result
