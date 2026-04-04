# BTC News Research Agent — Design Spec

**Date:** 2026-04-04  
**Status:** Approved

---

## Overview

A standalone news research agent that uses Perplexity (via its OpenAI-compatible API) as the LLM. Perplexity's built-in live web search means a single API call returns a grounded, cited news report — no tool loop required. The agent produces a structured 5-section report covering macro, crypto, and BTC-specific news with a directional 24h outlook.

---

## File: `news_agent.py`

### Structure

Follows the same pattern as `app.py`:
- Module-level constants: `SYSTEM_PROMPT`, `USER_PROMPT`
- `_build_client() -> ChatOpenAI` — returns a `ChatOpenAI` instance configured for Perplexity
- `run_news_agent(client=None) -> str` — executes the research call, streams output, returns full report string. Accepts an optional `client` argument so tests can inject a mock without network calls
- `main()` — calls `run_news_agent()`, gated by `if __name__ == "__main__"`

### Perplexity Client

```python
ChatOpenAI(
    model="sonar-pro",
    base_url="https://api.perplexity.ai",
    api_key=os.getenv("PERPLEXITY_API_KEY"),
)
```

Uses LangChain's `ChatOpenAI` with Perplexity's OpenAI-compatible endpoint. No custom HTTP client needed.

### Streaming

```python
for chunk in client.stream([SystemMessage(...), HumanMessage(...)]):
    print(chunk.content, end="", flush=True)
    full_report += chunk.content
```

Same streaming pattern as `app.py` — prints tokens as they arrive.

---

## Prompts

### System Prompt

Instructs Perplexity to act as a professional crypto news researcher with a BTC focus. Tells it to:
- Prioritise BTC-specific developments but include relevant macro and crypto context
- Ground every claim in recent news (within the last 24 hours where possible)
- Output the report in the exact structured format below
- Be concise and factual — no speculation beyond the Outlook section

### User Prompt

```
Research and write a BTC news report covering the last 24 hours.
Focus primarily on Bitcoin, with relevant macro and crypto context.
Today's date: {today}
```

Date is injected at runtime so Perplexity knows the current date window.

### Report Format (instructed in system prompt)

```
## BTC News Research Report — {date}

### 1. Macro Environment
[Fed policy, DXY, risk-on/off sentiment, equity market context]

### 2. Crypto Market Sentiment
[Overall crypto mood, BTC dominance, notable altcoin moves, Fear & Greed]

### 3. BTC-Specific News
[ETF flows, whale activity, exchange news, regulatory updates, on-chain highlights]

### 4. Key Risk Events (Next 24–48h)
[Scheduled macro events, protocol upgrades, expiries, known catalysts]

### 5. 24h Outlook
**Directional Bias:** Bullish / Neutral / Bearish
[2–3 sentence synthesis of the above into a directional view]
```

---

## Tracing

Out-of-the-box LangSmith tracing via environment variables — no code changes needed:
- `LANGCHAIN_TRACING_V2=true`
- `LANGCHAIN_API_KEY=<your key>`

Since `ChatOpenAI` is a LangChain component, all LLM calls, token counts, and latency are captured automatically.

---

## Unit Testing (`tests/test_news_agent.py`)

Tests inject a mock client so no network calls are made.

- **`test_run_returns_string`** — `run_news_agent(client=mock)` returns a non-empty string
- **`test_report_contains_section_headers`** — returned string contains all 5 expected section headings (`Macro Environment`, `Crypto Market Sentiment`, `BTC-Specific News`, `Key Risk Events`, `24h Outlook`)
- **`test_api_error_raises_cleanly`** — if the mock client raises an exception, `run_news_agent` propagates it (news agent fails loudly — no silent empty reports)
- **`test_date_injected_in_prompt`** — user prompt passed to the client contains today's date string

---

## Dependencies to Add

- None — `langchain-openai` already supports custom `base_url`

## New Environment Variables

- `PERPLEXITY_API_KEY` — Perplexity API key (add to `.env`, never commit)
- `LANGCHAIN_TRACING_V2=true` — enable LangSmith tracing (optional)
- `LANGCHAIN_API_KEY` — LangSmith API key (optional, only needed if tracing enabled)

## Files to Create

- `news_agent.py` — news research agent
- `tests/test_news_agent.py` — unit tests
