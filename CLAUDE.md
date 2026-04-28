# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent AI trading system for BTC/USDT. Uses LLM-powered agents (LangChain/LangGraph) to analyze crypto markets via technical analysis and news research, then runs a structured bull vs bear debate to produce a final trading decision.

## Commands

Uses **uv** for dependency management (Python 3.12):

```bash
uv sync                                          # Install dependencies
uv run main.py                                   # Live run (fetches fresh data)
uv run main.py --fake-news                       # Live run with fake news (skip Perplexity)
uv run data.py                                   # Refresh market data only
uv run data.py --backfill 90                     # Backfill 90 days of history
uv run ta_agent.py                               # Run TA agent standalone
uv run pytest                                    # Run tests
uv run pyright                                   # Type check
```

## Architecture

The pipeline runs sequentially in `main.py:run_pipeline()`:

1. **Data ingestion** (`data.py`): Fetches 2 days of BTC/USDT 5m candles from Binance via CCXT, computes indicators (RSI-14, MACD, Bollinger Bands, EMA-50), upserts into `trading.db` SQLite. Also supports `--backfill` for historical data.

2. **Analysis agents** (run in sequence):
   - `ta_agent.py`: Agent with a `query_db` SQL tool against `trading.db`. Uses `gpt-4.1-mini` with `response_format=TAOutput` for structured output. Queries use LIMIT clauses to stay within API token limits. Produces a 1-week price target with volume confirmation. Supports a `cutoff` parameter for backtesting (used by the harness).
   - `news_agent.py`: Single-phase — `perplexity/sonar-pro` (via OpenRouter) with `response_format` (JSON Schema) returns a structured `NewsOutput` directly. No separate OpenAI call.

3. **Debate** (`debate.py`): Bull and Bear agents (`gpt-4.1-mini`) make independent opening arguments (neither sees the other's opening), then 2 rebuttal rounds with full transcript visibility (each round must introduce new arguments). Bull/bear produce structured final assessments; PM reviews the transcript + both sides' numbers and renders the final decision. Risk:reward is intentionally not computed in v2 — execution concerns (R:R gating, sizing) are deferred to v3.

4. **Evaluation** (`harness.py`): The harness runs the pipeline across multiple historical cutoff dates, scores predictions against actual prices, and logs results to `eval.db`. Each cutoff is wrapped in try/except so one failure doesn't kill the whole backtest.

All structured outputs are Pydantic models in `models.py`: `TAOutput`, `NewsOutput`, `BullOutput`, `BearOutput`, `PMOutput`.

**Usage tracking** (`usage.py`): Every OpenAI call (TA + bull + bear + PM) is wrapped in `get_openai_callback()`, which yields tokens and cost from LangChain's internal price table — no hand-rolled pricing. Per-agent totals are accumulated into `AgentUsage` records and persisted to `eval.db`. News agent (Perplexity) isn't tracked because the harness uses fake news.

## Key Patterns

- **Structured output**: Debate rounds use free-text chat; final assessments use `ChatOpenAI.with_structured_output(PydanticModel)` directly. The TA agent uses `response_format=` on `create_agent`.
- **Harness backtesting**: The harness (`harness.py`) passes a cutoff timestamp to `ta_agent.py`, which replaces `btc_ohlcv` in SQL queries with a filtered subquery so the agent only sees data up to that time.
- **Two databases**: `trading.db` = market data (OHLCV + indicators). `eval.db` = harness evaluation results — two tables: `harness_results` (per-agent predictions + full reasoning: `report`, `key_point`, `winning_side`, trade levels + per-agent `model`, `tool_calls`, `input_tokens`, `output_tokens`, `cost`) and `run_totals` (one row per harness run with aggregate usage). Schema evolves via `_ensure_columns()` which issues `ALTER TABLE ADD COLUMN` so older eval databases pick up new columns automatically.
- **Grounding rules**: TA, bull, bear, and PM prompts all forbid citing data not present in their inputs (no funding rates, on-chain metrics, options open interest, order books). Added after a backtest showed agents fabricating evidence from these sources.
- **TA verdict structure**: The TA prompt requires the agent to state a bullish scenario AND a bearish scenario before picking a direction, and to default to NEUTRAL unless one side materially outweighs the other.

## Key Constraints

- **Risk rules are non-negotiable** — max 2% portfolio risk per trade, stop loss on every trade. These are enforced in the PM system prompt, not in code. Do not make them configurable or bypassable.
- `.env` contains API keys (`OPENAI_API_KEY`, `PERPLEXITY_API_KEY`) — never commit it.


Keep explanations clear and simple where possible do not over include technical jargon when asked to explain something. Explain to someone who is learning Computer science. 
