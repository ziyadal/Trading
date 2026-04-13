# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent AI trading system for BTC/USDT. Uses LLM-powered agents (LangChain/LangGraph) to analyze crypto markets via technical analysis and news research, then runs a structured bull vs bear debate to produce a final trading decision.

## Commands

Uses **uv** for dependency management (Python 3.12):

```bash
uv sync                                          # Install dependencies
uv run main.py                                   # Live run (fetches fresh data)
uv run main.py "my_label"                        # Live run with custom eval label
uv run main.py "backtest_v1" "2026-04-01 12:00:00"  # Backtest with cutoff timestamp
uv run data.py                                   # Refresh market data only
uv run ta_agent.py                               # Run TA agent standalone
uv run pytest                                    # Run tests
```

## Architecture

The pipeline runs sequentially in `main.py:run_pipeline()`:

1. **Data ingestion** (`data.py`): Fetches 2 days of BTC/USDT 5m candles from Binance via CCXT, computes indicators (RSI-14, MACD, Bollinger Bands, EMA-50), upserts into `trading.db` SQLite.

2. **Analysis agents** (run in sequence):
   - `ta_agent.py`: ReAct agent with a `query_db` SQL tool against `trading.db`. Uses `gpt-4.1` with `response_format=TAOutput` for structured output. Queries use LIMIT clauses (100 rows for full-table scans, 50 for filtered) to stay within API token limits. Produces a 1-week price target with volume confirmation. Supports backtest mode via a `cutoff` parameter that transparently filters the SQL table.
   - `news_agent.py`: Two-phase — Perplexity `sonar-deep-research` for raw research, then `gpt-4.1-mini` with `.with_structured_output(NewsOutput)` for structured assessment.

3. **Debate** (`debate.py`): Bull and Bear agents (`gpt-4.1-mini`) exchange opening arguments + 2 rebuttal rounds (with escalation — each round must introduce new arguments). PM sees the full transcript plus structured bull/bear numbers before deciding. Hard rule: R:R must be >= 1:1 or PM goes NEUTRAL.

4. **Eval logging** (`eval.py`): Logs all agent predictions (direction, targets, confidence) to `eval.db` SQLite for cross-run comparison.

All structured outputs are Pydantic models in `models.py`: `TAOutput`, `NewsOutput`, `BullOutput`, `BearOutput`, `PMOutput`.

## Key Patterns

- **Structured output**: Debate rounds use free-text `create_agent`; final assessments use `ChatOpenAI.with_structured_output(PydanticModel)` directly. The TA agent uses `response_format=` on `create_agent`.
- **Backtest mode**: Pass a cutoff timestamp to `run_pipeline()` or via CLI arg. `ta_agent.py` replaces `btc_ohlcv` in SQL queries with a filtered subquery. Data refresh is skipped in backtest mode.
- **Two databases**: `trading.db` = market data (OHLCV + indicators), `eval.db` = prediction logs.

## Key Constraints

- **Risk rules are non-negotiable** — max 2% portfolio risk per trade, stop loss on every trade. These are enforced in the PM system prompt, not in code. Do not make them configurable or bypassable.
- `.env` contains API keys (`OPENAI_API_KEY`, `PERPLEXITY_API_KEY`) — never commit it.


Keep explanations clear and simple where possible do not over include technical jargon when asked to explain something. Explain to someone who is learning Computer science. 
