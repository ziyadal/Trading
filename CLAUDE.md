# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent AI trading system that uses LLM-powered agents (via LangChain/LangGraph) to analyze crypto markets, make trading decisions, and execute trades. Currently in **early planning/prototyping phase** — the codebase is mostly scaffolding with a project blueprint in `notes/PROJECT_BLUEPRINT.md`.

**Target market:** Cryptocurrency first 

## Commands

This project uses **uv** for dependency management (Python 3.12):

```bash
uv sync              # Install dependencies
uv run main.py       # Run the main entry point
uv run pytest        # Run tests (none exist yet)
uv add <package>     # Add a dependency
```

## Architecture (Planned)

Three-layer agent system orchestrated by LangGraph:

1. **Analysis Layer** (parallel): Technical Analyst, Sentiment Analyst, News Analyst, Fundamentals Analyst — each produces signals with confidence scores
2. **Decision Layer**: Research Synthesizer (dialectical reasoning, consensus scoring) → Trader Agent (position sizing, entry/exit)
3. **Execution Layer**: Risk Manager (veto power, circuit breakers) → Order Executor → Position Tracker

See `notes/PROJECT_BLUEPRINT.md` for the full architecture diagram, phased roadmap, risk rules, and technology choices.

## Tech Stack

- **Agent framework:** LangChain + LangGraph (currently using `langchain.agents.create_agent`)
- **LLM:** OpenAI gpt-4.1 (configured in code, key in `.env`)
- **Planned additions:** CCXT (crypto exchange API), Backtrader (backtesting), FinBERT (sentiment NLP)

## Key Constraints

- **Risk rules are non-negotiable** — max 1-2% risk per trade, stop loss on every trade, circuit breakers at 5%/10%/15% drawdown levels. These must be hardcoded, not configurable to bypass.
- Config should live in YAML files (`config/`), not hardcoded values.
- All agent decisions must be logged with full reasoning chains (planned: structlog).
- `.env` contains API keys — never commit it.

## Current State

- `main.py`: Placeholder entry point
- `test.ipynb`: Prototype notebook showing basic LangChain agent creation with OpenAI
- `notes/`: Project blueprint, market research docs, and brainstorming notes
- No `src/`, `config/`, or `tests/` directories have been created yet — the blueprint's directory structure is aspirational
