# BTC/USDT Multi-Agent Trading System

A research-grade trading system that uses **five LLM-powered agents** to analyse Bitcoin price action, gather news, debate a trade thesis, and propose a one-week directional call — with a human approval gate before any decision is committed.

This is a portfolio project for agentic-AI engineering roles, not a live trading bot. The focus is on building a multi-agent system that is **measurably better than a sensible baseline**, runs cheaply, evaluates itself rigorously, and is honest about its limits.

---

## Headline numbers

Latest backtest: 50 weekly cutoffs from May 2025 to April 2026 (44 with realised 1-week returns), all on `gpt-4.1-mini`.

| | Cumulative return | Sharpe ratio | Max drawdown | Direction hit rate |
|---|---|---|---|---|
| **PM (this system)** | **+20.7%** | **0.86** | **11.8%** | 51.7% on 29 committed weeks |
| Buy-and-hold BTC | −31.0% | −0.97 | 48.7% | 45.5% |

**Inference cost: ~$0.036 per pipeline run.** Total for the 45-cutoff backtest: ~$1.60.

> The market trended down over this period, which makes any short-tolerant system look good. The headline isn't "the system makes money" — it's that the **maximum drawdown is roughly 4× smaller than buy-and-hold** and the system *declines* to take a position on 16 of 45 weeks rather than forcing a bad trade.

---

## What the system does

An **LLM** is a large language model (e.g. GPT). An **agent** is an LLM equipped with tools, structured output schemas, and a defined role — it can read data, run queries, and produce typed outputs instead of just chatting.

The system runs as a five-stage pipeline:

```
Market data  ─→  TA agent     ─┐
                                ├─→  Bull vs Bear debate  ─→  PM  ─→  Human approval
Web news     ─→  News agent   ─┘                                      (approve / reject / edit)
```

1. **Data ingestion** — pulls 5-minute Bitcoin candles from Binance, computes 13+ technical indicators, stores them in SQLite.
2. **TA agent** — runs SQL queries against the price database and writes a structured technical report.
3. **News agent** — searches the web via Perplexity for macro and crypto-specific catalysts.
4. **Bull and Bear agents** — argue opposite sides in a structured debate (independent openings + 2 rebuttal rounds).
5. **PM (Portfolio Manager) agent** — reviews the full debate and decides BULLISH / BEARISH / NEUTRAL with entry, stop-loss, and target.
6. **Human-in-the-loop gate** — the system pauses and asks the user to approve, reject, or edit the trade before recording it.

---

## The five agents

### TA — Technical Analysis
Has a SQL tool over `trading.db`. The agent picks which queries to run — recent candles, RSI extremes, MACD crosses, Bollinger Band width, EMA-50 position, volume ratios, regime classification — across 5-minute, 1-hour, and 4-hour timeframes. The prompt requires it to consider both a bullish and a bearish scenario *before* deciding direction, and to default to NEUTRAL unless one side materially outweighs the other.

### News
A single call to Perplexity's `sonar-pro` with a JSON Schema response format. Produces a structured news report (macro context, sentiment, BTC catalysts, risk events, directional bias, point estimate) in one shot. The harness substitutes hardcoded fake news during backtests so historical replays don't burn API spend on news the system cannot meaningfully act on retrospectively.

### Bull and Bear
Two adversarial agents arguing for opposite directions:

- **Round 1 (independent openings):** each writes its case without seeing the other's argument. Prevents anchoring bias.
- **Rounds 2 and 3 (rebuttals):** both see the full transcript and must introduce *new* arguments — repetition is forbidden by the prompt.
- **Final assessments:** each produces a structured Pydantic output with entry, stop, target, and confidence.

### PM — Portfolio Manager
Reads the full debate transcript plus both sides' final structured assessments, then produces the final decision. Forbidden by prompt from citing data outside the TA and news reports (no funding rates, on-chain metrics, order books) — a guardrail added after a backtest revealed agents fabricating evidence from sources they didn't actually have.

---

## Engineering highlights

These are the things to look at if you care about agentic-AI engineering rigor:

### Walk-forward evaluation harness
`harness.py` replays the entire pipeline across N historical weekly cutoffs and scores each agent's prediction against the realised 1-week price. Each cutoff is wrapped in `try/except` so a single failure doesn't kill the run. Results persist to a separate `eval.db` with per-agent rows (full reasoning preserved) and aggregate run totals — meaning every prompt change can be benchmarked against the same set of historical dates.

### Lookahead-leak prevention
The TA agent's SQL tool accepts a `cutoff` parameter from the harness. It silently rewrites every query, replacing the `btc_ohlcv` table with a filtered subquery so the agent literally cannot see candles past the cutoff timestamp. This eliminates the most common mistake in retrospective ML evaluation — accidentally training or evaluating on the future.

### Per-agent token + cost telemetry
Every LLM call is wrapped in LangChain's `get_openai_callback()`. Tokens, dollar cost, and tool-call counts are captured per agent, summed over all calls inside a role (e.g. bull's opening + 2 rebuttals + final structured call all roll up under one `bull` usage record), and written to `eval.db`. Dollar costs come from LangChain's price table — no hand-rolled pricing.

### Crash-safe pipeline with checkpointing
The orchestration uses **LangGraph** with a SQLite checkpointer. Every node writes its state to `checkpoints.db`. If the process crashes mid-debate or during the human-approval wait, the run resumes by passing the thread ID back in (`uv run main.py --thread <id>`).

### Structured outputs at every interface
Free-text in, typed Pydantic objects out. `TAOutput`, `NewsOutput`, `BullOutput`, `BearOutput`, `PMOutput` define every boundary between agents. Eliminates the "did the LLM say bullish or bearish?" parsing problem at the source.

### Multi-timeframe market data
`data.py` ingests 5-minute Binance candles via CCXT, then aggregates to 1h and 4h tables. Each timeframe carries the same 13+ indicators — RSI-14, MACD/signal/histogram, Bollinger Bands + width + width-rank, EMA-50, ATR-14, RSI slope, distance-from-EMA, volume EMA-20 + volume ratio, swing-high/low markers, regime classification, and forward 7-day return/high/low. The historical-backfill loop is rate-limited against Binance's API limits.

### Single LLM gateway
Every model call routes through OpenRouter via a small `make_llm()` helper. One env var, one integration — swapping `gpt-4.1-mini` for a different model is a string change in `AgentConfig`, not a refactor.

---

## What's intentionally out of scope

- **No live order routing.** No FIX gateway, no broker integration, no real money. This is a *prediction* system; an execution layer is on the v3 roadmap.
- **No position sizing or risk:reward gating.** v2 focuses on direction + level accuracy. Kelly-style sizing and R:R filters are deferred to v3.
- **Single exchange.** Binance only. CCXT supports more, but multi-venue routing isn't wired.
- **Single asset.** BTC/USDT. The architecture is general but data ingestion is hardcoded to one symbol.

---

## Honest caveats

- The 44-week backtest period was **net bearish** — results partly reflect a regime tailwind, not pure alpha.
- 44 weeks is a small sample. The standard error on Sharpe at this n is roughly ±0.30, so the 0.86 figure should be read as "looks good" not "proven edge."
- Direction accuracy is ~52% — close to random. The alpha comes from **filtering bad trades** (the PM goes NEUTRAL on 16 of 45 weeks and avoids exposure) and **asymmetric payoffs**, not from being right more often.
- 5 of 50 cutoffs failed silently in the latest run — a 10% failure rate that needs investigation before this is production-grade.

---

## How to run it

Requires **Python 3.12** and [uv](https://docs.astral.sh/uv/). Set `OPENROUTER_API_KEY` in `.env`.

```bash
uv sync                           # install dependencies
uv run main.py                    # full live run (real news via Perplexity)
uv run main.py --fake-news        # live run with hardcoded news (skip Perplexity)
uv run data.py --backfill 90      # download 90 days of history
uv run pytest                     # tests
uv run pyright                    # type check
```

The pipeline pauses at the human-in-the-loop gate with a `[a]pprove / [r]eject / [e]dit notes:` prompt and resumes on response. Checkpointing means you can crash, lose the terminal, and pick up where you left off via the thread ID printed at start.

---

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 (managed via uv) |
| Agent framework | LangChain + LangGraph |
| LLMs | `gpt-4.1-mini` (TA, Bull, Bear, PM) and `perplexity/sonar-pro` (News), all via OpenRouter |
| Structured outputs | Pydantic |
| Storage | SQLite (`trading.db`, `eval.db`, `checkpoints.db`) |
| Market data | CCXT → Binance |
| Type checker | pyright |
| Testing | pytest |

---

## Repository layout

```
main.py         pipeline orchestrator (live mode + HITL)
graph.py        LangGraph wiring + SQLite checkpointer
data.py         Binance/CCXT ingestion + indicators
ta_agent.py     TA agent with SQL tool (cutoff-aware)
news_agent.py   Perplexity news agent (structured output)
debate.py       Bull/Bear debate + structured final assessments
pm.py           PM agent
models.py       Pydantic schemas for every agent boundary
usage.py        Per-agent token + cost tracking
harness.py      Walk-forward backtest harness
llm.py          Single OpenRouter gateway
docs/           architecture.md, roadmap.md, backlog.md
```

For deeper architecture detail and Mermaid diagrams of the data and debate flows, see [`docs/architecture.md`](docs/architecture.md).
