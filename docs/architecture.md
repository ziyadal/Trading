# Architecture

## System Diagram

```mermaid
flowchart TD
    subgraph Pipeline["main.py — Pipeline Orchestrator"]
        direction TB
        DATA["data.py\nMarket Data Ingestion"]
        TA["ta_agent.py\nTechnical Analysis Agent"]
        NEWS["news_agent.py\nNews Research Agent"]
        DEBATE["debate.py\nBull vs Bear Debate"]
    end

    BINANCE["Binance API"]
    PERPLEXITY["Perplexity API\n(structured output)"]
    OPENAI["OpenAI API"]
    TRADING_DB[("trading.db\nOHLCV + Indicators")]

    BINANCE -- "2 days of 5m candles" --> DATA
    DATA -- "OHLCV + RSI, MACD,\nBollinger, EMA-50" --> TRADING_DB

    TRADING_DB -- "SQL queries\n(via tool)" --> TA
    TA -- "TA report" --> DEBATE
    OPENAI --> TA

    PERPLEXITY -- "NewsOutput\n(JSON Schema)" --> NEWS
    NEWS -- "News report" --> DEBATE

    OPENAI --> DEBATE
```



## Harness Diagram

```mermaid
flowchart TD
    subgraph Harness["harness.py — Evaluation Harness"]
        direction TB
        CUTOFFS["Generate weekly\ncutoff timestamps"]
        RUN["Run agents at\neach cutoff"]
        SCORE["Score predictions\nvs actual prices"]
        LOG["Log results\nto eval.db"]
        SUMMARY["Print summary\n(accuracy, calibration)"]
        CUTOFFS --> RUN --> SCORE --> LOG --> SUMMARY
    end

    TRADING_DB[("trading.db")]
    EVAL_DB[("eval.db\nHarness Results")]
    TA["ta_agent.py\n(with cutoff filter)"]
    DEBATE["debate.py\n(with fake news)"]

    RUN -. "mode=ta" .-> TA
    RUN -. "mode=full" .-> DEBATE
    TRADING_DB -- "price lookups\nfor scoring" --> SCORE
    LOG -- "scored results" --> EVAL_DB
```



## Components

### main.py — Pipeline Orchestrator

Entry point that runs the full live pipeline in sequence: refresh market data from Binance, run TA agent, run news agent, run debate, print the final decision. Always fetches fresh data.

**In:** Optional `--fake-news` flag to skip Perplexity --> passes an empty  news report  
**Out:** Printed results (final PM decision with entry/stop/target)

### data.py — Market Data Ingestion

Fetches BTC/USDT 5-minute candles from Binance via CCXT, computes four technical indicators (RSI-14, MACD, Bollinger Bands, EMA-50), and upserts everything into SQLite. Also supports a `--backfill` mode to pull up to 90+ days of history for backtesting.

**In:** Nothing (pulls from Binance directly)
**Out:** `trading.db` with table `btc_ohlcv` (OHLCV + indicators, keyed by timestamp)

### ta_agent.py — Technical Analysis Agent

An agent (LangChain `create_agent`) with a SQL tool that queries `trading.db` to analyze price action. It decides which queries to run — recent candles, RSI extremes, MACD trend, Bollinger Band breakouts, EMA-50 position, volume — then produces a report and structured prediction. Uses `response_format=TAOutput` to get structured output directly from the agent.

When given a cutoff timestamp (used by the harness for backtesting), the SQL tool silently replaces the table name with a filtered subquery so the agent only sees data up to that time.

**In:** Optional cutoff timestamp, optional prompt/model overrides
**Out:** `TAOutput` (report, direction, price target range, support/resistance, confidence)

### news_agent.py — News Research Agent

Single-phase agent. Perplexity's `sonar-pro` model researches the latest BTC news and returns a structured `NewsOutput` directly via their JSON Schema `response_format` — no separate OpenAI call. Has a `fake` mode that skips the Perplexity call and returns a hardcoded `NewsOutput` (used by the harness to avoid expensive API calls during backtesting).

**In:** Optional `fake=True` flag
**Out:** `NewsOutput` (report, direction, price prediction, confidence, key catalyst)

### debate.py — Bull vs Bear Debate

Three agents argue over the trade. The flow is:

1. Bull and Bear opening arguments (independent — each sees only the market data, not the other's opening)
2. 2 rounds of rebuttals (both see full transcript, each must introduce new arguments)
3. Bull structured final assessment (`BullOutput`)
4. Bear structured final assessment (`BearOutput`)
5. PM reviews the full transcript + both sides' numbers, renders final decision (`PMOutput`)

Opening arguments are independent to prevent anchoring bias. After openings, both sides share a message transcript so each rebuttal directly responds to the other side. Opening/rebuttal rounds use free-text chat; final assessments use `.with_structured_output()`. The PM enforces a hard rule: risk:reward must be >= 1:1 or the decision is NEUTRAL.

**In:** TA report text, news report text, optional prompt/model overrides
**Out:** Dict with `bull` (BullOutput), `bear` (BearOutput), `pm` (PMOutput), `messages` (full transcript)

### models.py — Pydantic Output Models

Defines the structured output schemas for every agent: `TAOutput`, `NewsOutput`, `BullOutput`, `BearOutput`, `PMOutput`. These are Pydantic models used with LangChain's structured output to get reliable, typed data from the LLMs. Each model has a free-text `report` field plus numeric prediction fields.

### harness.py — Evaluation Harness

Runs the pipeline across multiple historical dates to measure prediction accuracy. Generates a list of weekly cutoff timestamps, runs the TA agent (and optionally the full debate) at each one, then looks up what the price actually did 1 week later to score the prediction. Also handles logging scored results to `eval.db`. Two modes: `"ta"` (TA agent only, fast and cheap) and `"full"` (TA + debate + PM, uses fake news). Supports `AgentConfig` overrides so you can test different prompts or models against the same dates.

**In:** Start date, number of weeks, mode, optional agent config overrides
**Out:** List of `TestResult` objects + results saved to eval.db + printed summary table

## Data Flow

```mermaid
flowchart LR
    subgraph Ingestion
        B[Binance] --> D[data.py]
        D -- "OHLCV + indicators" --> TDB[(trading.db)]
    end

    subgraph Analysis
        TDB -- "SQL queries" --> TA[TA Agent]
        P[Perplexity] -- "structured output" --> NA[News Agent]
    end

    subgraph Debate
        TA -- "TA report" --> BULL[Bull Agent]
        TA -- "TA report" --> BEAR[Bear Agent]
        NA -- "News report" --> BULL
        NA -- "News report" --> BEAR
        BULL -. "independent openings\n(don't see each other)" .-> BULL
        BEAR -. "independent openings\n(don't see each other)" .-> BEAR
        BULL <-- "shared transcript\n(2 rebuttal rounds)" --> BEAR
        BULL -- "BullOutput" --> PM[PM Agent]
        BEAR -- "BearOutput" --> PM
        BULL -. "full transcript" .-> PM
    end

    PM -- "PMOutput\nFinal Decision" --> OUT[Print Results]
```



## Running the Pipeline

### Live Mode (main.py)

```bash
uv run main.py                    # full live run
uv run main.py --fake-news        # skip Perplexity, use hardcoded news
```

1. Refreshes market data from Binance
2. TA agent analyzes current data
3. News agent calls Perplexity for live research (or uses fake data)
4. Debate runs with real data
5. Prints the final PM decision

### Evaluation (harness.py)

```python
from harness import run_harness
run_harness(start_date="2026-02-01 12:00:00", num_weeks=5, mode="ta", run_label="baseline")
```

The harness runs the TA agent (or full pipeline) at multiple past cutoff dates and scores predictions against what actually happened. This is how you measure whether changes to prompts or models are improving accuracy.

## Databases


| Database     | Table             | Purpose                                                          |
| ------------ | ----------------- | ---------------------------------------------------------------- |
| `trading.db` | `btc_ohlcv`       | 5m OHLCV candles + technical indicators. Primary key: timestamp. |
| `eval.db`    | `harness_results` | Scored predictions from harness evaluation runs.                 |


