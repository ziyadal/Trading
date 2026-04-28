# Roadmap

Strategic roadmap for the BTC trading multi-agent project. The goal is a **portfolio-ready demonstration of agentic AI engineering competence**, suitable for job applications in the agentic AI space.

For tactical task tracking, see [`backlog.md`](./backlog.md).

---

## Where we are (2026-04-24)

**Engineering substance is strong:**

- Multi-agent architecture (TA → News → Bull/Bear/PM debate)
- Backtesting harness with **lookahead-bias prevention** (cutoff-filtered SQL subqueries in `ta_agent.py`)
- Structured Pydantic outputs for every agent (`models.py`)
- Per-agent token / cost / tool-call tracking in `eval.db`
- Grounding rules in prompts to prevent hallucinated evidence (no fabricated funding rates, on-chain, etc.)
- Anti-anchoring debate design (independent openings; mandatory new arguments per rebuttal)
- Multi-timeframe analogue tool (historical base rates for current regime/RSI/BB-width)

**Presentation layer is empty:**

- Empty `README.md`
- Zero tests despite `pytest` being a dependency
- No CI / GitHub Actions
- Backtest results exist in `eval.db` but aren't published anywhere a recruiter can see
- No baseline comparison (is the agent system better than "always bullish"?)

**Status: ~60% to portfolio-ready. Substance > storefront.**

---

## Active work

1. **Model bake-off experiment** — 3 mini-tier models (`gpt-4.1-mini`, `claude-3-5-haiku`, `gemini-2.0-flash`) across providers, ~12 weekly cutoffs. Generates the data for the first LinkedIn post.
2. **LinkedIn post** — once bake-off is done, write up the design + numbers.
3. **README rewrite** — same data feeds into a real README so the linked GitHub isn't empty.

---

## Tier 1 — Portfolio-submittable (~1 week)

The minimum bar to make the project credible to a recruiter who clones the repo.

1. **Real README** — what/why, system diagram (already exists in `architecture.md`), how to run, **headline backtest results**, the design decisions worth talking about (grounding rule, independent openings, lookahead prevention, base-rate analogues).
2. **Basic pytest coverage** — indicator math (`data.py`), Pydantic validation (`models.py`), cutoff generation (`harness.py`). ~10-15 tests is plenty.
3. **GitHub Actions CI** — runs `pytest && pyright` on every push. Visible badge in README.
4. **Baseline comparison** — implement "always bullish" and "always trend continuation" baselines in the harness, so the agent's accuracy has something to be measured against. Honest if the agent doesn't beat them yet.
5. **Repo hygiene** — `.gitignore` the log files and `*.db` artifacts, fix `pyproject.toml` description, commit working state, delete empty `test.ipynb` and `docs/thingsnotsureabout.md`.

---

## Tier 2 — Stands out (~1 more week)

Pushes the project from "complete" to "noteworthy" for agentic AI roles specifically.

6. **LangSmith tracing wired in** — one debate trace screenshot in the README. `langsmith` is already in dependencies.
7. **Calibration plot saved as PNG** — embedded in README. The data already exists in `eval.db` (`harness.py:633-651`); just needs a small matplotlib script.
8. **`experiments/` directory** — one short markdown per prompt iteration showing before/after numbers. Demonstrates iterating on data, not vibes.
9. **Real news in backtests** — backfill Perplexity once per cutoff date, cache to JSON, so the harness can evaluate the full pipeline including news (currently uses `FAKE_RESEARCH = "empty"`).
10. **Honest "what didn't work" section** — README essay about a design choice that didn't pan out and why. Honesty is rare and senior-coded.

---

## Tier 3 — Senior signal (~2-3 more weeks)

Differentiates from "I built a cool agent project" to "I think like a senior engineer about agent systems".

11. **Multi-asset support** — add ETH alongside BTC, per-asset breakdown in harness summary. Proves generality.
12. **v3 execution stub** — paper trading layer that takes PM decisions and "places" orders, tracks PnL against the predicted target/stop. Doesn't have to be live; printing/logging is enough to show the design intent.
13. **Blog post / detailed essay** — walk through one specific design decision (e.g. "why independent openings prevented anchoring — measured before/after"). Writing about technical decisions is itself a senior signal.

---

## Out of scope (do not start)

- Real money trading
- Order routing / FIX protocol implementation
- More technical indicators (current set is sufficient until evals say otherwise)
- More agents in the debate (already over-engineered for v2)
- Frontend / UI / dashboard

---

## Working principles

1. Pick ONE item from the active list per session.
2. If you think of something new, add it to `backlog.md` — don't start it.
3. Don't skip Tier 2 before Tier 1 is done.
4. Run `uv run pytest && uv run pyright` after each session.
