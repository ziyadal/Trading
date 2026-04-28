# Backlog

Work items in priority order. Do one at a time, top to bottom.
Run `uv run pytest && uv run pyright` after each session to check nothing is broken.

## Phase 1: Simplify (understand your own code)

- [ ] **Read through every file end-to-end following the architecture doc.** No code changes — just read and make sure the doc matches your understanding. Update the doc if anything is unclear.

## Phase 2: Track (measure before you optimize)

- [x] **Run a baseline harness test** — Done: 9-week full-pipeline run (2026-02-14 → 2026-04-11) logged to `eval.db` with per-agent reasoning + usage.
- [ ] **Add basic pytest tests** — Not for prediction accuracy (that's the harness), but for code correctness:
  - `data.py`: indicators compute without errors on sample data
  - `models.py`: Pydantic models accept valid data, reject bad data
  - `eval.py`: logging writes and reads back correctly
  - `harness.py`: cutoff generation produces correct dates
- [ ] **Verify the end-of-session check works** — Run `uv run pytest && uv run pyright` and confirm both pass clean. This is your safety net going forward.

## Phase 3: Improve (now you can measure impact)

- [x] **TA agent prompt tuning** — v2 prompt requires explicit bull+bear scenarios and defaults NEUTRAL. Grounding rule added to forbid non-OHLCV references. Verified against 5-week re-run.
- [ ] **Test different models** — Try `gpt-4.1` vs `gpt-4.1-mini` for the TA agent. The harness `AgentConfig` already supports this.
- [x] **Debate prompt tuning** — R:R now computed in code and injected as a fact into the PM prompt (removed arithmetic errors). Grounding rule added to bull, bear, and PM prompts to forbid fabricated evidence. PM accuracy rose from 1/5 → 3/5 on the re-run.
- [ ] **Add more indicators or data** — Only if the harness results suggest the TA agent is missing something specific.

## Phase 4: Follow-ups from the 9-week backtest

- [ ] **Score targets as ranges, not points** — `TAOutput.target` is a single price; the prompt asks for a range. Capture both ends and score "did the actual close land inside the predicted range" alongside direction accuracy.
- [ ] **Calibration feedback loop** — Feed the model its past accuracy/confidence calibration so "High confidence" actually means high hit-rate.
- [ ] **Higher-timeframe indicators** — Add 1h or 4h candles so the TA agent sees macro trend context, not just 5m noise.
- [ ] **Revisit over-conservative NEUTRAL** — v2 went NEUTRAL on week 1 despite a clear -3.3% move. Watch whether the NEUTRAL default is too strict once more data accumulates.

## Phase 5: Execution (v3 — only after prediction accuracy is solid)

v2 focuses purely on prediction accuracy. All execution/risk-gating concerns live here and should NOT bleed back into the debate prompts or debate code (they skew PM reasoning away from "is the call right?" toward "is the trade nice?").

- [ ] **R:R gating** — Compute bull/bear risk:reward from entry/stop/target in the execution layer, reject trades below a configurable threshold (was previously in `debate.py` as `_compute_rr` — removed 2026-04-21).
- [ ] **Position sizing** — Derive size from the 2% max-risk rule and the distance between entry and stop.
- [ ] **FIX protocol order routing** — Place the actual orders via FIX to whichever venue we use for BTC/USDT.
- [ ] **Human-in-the-loop approval** — No order goes out without an explicit human confirmation step (CLI prompt or UI button) showing the PM decision, levels, size, and R:R.
- [ ] **Fills + PnL tracking** — Capture fill prices and track realised PnL against the PM's predicted target/stop so v3 results feed back into v2 calibration.

## Rules

1. Pick ONE item per session.
2. If you think of something new, add it to this file — don't start it.
3. Don't skip to Phase 3 before Phase 2 is done.
