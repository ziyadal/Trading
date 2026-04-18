# Backlog

Work items in priority order. Do one at a time, top to bottom.
Run `uv run pytest && uv run pyright` after each session to check nothing is broken.

## Phase 1: Simplify (understand your own code)

- [ ] **Read through every file end-to-end following the architecture doc.** No code changes — just read and make sure the doc matches your understanding. Update the doc if anything is unclear.

## Phase 2: Track (measure before you optimize)

- [ ] **Run a baseline harness test** — Run the harness on 4-5 cutoff dates with the current prompts. Save the results with label `baseline`. This is your "before" measurement.
- [ ] **Add basic pytest tests** — Not for prediction accuracy (that's the harness), but for code correctness:
  - `data.py`: indicators compute without errors on sample data
  - `models.py`: Pydantic models accept valid data, reject bad data
  - `eval.py`: logging writes and reads back correctly
  - `harness.py`: cutoff generation produces correct dates
- [ ] **Verify the end-of-session check works** — Run `uv run pytest && uv run pyright` and confirm both pass clean. This is your safety net going forward.

## Phase 3: Improve (now you can measure impact)

- [ ] **TA agent prompt tuning** — Change the prompt, run the harness on the same dates as baseline, compare. Label each run (e.g. `ta_v2_shorter_prompt`).
- [ ] **Test different models** — Try `gpt-4.1` vs `gpt-4.1-mini` for the TA agent. The harness `AgentConfig` already supports this.
- [ ] **Debate prompt tuning** — Same approach: change prompts, run harness in `full` mode, compare to baseline.
- [ ] **Add more indicators or data** — Only if the harness results suggest the TA agent is missing something specific.

## Rules

1. Pick ONE item per session.
2. If you think of something new, add it to this file — don't start it.
3. Don't skip to Phase 3 before Phase 2 is done.
