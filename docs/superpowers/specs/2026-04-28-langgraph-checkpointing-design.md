# LangGraph Checkpointing — Design Spec

**Date:** 2026-04-28
**Status:** Approved (verbal)
**Goal:** Convert the trading pipeline into a LangGraph `StateGraph` so we get
(a) rollback at four points (after TA, news, debate, PM), and (b) a human-in-the-loop
gate before any future trade execution.

## Architecture

```
            ┌─cp1─┐ ┌─cp2─┐ ┌──cp3──┐ ┌─cp4─┐
START ──▶ ta ──▶ news ──▶ debate ──▶ pm ──▶ hitl ──▶ END
                                              │
                                       interrupt() — pauses
                                       for human approve/
                                       reject/edit
```

- **Nodes:** `ta`, `news`, `debate`, `pm`, `hitl` (HITL only when built with `with_hitl=True`)
- **Checkpointer:** `SqliteSaver` from `langgraph-checkpoint-sqlite`, persisted to
  `checkpoints.db` at the project root. New file — keeps market data
  (`trading.db`) and eval results (`eval.db`) cleanly separated from agent run state.
- **Each node-completion = one checkpoint.** Rollback = `graph.get_state_history(...)` →
  pick a checkpoint → resume with `Command(...)`.
- **HITL gate** uses LangGraph's `interrupt()` so the graph pauses with state
  persisted; resuming sends `Command(resume=<decision>)`. Today there is no
  execution node — HITL just records the decision and ends. When execution
  is built later, it slots between `hitl` and `END`.

## State Schema

```python
class TradingState(TypedDict, total=False):
    ta_output: TAOutput | None
    news_output: NewsOutput | None
    debate_messages: list[BaseMessage]   # full transcript
    bull_output: BullOutput | None
    bear_output: BearOutput | None
    pm_output: PMOutput | None
    # Per-agent usage (populated as nodes finish)
    ta_usage: AgentUsage | None
    news_usage: AgentUsage | None        # always None — news isn't tracked
    bull_usage: AgentUsage | None
    bear_usage: AgentUsage | None
    pm_usage: AgentUsage | None
    # HITL outcome
    human_decision: Literal["approve", "reject", "edit"] | None
    human_notes: str | None
```

## Per-Run Configuration

Per-run knobs (model names, prompt overrides, cutoff, fake_news) flow through
LangGraph's native `config["configurable"]` channel — **not** through state.
This is the same channel LangGraph uses for `thread_id`, so it's the canonical
place. Nodes read it via their second arg:

```python
def ta_node(state, config):
    cfg = config["configurable"]
    output, usage = run_ta_agent(
        cutoff=cfg.get("cutoff"),
        model_name=cfg.get("ta_model"),
        system_prompt=cfg.get("ta_prompt"),
    )
    return {"ta_output": output, "ta_usage": usage}
```

Recognized configurable keys: `cutoff`, `fake_news`, `ta_model`, `ta_prompt`,
`bull_model`, `bull_prompt`, `bear_model`, `bear_prompt`, `pm_model`,
`pm_prompt`, `num_rebuttals`. Plus LangGraph's built-in `thread_id`.

## Thread IDs

`<source>_<uuidv7>` — source-prefixed for greppability, UUIDv7 for uniqueness +
time-ordering.

```
live_018f2a3b-7c4d-7e5f-9a8b-1c2d3e4f5a6b
harness_018f2a3b-7c4d-7e5f-9a8b-1c2d3e4f5a6b
```

Human-meaningful fields (`cutoff`, `started_at`, `run_label`) live in checkpoint
metadata via `config["metadata"]` so they're queryable through `.list(filter={...})`.

## Two Compile Modes (one graph definition)

```python
def build_graph(checkpointer, *, with_hitl: bool = True) -> CompiledGraph: ...
```

- `main.py`: `build_graph(saver, with_hitl=True)` — pauses for human approval
- `harness.py`: `build_graph(saver, with_hitl=False)` — never pauses; ends after PM

Both share the same node functions, state shape, and config keys. The HITL
node is *omitted* in harness mode rather than auto-approved — keeps the
deterministic flow obvious.

## File Layout

New / changed files (no nesting, project stays flat):

- **NEW** `graph.py` — `TradingState`, all node functions, `build_graph(...)`,
  HITL `interrupt()` logic. Single file because it stays small (~250 LOC).
- **NEW** `pm.py` — `run_pm()` extracted from `debate.py` so PM can be its own node.
- **MODIFIED** `debate.py` — drops PM. Returns `{messages, bull, bear, usage}` only.
  Renamed top-level function stays `run_debate(...)` (callers in nodes).
- **MODIFIED** `main.py` — replaces direct `run_*_agent()` calls with
  `graph.invoke(...)` + handles HITL resume.
- **MODIFIED** `harness.py` — replaces `_run_one_cutoff` body with `graph.invoke()`
  and reads usage/outputs from final state.
- **MODIFIED** `pyproject.toml` — adds `langgraph-checkpoint-sqlite`.
- **MODIFIED** `.gitignore` — ignores `checkpoints.db`.

## HITL Behavior (today)

Today, with no execution layer:

1. Pipeline runs TA → News → Debate → PM
2. `hitl` node prints PM's decision, calls `interrupt({"pm_output": ...})`
3. `main.py` catches the interrupt, prompts user (terminal `input()`), resumes
   with `Command(resume={"decision": "approve"|"reject"|"edit", "notes": ...})`
4. `hitl` node writes the decision to state, returns, graph ends

When execution is added: replace the post-HITL `END` with an `execute` node
that consumes `human_decision` and acts (or no-ops on reject).

## Out of Scope

- Resumability for partially-completed harness runs (--resume flag)
  → easy to add later with `graph.get_state_history` filtering by metadata
- Full conversion of TA's tool-call loop into nodes (`create_agent` stays as-is)
- Replacing `get_openai_callback` with OpenRouter-native usage tracking
  (orthogonal — flagged for follow-up)
