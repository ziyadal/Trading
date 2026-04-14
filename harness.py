"""
harness.py — Evaluation harness for the BTC trading pipeline.

Runs agents across historical cutoffs and scores predictions against actual
price movement. Two modes:
  - "ta":   TA agent only (fast, cheap)
  - "full": TA + debate + PM with fake news (no news agent)

Usage (notebook or script):

    from harness import run_harness, AgentConfig

    # TA only, 5 weekly tests starting Feb 1
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="ta",
        run_label="ta_v1",
    )

    # Test a new TA prompt
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="ta",
        ta=AgentConfig(system_prompt=MY_NEW_PROMPT),
        run_label="ta_v2",
    )

    # Full pipeline (TA + debate + PM, fake news)
    results = run_harness(
        start_date="2026-02-01 12:00:00",
        num_weeks=5,
        mode="full",
        run_label="full_v1",
    )

    # Custom date range with 4 evenly-spaced tests
    results = run_harness(
        start_date="2026-01-15 12:00:00",
        end_date="2026-04-01 12:00:00",
        num_weeks=4,
        mode="ta",
        run_label="spread_test",
    )
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)

from ta_agent import run_ta_agent
from debate import run_debate
from eval import log_harness_results
from news_agent import FAKE_RESEARCH


# ---------------------------------------------------------------------------
# Config / result types
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Configuration overrides for an agent. None = use agent's default."""

    system_prompt: str | None = None
    user_prompt: str | None = None
    model: str | None = None


@dataclass
class TestResult:
    """Result from scoring a single agent prediction at one cutoff."""

    week: int
    cutoff: str
    agent: str
    direction: str
    target: float | None
    confidence: float
    price_at_cutoff: float | None = None
    actual_price: float | None = None
    direction_correct: bool | None = None
    target_error: float | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_cutoffs(
    start_date: str, num_weeks: int, end_date: str | None = None
) -> list[str]:
    """Generate cutoff timestamps.

    If end_date is provided, cutoffs are evenly spaced between start and end.
    Otherwise, cutoffs are 1 week apart starting from start_date.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d %H:%M:%S")

    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d %H:%M:%S")
        if num_weeks <= 1:
            return [start_date]
        step = (end - start) / (num_weeks - 1)
        return [
            (start + step * i).strftime("%Y-%m-%d %H:%M:%S")
            for i in range(num_weeks)
        ]

    return [
        (start + timedelta(weeks=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(num_weeks)
    ]


def _get_price(timestamp: str, direction: str = "before") -> float | None:
    """Get the closest BTC close price at/before or at/after a timestamp."""
    conn = sqlite3.connect("trading.db")
    try:
        if direction == "before":
            row = conn.execute(
                "SELECT close FROM btc_ohlcv WHERE timestamp <= ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (timestamp,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT close FROM btc_ohlcv WHERE timestamp >= ? "
                "ORDER BY timestamp ASC LIMIT 1",
                (timestamp,),
            ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _score_direction(
    predicted: str, price_before: float, price_after: float
) -> bool:
    """Check if the predicted direction matches actual price movement."""
    went_up = price_after > price_before
    if predicted in ("BULLISH", "BUY", "UP"):
        return went_up
    elif predicted in ("BEARISH", "SELL", "DOWN"):
        return not went_up
    else:  # NEUTRAL
        pct_change = abs(price_after - price_before) / price_before
        return pct_change < 0.02


# ---------------------------------------------------------------------------
# Main harness
# ---------------------------------------------------------------------------

def run_harness(
    start_date: str,
    num_weeks: int = 4,
    end_date: str | None = None,
    mode: str = "ta",
    ta: AgentConfig | None = None,
    bull: AgentConfig | None = None,
    bear: AgentConfig | None = None,
    pm: AgentConfig | None = None,
    num_rebuttals: int = 2,
    run_label: str = "eval",
) -> list[TestResult]:
    """Run the evaluation harness across weekly cutoffs.

    Args:
        start_date: First cutoff timestamp (e.g. "2026-02-01 12:00:00").
        num_weeks:  Number of tests to run.
        end_date:   Optional end date — if provided, cutoffs are evenly spaced
                    between start_date and end_date instead of weekly.
        mode:       "ta" for TA agent only, "full" for TA + debate + PM.
        ta:         Config overrides for the TA agent.
        bull:       Config overrides for the bull debate agent.
        bear:       Config overrides for the bear debate agent.
        pm:         Config overrides for the PM agent.
        num_rebuttals: Debate rebuttal rounds (full mode only).
        run_label:  Label for this evaluation run.

    Returns:
        List of TestResult objects with scores for every agent at each cutoff.
    """
    ta = ta or AgentConfig()
    bull = bull or AgentConfig()
    bear = bear or AgentConfig()
    pm = pm or AgentConfig()

    cutoffs = generate_cutoffs(start_date, num_weeks, end_date)
    results: list[TestResult] = []

    print(f"\n{'=' * 60}")
    print("EVALUATION HARNESS")
    print(f"  Mode:     {mode}")
    print(f"  Tests:    {num_weeks}")
    print(f"  Range:    {cutoffs[0]}  ->  {cutoffs[-1]}")
    print(f"  Label:    {run_label}")
    print(f"{'=' * 60}")

    for i, cutoff in enumerate(cutoffs):
        week = i + 1

        print(f"\n{'-' * 60}")
        print(f"TEST {week}/{num_weeks} | Cutoff: {cutoff}")
        print(f"{'-' * 60}")

        # --- Scoring setup ---
        future_ts = (
            datetime.strptime(cutoff, "%Y-%m-%d %H:%M:%S") + timedelta(weeks=1)
        ).strftime("%Y-%m-%d %H:%M:%S")
        price_at_cutoff = _get_price(cutoff, "before")
        actual_price = _get_price(future_ts, "after")
        can_score = price_at_cutoff is not None and actual_price is not None

        # --- TA Agent ---
        ta_output = run_ta_agent(
            cutoff=cutoff,
            system_prompt=ta.system_prompt,
            user_prompt=ta.user_prompt,
            model_name=ta.model,
        )

        ta_correct = None
        ta_error = None
        if can_score:
            ta_correct = _score_direction(
                ta_output.direction, price_at_cutoff, actual_price
            )
            ta_error = abs(ta_output.target_high - actual_price)

        results.append(
            TestResult(
                week=week,
                cutoff=cutoff,
                agent="ta",
                direction=ta_output.direction,
                target=ta_output.target_high,
                confidence=ta_output.confidence,
                price_at_cutoff=price_at_cutoff,
                actual_price=actual_price,
                direction_correct=ta_correct,
                target_error=ta_error,
            )
        )

        _print_agent_line(
            "TA", ta_output.direction, ta_output.target_high,
            ta_output.confidence, can_score, ta_correct, actual_price, ta_error,
        )

        # --- Full mode: Debate + PM ---
        if mode == "full":
            fake_news = FAKE_RESEARCH.format(date=cutoff[:10])

            debate_result = run_debate(
                ta_report=ta_output.report,
                news_report=fake_news,
                bull_prompt=bull.system_prompt,
                bear_prompt=bear.system_prompt,
                pm_prompt=pm.system_prompt,
                bull_model=bull.model,
                bear_model=bear.model,
                pm_model=pm.model,
                num_rebuttals=num_rebuttals,
            )

            pm_output = debate_result["pm"]
            pm_correct = None
            pm_error = None
            if can_score:
                pm_correct = _score_direction(
                    pm_output.decision, price_at_cutoff, actual_price
                )
                if pm_output.target:
                    pm_error = abs(pm_output.target - actual_price)

            results.append(
                TestResult(
                    week=week,
                    cutoff=cutoff,
                    agent="pm",
                    direction=pm_output.decision,
                    target=pm_output.target,
                    confidence=pm_output.confidence,
                    price_at_cutoff=price_at_cutoff,
                    actual_price=actual_price,
                    direction_correct=pm_correct,
                    target_error=pm_error,
                )
            )

            _print_agent_line(
                "PM", pm_output.decision, pm_output.target,
                pm_output.confidence, can_score, pm_correct, actual_price,
                pm_error,
            )

    # --- Save to database ---
    log_harness_results(results, run_label)

    # --- Summary ---
    print_summary(results, run_label)
    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_agent_line(
    agent: str,
    direction: str,
    target: float | None,
    confidence: float,
    can_score: bool,
    correct: bool | None,
    actual_price: float | None,
    error: float | None,
) -> None:
    """Print one agent's result for a single test."""
    target_str = f"${target:,.0f}" if target else "N/A"
    print(f"  {agent}: {direction} | target={target_str} | conf={confidence:.0%}")
    if can_score and correct is not None:
        mark = "YES" if correct else "NO"
        error_str = f" | error=${error:,.0f}" if error is not None else ""
        print(f"       correct={mark} | actual=${actual_price:,.0f}{error_str}")
    else:
        print("       (no future data to score)")


def print_summary(results: list[TestResult], run_label: str = "") -> None:
    """Print aggregate evaluation summary."""
    if not results:
        print("\nNo results to summarise.")
        return

    print(f"\n{'=' * 60}")
    print(f"SUMMARY — {run_label}" if run_label else "SUMMARY")
    print(f"{'=' * 60}")

    agents = sorted(set(r.agent for r in results))

    header = f"{'Agent':<8} {'Tests':<7} {'Accuracy':<14} {'Avg Error':<12} {'Avg Conf':<10}"
    print(f"\n{header}")
    print("-" * len(header))

    for agent in agents:
        agent_results = [r for r in results if r.agent == agent]
        total = len(agent_results)

        scored = [r for r in agent_results if r.direction_correct is not None]
        if scored:
            correct = sum(1 for r in scored if r.direction_correct)
            accuracy = f"{correct}/{len(scored)} ({correct / len(scored):.0%})"
        else:
            accuracy = "no data"

        errors = [
            r.target_error for r in agent_results if r.target_error is not None
        ]
        avg_error = f"${sum(errors) / len(errors):,.0f}" if errors else "N/A"

        avg_conf = sum(r.confidence for r in agent_results) / total

        print(f"{agent:<8} {total:<7} {accuracy:<14} {avg_error:<12} {avg_conf:.0%}")

    # --- Calibration (needs enough data) ---
    scored_all = [r for r in results if r.direction_correct is not None]
    if len(scored_all) >= 4:
        print("\nCalibration:")
        buckets: dict[float, dict[str, int]] = {}
        for r in scored_all:
            bucket = round(r.confidence * 5) / 5  # nearest 0.2
            if bucket not in buckets:
                buckets[bucket] = {"total": 0, "correct": 0}
            buckets[bucket]["total"] += 1
            if r.direction_correct:
                buckets[bucket]["correct"] += 1

        for conf in sorted(buckets):
            b = buckets[conf]
            actual = b["correct"] / b["total"]
            print(
                f"  Stated {conf:.0%} conf -> Actual {actual:.0%}  (n={b['total']})"
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Quick smoke test: 2 weeks of TA-only evaluation
    start = sys.argv[1] if len(sys.argv) > 1 else "2026-04-06 12:00:00"
    weeks = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    label = sys.argv[3] if len(sys.argv) > 3 else "cli_test"

    run_harness(
        start_date=start,
        num_weeks=weeks,
        mode="ta",
        run_label=label,
    )
