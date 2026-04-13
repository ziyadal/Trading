"""
main.py — BTC/USDT trading pipeline orchestrator.

Usage:
    uv run main.py                          # live run
    uv run main.py "my_label"               # live run with custom label
    uv run main.py "backtest_v1" "2026-04-01 12:00:00"  # backtest with cutoff
"""

import sys

from dotenv import load_dotenv

from data import refresh_db
from debate import run_debate
from eval import init_eval_db, log_run, print_comparison
from news_agent import run_news_agent
from ta_agent import run_ta_agent

load_dotenv(override=True)


def run_pipeline(
    cutoff: str | None = None,
    run_label: str = "live",
    fake_news: bool = False,
) -> None:
    """Run the full trading pipeline.

    Args:
        cutoff: Optional timestamp for backtesting (e.g. '2026-04-01 12:00:00').
                When set, the TA agent only sees data up to this time and
                market data is NOT refreshed.
        run_label: Label for this run in the eval log (e.g. 'baseline', 'v2_longer_pm').
        fake_news: If True, use hardcoded news data instead of calling Perplexity.
    """
    # 1. Fresh market data (skip in backtest mode — data already exists)
    if not cutoff:
        print("Refreshing market data...")
        refresh_db()

    # 2. Analysis agents
    ta_output = run_ta_agent(cutoff=cutoff)

    print(f"\n{'=' * 60}")
    print("TA STRUCTURED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Direction:   {ta_output.direction}")
    print(f"  Price:       ${ta_output.current_price:,.0f}")
    print(f"  Target:      ${ta_output.target_low:,.0f} – ${ta_output.target_high:,.0f}")
    print(f"  Support:     ${ta_output.key_support:,.0f}")
    print(f"  Resistance:  ${ta_output.key_resistance:,.0f}")
    print(f"  Confidence:  {ta_output.confidence:.0%}")

    news_output = run_news_agent(fake=fake_news)

    print(f"\n{'=' * 60}")
    print("NEWS STRUCTURED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Direction:      {news_output.direction}")
    print(f"  Price Target:   ${news_output.price_prediction:,.0f}")
    print(f"  Key Catalyst:   {news_output.key_catalyst}")
    print(f"  Confidence:     {news_output.confidence:.0%}")

    # 3. Debate + PM decision
    print("\n" + "=" * 60)
    print("BTC/USDT Bull vs Bear Debate")
    print("=" * 60)
    debate_result = run_debate(ta_output.report, news_output.report)

    # 4. Log all predictions
    init_eval_db()
    log_run(
        run_label=run_label,
        ta=ta_output,
        news=news_output,
        bull=debate_result["bull"],
        bear=debate_result["bear"],
        pm=debate_result["pm"],
    )

    # 5. Show historical comparison
    print_comparison()

    # 6. Print final decision
    pm = debate_result["pm"]
    print(f"\n{'=' * 60}")
    print(f"FINAL DECISION: {pm.decision}")
    if pm.entry:
        print(f"  Entry:      ${pm.entry:,.0f}")
        print(f"  Stop Loss:  ${pm.stop_loss:,.0f}")
        print(f"  Target:     ${pm.target:,.0f}")
        print(f"  Size:       {pm.position_size}%")
    print(f"  Confidence: {pm.confidence:.0%}")
    print(f"  Winner:     {pm.winning_side}")
    print(f"  Reason:     {pm.key_reason}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    use_fake_news = "--fake-news" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--fake-news"]
    label = args[0] if len(args) > 0 else "live"
    cutoff_arg = args[1] if len(args) > 1 else None
    run_pipeline(cutoff=cutoff_arg, run_label=label, fake_news=use_fake_news)
