"""
main.py — BTC/USDT trading pipeline orchestrator.

Usage:
    uv run main.py                  # live run
    uv run main.py --fake-news      # live run with fake news (skip Perplexity)
"""

import sys

from dotenv import load_dotenv

from data import refresh_db
from debate import run_debate
from news_agent import run_news_agent
from ta_agent import run_ta_agent

load_dotenv(override=True)


def run_pipeline(fake_news: bool = False) -> None:
    """Run the full trading pipeline (live data only).

    Args:
        fake_news: If True, use hardcoded news data instead of calling Perplexity.
    """
    # 1. Fresh market data
    print("Refreshing market data...")
    refresh_db()

    # 2. Analysis agents
    ta_output = run_ta_agent()

    print(f"\n{'=' * 60}")
    print("TA STRUCTURED OUTPUT")
    print(f"{'=' * 60}")
    print(f"  Direction:   {ta_output.direction}")
    print(f"  Price:       ${ta_output.current_price:,.0f}")
    print(f"  Target:      ${ta_output.target_low:,.0f} – ${ta_output.target_high:,.0f}")
    print(f"  Support:     ${ta_output.key_support:,.0f}")
    print(f"  Resistance:  ${ta_output.key_resistance:,.0f}")
    print(f"  Confidence:  {ta_output.confidence:.0%}")

    print(f"\n{'=' * 60}")
    print(f"TA REPORT:  {ta_output.report}")

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

    # 4. Print final decision
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
    run_pipeline(fake_news=use_fake_news)
