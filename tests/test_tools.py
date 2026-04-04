"""
tests/test_tools.py — unit tests for the query_market_data tool logic.

Tests target _query_df() directly (the underlying function) so they run
without importing the full app.py (which would trigger data.refresh_db() and
require an OpenAI key on import).
"""

import pandas as pd
import pytest

from app import _query_df


# ---------------------------------------------------------------------------
# Shared test DataFrame fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_df():
    """A small DataFrame that mirrors the btc_ohlcv schema."""
    return pd.DataFrame({
        "timestamp":   ["2026-04-03 10:00:00", "2026-04-03 10:05:00", "2026-04-03 10:10:00"],
        "open":        [80_000.0, 80_100.0, 79_900.0],
        "high":        [80_200.0, 80_300.0, 80_100.0],
        "low":         [79_800.0, 79_900.0, 79_700.0],
        "close":       [80_100.0, 80_200.0, 79_950.0],
        "volume":      [1.2, 0.8, 1.5],
        "rsi_14":      [55.0, 60.0, 28.0],
        "macd":        [10.0, 15.0, -5.0],
        "macd_signal": [8.0,  12.0, -3.0],
        "macd_hist":   [2.0,   3.0, -2.0],
        "bb_upper":    [81_000.0, 81_100.0, 80_900.0],
        "bb_mid":      [80_000.0, 80_050.0, 79_980.0],
        "bb_lower":    [79_000.0, 79_000.0, 79_060.0],
        "ema_50":      [79_900.0, 80_000.0, 80_050.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_valid_query_returns_string(sample_df):
    """A valid query that matches rows returns a non-empty string."""
    result = _query_df(sample_df, "close > 0")
    assert isinstance(result, str)
    assert len(result) > 0


def test_no_match_returns_message(sample_df):
    """A query matching nothing returns the 'No rows match' message."""
    result = _query_df(sample_df, "close > 9_999_999")
    assert result == "No rows match the query."


def test_invalid_query_returns_error_string(sample_df):
    """A malformed query returns an error string and does not raise."""
    result = _query_df(sample_df, "invalid ==")
    assert isinstance(result, str)
    assert "error" in result.lower() or "query" in result.lower()


def test_returns_all_matching_rows(sample_df):
    """Result contains all rows matched by the query, not a capped subset."""
    # Query matches 2 of the 3 rows (rsi_14 > 50)
    matching_count = len(sample_df.query("rsi_14 > 50", engine="python"))
    result = _query_df(sample_df, "rsi_14 > 50")
    # Count data lines (non-header lines that contain numeric data)
    lines = [ln for ln in result.strip().splitlines() if ln.strip()]
    # Subtract 1 for the header row
    data_lines = len(lines) - 1
    assert data_lines == matching_count


def test_oversold_query(sample_df):
    """RSI < 30 query correctly identifies the oversold row."""
    result = _query_df(sample_df, "rsi_14 < 30")
    assert "28.0" in result  # the oversold RSI value is in the output


def test_timestamp_filter_query(sample_df):
    """Timestamp string comparisons work as expected."""
    result = _query_df(sample_df, "timestamp >= '2026-04-03 10:05:00'")
    assert "10:05:00" in result
    assert "10:00:00" not in result
