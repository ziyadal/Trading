"""
tests/test_data.py — unit tests for data.py

CCXT network calls are mocked with synthetic OHLCV data so tests run offline.
All tests use a temporary SQLite database via pytest's tmp_path fixture.
"""

import re
import sqlite3
import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import data

# ---------------------------------------------------------------------------
# Synthetic OHLCV fixture (600 rows — enough for all indicators)
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 600):
    """Generate n rows of fake 5m OHLCV data starting 2 days ago."""
    base_ms = int(time.time() * 1000) - n * 5 * 60 * 1000
    price = 80_000.0
    rows = []
    for i in range(n):
        ts = base_ms + i * 5 * 60 * 1000
        rows.append([ts, price - 50, price + 100, price - 100, price + 25, 1.5])
        price += (i % 7 - 3) * 10  # gentle drift
    return rows


@pytest.fixture()
def db(tmp_path):
    """Return path to a fresh temporary SQLite database."""
    return str(tmp_path / "test_trading.db")


@pytest.fixture()
def populated_db(db):
    """Call refresh_db() once with mocked CCXT and return the db path."""
    mock_exchange = MagicMock()
    mock_exchange.milliseconds.return_value = int(time.time() * 1000)
    mock_exchange.fetch_ohlcv.return_value = _make_ohlcv()

    with patch("data.ccxt.binance", return_value=mock_exchange):
        data.refresh_db(db_path=db)

    return db


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_refresh_db_creates_table(populated_db):
    """After refresh_db(), the btc_ohlcv table exists and has rows."""
    conn = sqlite3.connect(populated_db)
    cur = conn.execute("SELECT COUNT(*) FROM btc_ohlcv")
    count = cur.fetchone()[0]
    conn.close()
    assert count > 0


def test_load_df_returns_dataframe(populated_db):
    """load_df() returns a pandas DataFrame."""
    df = data.load_df(populated_db)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_df_has_correct_columns(populated_db):
    """All 14 expected columns are present."""
    df = data.load_df(populated_db)
    assert set(data.COLUMNS).issubset(set(df.columns))


def test_indicators_not_all_null(populated_db):
    """Indicator columns are not entirely NaN (sufficient rows exist)."""
    df = data.load_df(populated_db)
    for col in ["rsi_14", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_mid", "bb_lower", "ema_50"]:
        non_null = df[col].notna().sum()
        assert non_null > 0, f"Column {col!r} is entirely NaN"


def test_timestamp_format(populated_db):
    """Timestamp values match YYYY-MM-DD HH:MM:SS format."""
    df = data.load_df(populated_db)
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
    for ts in df["timestamp"].head(5):
        assert pattern.match(ts), f"Unexpected timestamp format: {ts!r}"


def test_upsert_does_not_duplicate(db):
    """Calling refresh_db() twice does not duplicate rows."""
    mock_exchange = MagicMock()
    mock_exchange.milliseconds.return_value = int(time.time() * 1000)
    mock_exchange.fetch_ohlcv.return_value = _make_ohlcv()

    with patch("data.ccxt.binance", return_value=mock_exchange):
        data.refresh_db(db_path=db)
        data.refresh_db(db_path=db)

    conn = sqlite3.connect(db)
    cur = conn.execute("SELECT COUNT(*) FROM btc_ohlcv")
    count_after_two_runs = cur.fetchone()[0]
    conn.close()

    assert count_after_two_runs == len(_make_ohlcv()), (
        f"Expected {len(_make_ohlcv())} rows, got {count_after_two_runs} — possible duplication"
    )
