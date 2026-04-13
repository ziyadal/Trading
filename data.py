"""
data.py — BTC/USDT 5m OHLCV ingestion and indicator calculation.

Public API:
    refresh_db(db_path)  — fetch last 2 days from Binance, compute indicators, upsert to SQLite
    load_df(db_path)     — load full btc_ohlcv table from SQLite into a DataFrame
"""

import sqlite3

import ccxt
import pandas as pd

DB_PATH = "trading.db"
TABLE = "btc_ohlcv"
COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "ema_50",
]


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(close: pd.Series, length: int = 20, num_std: float = 2.0):
    mid = close.rolling(length).mean()
    std = close.rolling(length).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def _ema(close: pd.Series, span: int = 50) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_db(db_path: str = DB_PATH) -> None:
    """Fetch last 2 days of BTC/USDT 5m candles from Binance, compute the 4
    technical indicators, and upsert all rows into the SQLite database at
    *db_path*.  Existing rows are replaced by timestamp (primary key) so data
    persists across runs without duplication.
    """
    exchange = ccxt.binance()

    # 2 days = 576 candles at 5m; fetch up to 1000 to be safe
    since_ms = exchange.milliseconds() - 2 * 24 * 60 * 60 * 1000
    raw = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", since=since_ms, limit=1000)

    df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = (
        pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    df = df.drop(columns=["ts_ms"])

    # Compute indicators
    df["rsi_14"] = _rsi(df["close"])
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bbands(df["close"])
    df["ema_50"] = _ema(df["close"])

    df = df[COLUMNS]

    # Upsert into SQLite
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                timestamp   TEXT PRIMARY KEY,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                volume      REAL,
                rsi_14      REAL,
                macd        REAL,
                macd_signal REAL,
                macd_hist   REAL,
                bb_upper    REAL,
                bb_mid      REAL,
                bb_lower    REAL,
                ema_50      REAL
            )
        """)
        placeholders = ", ".join(["?"] * len(COLUMNS))
        col_list = ", ".join(COLUMNS)
        conn.executemany(
            f"INSERT OR REPLACE INTO {TABLE} ({col_list}) VALUES ({placeholders})",
            df.values.tolist(),
        )
        conn.commit()
    finally:
        conn.close()


def load_df(db_path: str = DB_PATH) -> pd.DataFrame:
    """Return the full btc_ohlcv table as a pandas DataFrame.

    The *timestamp* column is a string in ``YYYY-MM-DD HH:MM:SS`` format so
    that pandas ``df.query()`` string comparisons work directly, e.g.::

        df.query("timestamp >= '2026-04-03 00:00:00'")
    """
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY timestamp", conn)
    finally:
        conn.close()


if __name__ == "__main__":
    print("Fetching BTC/USDT 5m candles from Binance...")
    refresh_db()
    df = load_df()
    print(f"Database built: {len(df)} candles.  Latest: {df['timestamp'].iloc[-1]}")
