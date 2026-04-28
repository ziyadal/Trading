"""
data.py — BTC/USDT OHLCV ingestion, indicator calculation, and multi-timeframe storage.

Public API:
    refresh_db(db_path)  — fetch last 2 days of 5m candles from Binance, compute indicators,
                           upsert into the 5m table and rebuild 1h/4h aggregate tables.
    backfill_db(days, db_path) — paginate through Binance history and rebuild all tables.
    load_df(db_path, table) — load one of the three tables as a DataFrame.

Tables written:
    btc_ohlcv     — 5m candles + indicators (primary)
    btc_ohlcv_1h  — 1h aggregated + indicators
    btc_ohlcv_4h  — 4h aggregated + indicators

Indicators (computed on every timeframe):
    Core:        rsi_14, macd/signal/hist, bb_upper/mid/lower, ema_50
    Volatility:  atr_14, atr_pct, bb_width, bb_width_rank
    Momentum:    rsi_slope_5
    Trend:       dist_from_ema50_pct
    Volume:      vol_ema_20, vol_ratio
    Structure:   swing_high, swing_low (bool), regime
    Forward:     fwd_7d_return_pct, fwd_7d_high_pct, fwd_7d_low_pct
                 (computed from future bars on the same timeframe — used by
                 find_analogues at query time. NULL for rows too recent to score.)
"""

import sqlite3
import time

import ccxt
import numpy as np
import pandas as pd

DB_PATH = "trading.db"

TABLE_5M = "btc_ohlcv"
TABLE_1H = "btc_ohlcv_1h"
TABLE_4H = "btc_ohlcv_4h"

COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower", "ema_50",
    "atr_14", "atr_pct",
    "bb_width", "bb_width_rank",
    "rsi_slope_5",
    "dist_from_ema50_pct",
    "vol_ema_20", "vol_ratio",
    "swing_high", "swing_low",
    "regime",
    "fwd_7d_return_pct", "fwd_7d_high_pct", "fwd_7d_low_pct",
]

# 7 days expressed in bars on each timeframe — used for forward-return windows.
FWD_BARS = {
    TABLE_5M: 7 * 24 * 12,   # 2016
    TABLE_1H: 7 * 24,        # 168
    TABLE_4H: 7 * 6,         # 42
}

# Window sizes for swing-point detection on each timeframe.
SWING_WINDOW = {
    TABLE_5M: 24,   # 2 hours either side
    TABLE_1H: 8,    # 8 hours
    TABLE_4H: 5,    # ~20 hours
}

# Lookback (in bars) for bb_width percentile rank. ~30d on each timeframe.
BB_RANK_WINDOW = {
    TABLE_5M: 30 * 24 * 12,
    TABLE_1H: 30 * 24,
    TABLE_4H: 30 * 6,
}


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


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _swing_flags(high: pd.Series, low: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Local-maximum / local-minimum flags over a +/- window bar span."""
    size = 2 * window + 1
    rolling_max = high.rolling(size, center=True).max()
    rolling_min = low.rolling(size, center=True).min()
    return (high == rolling_max), (low == rolling_min)


def _regime(
    close: pd.Series,
    ema_50: pd.Series,
    bb_width_rank: pd.Series,
) -> pd.Series:
    """Rule-based regime tag per bar.

    TRENDING_UP    — close > EMA-50 and EMA-50 rising over 10 bars
    TRENDING_DOWN  — close < EMA-50 and EMA-50 falling over 10 bars
    SQUEEZE        — bb_width_rank < 0.20 (volatility compressed vs recent history)
    RANGE          — everything else
    """
    ema_slope = ema_50.diff(10)
    trending_up = (close > ema_50) & (ema_slope > 0)
    trending_down = (close < ema_50) & (ema_slope < 0)
    squeeze = bb_width_rank < 0.20

    regime = pd.Series("RANGE", index=close.index)
    regime = regime.where(~trending_up, "TRENDING_UP")
    regime = regime.where(~trending_down, "TRENDING_DOWN")
    # Squeeze overrides range but not a clean trend.
    regime = regime.where(~(squeeze & (regime == "RANGE")), "SQUEEZE")
    return regime


def _compute_indicators(df: pd.DataFrame, bars_30d: int, swing_window: int, fwd_bars: int) -> pd.DataFrame:
    """Populate every indicator column on a dataframe of OHLCV candles.

    *df* must be sorted ascending by timestamp and contain open/high/low/close/volume.
    Returns the same DataFrame with all columns in COLUMNS present.
    """
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    df["rsi_14"] = _rsi(close)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(close)
    df["bb_upper"], df["bb_mid"], df["bb_lower"] = _bbands(close)
    df["ema_50"] = _ema(close)

    df["atr_14"] = _atr(high, low, close)
    df["atr_pct"] = df["atr_14"] / close * 100

    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"] * 100
    df["bb_width_rank"] = (
        df["bb_width"].rolling(bars_30d, min_periods=50).rank(pct=True)
    )

    df["rsi_slope_5"] = df["rsi_14"] - df["rsi_14"].shift(5)
    df["dist_from_ema50_pct"] = (close - df["ema_50"]) / df["ema_50"] * 100

    df["vol_ema_20"] = volume.ewm(span=20, adjust=False).mean()
    df["vol_ratio"] = volume / df["vol_ema_20"]

    swing_high, swing_low = _swing_flags(high, low, swing_window)
    df["swing_high"] = swing_high.astype("Int64")
    df["swing_low"] = swing_low.astype("Int64")

    df["regime"] = _regime(close, df["ema_50"], df["bb_width_rank"])

    # Forward-looking stats over the next `fwd_bars` bars. The last fwd_bars
    # rows naturally get NaN because the window extends past the data.
    fwd_close = close.shift(-fwd_bars)
    fwd_high = high.rolling(fwd_bars).max().shift(-fwd_bars)
    fwd_low = low.rolling(fwd_bars).min().shift(-fwd_bars)
    df["fwd_7d_return_pct"] = (fwd_close - close) / close * 100
    df["fwd_7d_high_pct"] = (fwd_high - close) / close * 100
    df["fwd_7d_low_pct"] = (fwd_low - close) / close * 100

    # Replace inf/nan that the downstream SQLite layer can't store.
    df = df.replace([np.inf, -np.inf], np.nan)

    return df[COLUMNS]


def _resample(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate the 5m OHLCV frame up to *rule* (pandas offset, e.g. '1H', '4H').

    Returns a new frame with the same timestamp-string format as the input.
    """
    df = df_5m.copy()
    df["ts"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("ts")

    agg = df.resample(rule, label="right", closed="right").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    agg = agg.reset_index()
    agg["timestamp"] = agg["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return agg.drop(columns=["ts"])[["timestamp", "open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

_CREATE_SQL = """
    CREATE TABLE IF NOT EXISTS {table} (
        timestamp              TEXT PRIMARY KEY,
        open                   REAL,
        high                   REAL,
        low                    REAL,
        close                  REAL,
        volume                 REAL,
        rsi_14                 REAL,
        macd                   REAL,
        macd_signal            REAL,
        macd_hist              REAL,
        bb_upper               REAL,
        bb_mid                 REAL,
        bb_lower               REAL,
        ema_50                 REAL,
        atr_14                 REAL,
        atr_pct                REAL,
        bb_width               REAL,
        bb_width_rank          REAL,
        rsi_slope_5            REAL,
        dist_from_ema50_pct    REAL,
        vol_ema_20             REAL,
        vol_ratio              REAL,
        swing_high             INTEGER,
        swing_low              INTEGER,
        regime                 TEXT,
        fwd_7d_return_pct      REAL,
        fwd_7d_high_pct        REAL,
        fwd_7d_low_pct         REAL
    )
"""


def _ensure_columns(conn: sqlite3.Connection, table: str) -> None:
    """ALTER TABLE ADD COLUMN for any column missing on an older DB.

    Safe to run repeatedly — skips existing columns.
    """
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    type_map = {
        "swing_high": "INTEGER",
        "swing_low": "INTEGER",
        "regime": "TEXT",
    }
    for col in COLUMNS:
        if col in existing or col == "timestamp":
            continue
        col_type = type_map.get(col, "REAL")
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")


def _write(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    conn.execute(_CREATE_SQL.format(table=table))
    _ensure_columns(conn, table)
    placeholders = ", ".join(["?"] * len(COLUMNS))
    col_list = ", ".join(COLUMNS)
    # Replace NaN with None so sqlite stores NULLs rather than the string 'nan'.
    rows = df.where(pd.notnull(df), None).values.tolist()
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})",
        rows,
    )


def _build_and_store(conn: sqlite3.Connection, table: str, df_ohlcv: pd.DataFrame) -> None:
    """Run the indicator pipeline for one timeframe and upsert."""
    enriched = _compute_indicators(
        df_ohlcv.copy(),
        bars_30d=BB_RANK_WINDOW[table],
        swing_window=SWING_WINDOW[table],
        fwd_bars=FWD_BARS[table],
    )
    _write(conn, table, enriched)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _fetch_ohlcv_to_df(since_ms: int, limit: int, exchange: ccxt.Exchange) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv("BTC/USDT", timeframe="5m", since=since_ms, limit=limit)
    df = pd.DataFrame(raw, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    df["timestamp"] = (
        pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    return df.drop(columns=["ts_ms"])[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]


def _load_ohlcv_only(conn: sqlite3.Connection) -> pd.DataFrame:
    """Read raw OHLCV from the 5m table (for rebuilding 1h/4h aggregates)."""
    return pd.read_sql(
        f"SELECT timestamp, open, high, low, close, volume FROM {TABLE_5M} ORDER BY timestamp",
        conn,
    )


def refresh_db(db_path: str = DB_PATH) -> None:
    """Fetch last 2 days of 5m candles, upsert, and rebuild 1h/4h aggregate tables."""
    exchange = ccxt.binance()
    since_ms = exchange.milliseconds() - 2 * 24 * 60 * 60 * 1000
    new_5m = _fetch_ohlcv_to_df(since_ms, limit=1000, exchange=exchange)

    conn = sqlite3.connect(db_path)
    try:
        # Merge newly-fetched 5m candles with whatever is already in the DB so
        # indicators on the 2-day window are computed against full history.
        existing = pd.DataFrame()
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        if TABLE_5M in tables:
            existing = _load_ohlcv_only(conn)
        combined = (
            pd.concat([existing, new_5m], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        _build_and_store(conn, TABLE_5M, combined)
        _build_and_store(conn, TABLE_1H, _resample(combined, "1h"))
        _build_and_store(conn, TABLE_4H, _resample(combined, "4h"))
        conn.commit()
    finally:
        conn.close()


def backfill_db(days: int = 90, db_path: str = DB_PATH) -> None:
    """Fetch *days* of 5m history and rebuild all tables (5m, 1h, 4h) from scratch."""
    exchange = ccxt.binance()
    since_ms = exchange.milliseconds() - days * 24 * 60 * 60 * 1000
    now_ms = exchange.milliseconds()
    all_candles: list = []

    print(f"Backfilling {days} days of BTC/USDT 5m data...")

    while since_ms < now_ms:
        candles = exchange.fetch_ohlcv(
            "BTC/USDT", timeframe="5m", since=since_ms, limit=1000
        )
        if not candles:
            break
        all_candles.extend(candles)
        since_ms = candles[-1][0] + 1
        print(f"  Fetched {len(all_candles)} candles...", end="\r")
        time.sleep(exchange.rateLimit / 1000)

    print(f"  Fetched {len(all_candles)} candles total.     ")

    if not all_candles:
        print("No data fetched.")
        return

    df = pd.DataFrame(
        all_candles, columns=["ts_ms", "open", "high", "low", "close", "volume"]
    )
    df["timestamp"] = (
        pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    df = df.drop(columns=["ts_ms"]).drop_duplicates(subset="timestamp", keep="last")
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    conn = sqlite3.connect(db_path)
    try:
        _build_and_store(conn, TABLE_5M, df)
        _build_and_store(conn, TABLE_1H, _resample(df, "1h"))
        _build_and_store(conn, TABLE_4H, _resample(df, "4h"))
        conn.commit()
    finally:
        conn.close()

    print(
        f"Backfilled {len(df)} 5m candles: "
        f"{df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"  # type: ignore[reportAttributeAccessIssue]
    )


def load_df(db_path: str = DB_PATH, table: str = TABLE_5M) -> pd.DataFrame:
    """Return one of the three ohlcv tables as a DataFrame.

    *table* must be one of TABLE_5M, TABLE_1H, TABLE_4H.
    """
    conn = sqlite3.connect(db_path)
    try:
        return pd.read_sql(f"SELECT * FROM {table} ORDER BY timestamp", conn)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--backfill":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 90
        backfill_db(days=days)
    else:
        print("Fetching BTC/USDT 5m candles from Binance...")
        refresh_db()

    for table in (TABLE_5M, TABLE_1H, TABLE_4H):
        df = load_df(table=table)
        print(
            f"{table}: {len(df)} candles. "
            f"Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}"
        )
