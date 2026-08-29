"""
SQLite schema and connection helper for the swing trading pipeline.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "market.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS features (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_hist REAL,
    bb_upper REAL,
    bb_lower REAL,
    bb_mid REAL,
    atr_14 REAL,
    adx_14 REAL,
    return_5d REAL,
    return_10d REAL,
    return_20d REAL,
    volume_ratio REAL,
    dist_from_52wk_high REAL,
    dist_from_52wk_low REAL,
    rsi_rank REAL,
    return_5d_rank REAL,
    nifty_trend REAL,
    india_vix REAL,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS labels (
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    label INTEGER,
    barrier_hit TEXT,
    exit_date TEXT,
    PRIMARY KEY (date, symbol)
);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    signal TEXT,
    confidence REAL,
    entry_price REAL,
    stop_loss REAL,
    target REAL,
    top_features TEXT,
    outcome TEXT DEFAULT 'pending',
    exit_date TEXT,
    exit_price REAL
);
"""


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


if __name__ == "__main__":
    conn = get_connection()
    print(f"Database initialised at {DB_PATH}")
    conn.close()
