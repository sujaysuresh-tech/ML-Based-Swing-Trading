"""
Fetch OHLCV data for the Nifty 100 universe (+ Nifty 50 index + India VIX
for regime features) using yfinance, and store it in SQLite.

Usage:
    python fetch_data.py                 # incremental: last 5 days
    python fetch_data.py --full-history  # full 10-year pull (first run)
"""
import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

from db import get_connection

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SYMBOLS_FILE = DATA_DIR / "nifty100_list.csv"

REGIME_TICKERS = {
    "^NSEI": "NIFTY50",
    "^INDIAVIX": "INDIAVIX",
}


def load_symbols():
    df = pd.read_csv(SYMBOLS_FILE)
    return df["symbol"].tolist()


def fetch_and_store(symbol, period, conn):
    try:
        df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    except Exception as e:
        print(f"  [WARN] failed to fetch {symbol}: {e}")
        return 0

    if df.empty:
        print(f"  [WARN] no data returned for {symbol}")
        return 0

    # Newer yfinance versions can return MultiIndex columns (Price, Ticker)
    # even for a single symbol -- flatten to plain column names first.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    # The date column can come back as "date", "datetime", or "index"
    # depending on yfinance version -- normalise it.
    date_col = next((c for c in df.columns if c in ("date", "datetime", "index")), None)
    if date_col is None:
        print(f"  [WARN] could not find a date column for {symbol}, columns were: {list(df.columns)}")
        return 0
    df = df.rename(columns={date_col: "date"})

    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [WARN] {symbol} missing columns {missing}, skipping")
        return 0

    df["symbol"] = symbol
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    rows = df[["date", "symbol", "open", "high", "low", "close", "volume"]].values.tolist()
    conn.executemany(
        """INSERT OR REPLACE INTO prices (date, symbol, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


def main_args(full_history: bool = False):
    """Callable version (no argparse) so other scripts/the dashboard can invoke this directly."""
    period = "10y" if full_history else "5d"
    symbols = load_symbols()
    all_tickers = symbols + list(REGIME_TICKERS.keys())

    conn = get_connection()
    total_rows = 0
    for i, symbol in enumerate(all_tickers, 1):
        print(f"[{i}/{len(all_tickers)}] fetching {symbol} ({period}) ...")
        total_rows += fetch_and_store(symbol, period, conn)

    conn.close()
    print(f"\nDone. Wrote/updated {total_rows} rows across {len(all_tickers)} tickers.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-history", action="store_true", help="Pull 10 years instead of last 5 days")
    args = parser.parse_args()
    main_args(full_history=args.full_history)


if __name__ == "__main__":
    main()
