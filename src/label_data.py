"""
Triple-barrier labeling for training data, using ATR-based (volatility
normalized) barriers instead of a flat percentage. This means a volatile
stock and a calm stock get proportionally different target/stop distances,
which produces cleaner, more comparable labels across the whole universe.

For each (date, symbol), look forward up to `max_days` trading days and
label based on whichever barrier is hit first:
    1  -> profit target hit first (upper barrier)
    0  -> stop-loss hit first (lower barrier)
   -1  -> neither hit, time limit expired (excluded from training)

Usage:
    python label_data.py
"""
import numpy as np
import pandas as pd

from db import get_connection

ATR_TARGET_MULT = 2.0   # target = entry + 2.0 * ATR(14)
ATR_STOP_MULT = 1.2     # stop   = entry - 1.2 * ATR(14)
ATR_LOOKBACK = 14
MAX_DAYS = 10


def compute_atr(df):
    """Simple ATR(14) from OHLC, used only for labeling (independent of the
    features table so this script can run standalone right after fetch)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(ATR_LOOKBACK).mean()


def label_symbol(df):
    df = df.sort_values("date").reset_index(drop=True)
    df["atr"] = compute_atr(df)

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atr = df["atr"].values
    dates = df["date"].values
    n = len(df)

    labels = np.full(n, np.nan)
    barrier_hit = [None] * n
    exit_dates = [None] * n

    for i in range(n - 1):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = closes[i]
        upper = entry + ATR_TARGET_MULT * atr[i]
        lower = entry - ATR_STOP_MULT * atr[i]
        end = min(i + MAX_DAYS, n - 1)

        for j in range(i + 1, end + 1):
            if highs[j] >= upper:
                labels[i] = 1
                barrier_hit[i] = "target"
                exit_dates[i] = dates[j]
                break
            if lows[j] <= lower:
                labels[i] = 0
                barrier_hit[i] = "stop"
                exit_dates[i] = dates[j]
                break
        else:
            labels[i] = -1
            barrier_hit[i] = "expired"
            exit_dates[i] = dates[end]

    df["label"] = labels
    df["barrier_hit"] = barrier_hit
    df["exit_date"] = exit_dates
    return df


def main():
    conn = get_connection()
    prices = pd.read_sql(
        "SELECT date, symbol, close, high, low FROM prices ORDER BY symbol, date",
        conn,
    )
    if prices.empty:
        print("No price data found. Run fetch_data.py first.")
        return

    all_labeled = []
    symbols = prices["symbol"].unique()
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] labeling {sym} ...")
        sdf = prices[prices["symbol"] == sym]
        all_labeled.append(label_symbol(sdf))

    labeled = pd.concat(all_labeled, ignore_index=True)
    labeled = labeled.dropna(subset=["label"])

    rows = labeled[["date", "symbol", "label", "barrier_hit", "exit_date"]].values.tolist()
    conn.executemany(
        """INSERT OR REPLACE INTO labels (date, symbol, label, barrier_hit, exit_date)
           VALUES (?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"\nDone. Wrote {len(rows)} labels ({(labeled['label']==1).sum()} target-hit, "
          f"{(labeled['label']==0).sum()} stop-hit, {(labeled['label']==-1).sum()} expired).")


if __name__ == "__main__":
    main()
