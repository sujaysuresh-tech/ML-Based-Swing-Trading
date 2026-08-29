"""
Build technical, price-action, cross-sectional, and market-regime features
from raw OHLCV data stored in SQLite, and write them back to the `features`
table.

Usage:
    python build_features.py
"""
import pandas as pd
import pandas_ta as ta

from db import get_connection

REGIME_SYMBOLS = {"^NSEI", "^INDIAVIX"}


def load_prices(conn):
    return pd.read_sql("SELECT * FROM prices ORDER BY symbol, date", conn, parse_dates=["date"])


def compute_stock_features(df):
    """df: single-symbol OHLCV dataframe, sorted by date."""
    df = df.copy()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(append=True)
    df.ta.bbands(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)

    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["dist_from_52wk_high"] = df["close"] / df["close"].rolling(252, min_periods=20).max() - 1
    df["dist_from_52wk_low"] = df["close"] / df["close"].rolling(252, min_periods=20).min() - 1

    # Standardise pandas-ta's generated column names to our schema
    rename_map = {}
    for col in df.columns:
        if col.startswith("RSI_14"):
            rename_map[col] = "rsi_14"
        elif col.startswith("MACD_") and "h" not in col.lower() and "s" not in col.lower():
            rename_map[col] = "macd"
        elif col.startswith("MACDh_"):
            rename_map[col] = "macd_hist"
        elif col.startswith("MACDs_"):
            rename_map[col] = "macd_signal"
        elif col.startswith("BBU_"):
            rename_map[col] = "bb_upper"
        elif col.startswith("BBL_"):
            rename_map[col] = "bb_lower"
        elif col.startswith("BBM_"):
            rename_map[col] = "bb_mid"
        elif col.startswith("ATRr_") or col.startswith("ATR_"):
            rename_map[col] = "atr_14"
        elif col.startswith("ADX_"):
            rename_map[col] = "adx_14"
    df = df.rename(columns=rename_map)
    return df


def main():
    conn = get_connection()
    prices = load_prices(conn)

    if prices.empty:
        print("No price data found. Run fetch_data.py first.")
        return

    stock_prices = prices[~prices["symbol"].isin(REGIME_SYMBOLS)]
    regime_prices = prices[prices["symbol"].isin(REGIME_SYMBOLS)]

    # --- Regime features (shared across all stocks) ---
    nifty = regime_prices[regime_prices["symbol"] == "^NSEI"][["date", "close"]].rename(
        columns={"close": "nifty_close"}
    )
    nifty["nifty_ema50"] = nifty["nifty_close"].ewm(span=50).mean()
    nifty["nifty_trend"] = (nifty["nifty_close"] > nifty["nifty_ema50"]).astype(int)

    vix = regime_prices[regime_prices["symbol"] == "^INDIAVIX"][["date", "close"]].rename(
        columns={"close": "india_vix"}
    )
    regime = pd.merge(nifty[["date", "nifty_trend"]], vix[["date", "india_vix"]], on="date", how="outer")

    # --- Per-stock technical + price-action features ---
    all_feats = []
    symbols = stock_prices["symbol"].unique()
    for i, sym in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] computing features for {sym} ...")
        sdf = stock_prices[stock_prices["symbol"] == sym].sort_values("date")
        sdf = compute_stock_features(sdf)
        all_feats.append(sdf)

    feats = pd.concat(all_feats, ignore_index=True)

    # --- Cross-sectional ranks (per date, across all stocks) ---
    feats["rsi_rank"] = feats.groupby("date")["rsi_14"].rank(pct=True)
    feats["return_5d_rank"] = feats.groupby("date")["return_5d"].rank(pct=True)

    # --- Merge regime features ---
    feats = pd.merge(feats, regime, on="date", how="left")

    cols = [
        "date", "symbol", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_mid", "atr_14", "adx_14",
        "return_5d", "return_10d", "return_20d", "volume_ratio",
        "dist_from_52wk_high", "dist_from_52wk_low",
        "rsi_rank", "return_5d_rank", "nifty_trend", "india_vix",
    ]
    for c in cols:
        if c not in feats.columns:
            feats[c] = None

    out = feats[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["symbol"])

    rows = out.values.tolist()
    placeholders = ", ".join(["?"] * len(cols))
    conn.executemany(
        f"INSERT OR REPLACE INTO features ({', '.join(cols)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"\nDone. Wrote {len(rows)} feature rows for {len(symbols)} symbols.")


if __name__ == "__main__":
    main()
