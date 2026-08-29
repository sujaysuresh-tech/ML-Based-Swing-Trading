"""
Train a LightGBM classifier (with a scikit-learn Random Forest baseline)
using walk-forward (expanding window) validation on Nifty 100 features +
triple-barrier labels.

Usage:
    python train_model.py
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from db import get_connection

MODEL_DIR = Path(__file__).resolve().parent.parent / "data"
FEATURE_COLS = [
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_mid", "atr_14", "adx_14",
    "return_5d", "return_10d", "return_20d", "volume_ratio",
    "dist_from_52wk_high", "dist_from_52wk_low",
    "rsi_rank", "return_5d_rank", "nifty_trend", "india_vix",
]


def load_dataset(conn):
    df = pd.read_sql(
        """
        SELECT f.*, l.label
        FROM features f
        JOIN labels l ON f.date = l.date AND f.symbol = l.symbol
        WHERE l.label IN (0, 1)
        ORDER BY f.date
        """,
        conn,
        parse_dates=["date"],
    )
    return df.dropna(subset=FEATURE_COLS + ["label"])


def walk_forward_splits(df, n_folds=3):
    """Yield (train_df, test_df) using an expanding window over calendar years."""
    years = sorted(df["date"].dt.year.unique())
    if len(years) < n_folds + 1:
        # not enough distinct years for the requested folds -> single split
        cutoff = df["date"].quantile(0.8)
        yield df[df["date"] <= cutoff], df[df["date"] > cutoff]
        return

    test_years = years[-n_folds:]
    for ty in test_years:
        train = df[df["date"].dt.year < ty]
        test = df[df["date"].dt.year == ty]
        if len(train) > 0 and len(test) > 0:
            yield train, test


def main():
    conn = get_connection()
    df = load_dataset(conn)
    conn.close()

    if df.empty:
        print("No labeled feature data found. Run build_features.py and label_data.py first.")
        return

    print(f"Dataset: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")

    # --- Baseline: scikit-learn Random Forest, walk-forward ---
    print("\n=== Baseline: Random Forest (walk-forward) ===")
    for i, (train, test) in enumerate(walk_forward_splits(df), 1):
        rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(train[FEATURE_COLS], train["label"])
        preds = rf.predict(test[FEATURE_COLS])
        proba = rf.predict_proba(test[FEATURE_COLS])[:, 1]
        acc = accuracy_score(test["label"], preds)
        auc = roc_auc_score(test["label"], proba) if test["label"].nunique() > 1 else float("nan")
        print(f"Fold {i}: train={len(train)} test={len(test)} acc={acc:.3f} auc={auc:.3f}")

    # --- Production model: LightGBM, walk-forward ---
    print("\n=== Production model: LightGBM (walk-forward) ===")
    final_model = None
    for i, (train, test) in enumerate(walk_forward_splits(df), 1):
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model.fit(train[FEATURE_COLS], train["label"])
        preds = model.predict(test[FEATURE_COLS])
        proba = model.predict_proba(test[FEATURE_COLS])[:, 1]
        acc = accuracy_score(test["label"], preds)
        auc = roc_auc_score(test["label"], proba) if test["label"].nunique() > 1 else float("nan")
        print(f"Fold {i}: train={len(train)} test={len(test)} acc={acc:.3f} auc={auc:.3f}")
        final_model = model  # keep the most recent fold's model

    # Refit on ALL data for the final deployed model
    print("\nRefitting final model on full dataset for deployment...")
    final_model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    final_model.fit(df[FEATURE_COLS], df["label"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final_model, "features": FEATURE_COLS}, MODEL_DIR / "model.pkl")
    print(f"Saved deployed model to {MODEL_DIR / 'model.pkl'}")
    print("\nNOTE: Accuracy/AUC alone don't tell you if this is tradeable — "
          "run a proper backtest (Sharpe, drawdown, profit factor) before trusting signals.")


if __name__ == "__main__":
    main()
