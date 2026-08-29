"""
Score the latest day's features with the trained model, apply filters
(liquidity, confidence threshold), attach a SHAP-based explanation, and
write the resulting suggestions into the `signals` table.

Usage:
    python score_signals.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import shap

from db import get_connection

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "model.pkl"

MIN_CONFIDENCE = 0.65
TOP_N_SIGNALS = 10
PROFIT_TARGET_PCT = 0.025
STOP_LOSS_PCT = 0.015
MIN_AVG_VOLUME = 100_000  # shares/day, adjust per your liquidity comfort


def main():
    if not MODEL_PATH.exists():
        print("No trained model found. Run train_model.py first.")
        return

    bundle = joblib.load(MODEL_PATH)
    model, feature_cols = bundle["model"], bundle["features"]

    conn = get_connection()
    latest_date = pd.read_sql("SELECT MAX(date) AS d FROM features", conn)["d"].iloc[0]
    if latest_date is None:
        print("No feature data found. Run build_features.py first.")
        return

    feats = pd.read_sql(
        "SELECT * FROM features WHERE date = ?", conn, params=(latest_date,)
    ).dropna(subset=feature_cols)

    prices_today = pd.read_sql(
        "SELECT symbol, close, volume FROM prices WHERE date = ?", conn, params=(latest_date,)
    )
    avg_vol = pd.read_sql(
        """SELECT symbol, AVG(volume) AS avg_volume FROM prices
           WHERE date >= date(?, '-20 days') GROUP BY symbol""",
        conn,
        params=(latest_date,),
    )

    if feats.empty:
        print(f"No features available for {latest_date}.")
        return

    merged = feats.merge(prices_today, on="symbol").merge(avg_vol, on="symbol")

    # --- Liquidity filter ---
    merged = merged[merged["avg_volume"] >= MIN_AVG_VOLUME]

    # --- Score ---
    proba = model.predict_proba(merged[feature_cols])[:, 1]
    merged["confidence"] = proba
    merged = merged[merged["confidence"] >= MIN_CONFIDENCE]
    merged = merged.sort_values("confidence", ascending=False).head(TOP_N_SIGNALS)

    if merged.empty:
        print(f"No signals above confidence threshold ({MIN_CONFIDENCE}) for {latest_date}.")
        conn.close()
        return

    # --- SHAP explanations ---
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(merged[feature_cols])
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    rows = []
    for idx, (_, row) in enumerate(merged.iterrows()):
        contributions = dict(zip(feature_cols, sv[idx]))
        top_features = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_features_str = json.dumps([{"feature": f, "impact": round(float(v), 4)} for f, v in top_features])

        entry_price = row["close"]
        rows.append((
            latest_date,
            row["symbol"],
            "long",
            round(float(row["confidence"]), 4),
            round(float(entry_price), 2),
            round(float(entry_price * (1 - STOP_LOSS_PCT)), 2),
            round(float(entry_price * (1 + PROFIT_TARGET_PCT)), 2),
            top_features_str,
        ))

    conn.executemany(
        """INSERT INTO signals (date, symbol, signal, confidence, entry_price, stop_loss, target, top_features)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    conn.close()

    print(f"\n{len(rows)} suggestion(s) for {latest_date}:")
    for r in rows:
        print(f"  {r[1]:<15} conf={r[3]:.2f}  entry={r[4]}  stop={r[5]}  target={r[6]}")
    print("\nOpen the dashboard to review: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
