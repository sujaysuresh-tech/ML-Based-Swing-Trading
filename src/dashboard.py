"""
Streamlit dashboard — read-only view of daily stock suggestions, per-stock
charts, and historical signal performance.

Usage:
    streamlit run dashboard.py
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from db import get_connection
from run_pipeline import run_daily_pipeline

st.set_page_config(page_title="Nifty 100 Swing Signals", layout="wide")


@st.cache_data(ttl=300)
def load_signals():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM signals ORDER BY date DESC, confidence DESC", conn)
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_price_history(symbol, days=120):
    conn = get_connection()
    df = pd.read_sql(
        """SELECT * FROM prices WHERE symbol = ? ORDER BY date DESC LIMIT ?""",
        conn, params=(symbol, days),
    )
    conn.close()
    return df.sort_values("date")


def render_suggestions_tab(signals):
    if signals.empty:
        st.info("No signals yet. Run `python score_signals.py` first.")
        return

    latest_date = signals["date"].max()
    today = signals[signals["date"] == latest_date].copy()

    st.subheader(f"Suggestions for {latest_date}")
    display_cols = ["symbol", "signal", "confidence", "entry_price", "stop_loss", "target", "outcome"]
    st.dataframe(today[display_cols], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Inspect a suggestion")
    symbol = st.selectbox("Select stock", today["symbol"].tolist())
    row = today[today["symbol"] == symbol].iloc[0]

    col1, col2 = st.columns([2, 1])
    with col1:
        hist = load_price_history(symbol)
        if not hist.empty:
            fig = go.Figure(data=[go.Candlestick(
                x=hist["date"], open=hist["open"], high=hist["high"],
                low=hist["low"], close=hist["close"],
            )])
            fig.add_hline(y=row["entry_price"], line_dash="dot", annotation_text="Entry")
            fig.add_hline(y=row["stop_loss"], line_color="red", annotation_text="Stop")
            fig.add_hline(y=row["target"], line_color="green", annotation_text="Target")
            fig.update_layout(height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.metric("Confidence", f"{row['confidence']:.0%}")
        st.metric("Entry", row["entry_price"])
        st.metric("Stop-loss", row["stop_loss"])
        st.metric("Target", row["target"])

        st.markdown("**Why this signal (SHAP)**")
        try:
            top_feats = json.loads(row["top_features"])
            for tf in top_feats:
                direction = "up" if tf["impact"] > 0 else "down"
                st.write(f"- `{tf['feature']}` pushed confidence **{direction}** ({tf['impact']:+.3f})")
        except (TypeError, json.JSONDecodeError):
            st.write("No explanation available.")


def render_performance_tab(signals):
    if signals.empty:
        st.info("No signal history yet.")
        return

    st.subheader("Historical signal performance")
    resolved = signals[signals["outcome"] != "pending"]
    if resolved.empty:
        st.info("No resolved signals yet — outcomes populate once trades hit target/stop or expire.")
        return

    win_rate = (resolved["outcome"] == "hit_target").mean()
    st.metric("Win rate (target vs stop)", f"{win_rate:.0%}")
    st.dataframe(
        resolved[["date", "symbol", "signal", "confidence", "outcome", "exit_date", "exit_price"]],
        use_container_width=True, hide_index=True,
    )


def main():
    st.title("📈 Nifty 100 Swing Trading — Suggestions Dashboard")
    st.caption("Suggestion-only system. You review and execute trades manually.")

    col1, col2 = st.columns([1, 3])
    with col1:
        run_clicked = st.button("🔄 Fetch data + generate today's suggestions", type="primary")
    with col2:
        st.caption("Runs fetch_data → build_features → score_signals automatically. "
                    "Requires a trained model (run train_model.py at least once beforehand).")

    if run_clicked:
        status_box = st.empty()
        log_lines = []

        def log(msg):
            log_lines.append(msg)
            status_box.info("\n\n".join(log_lines))

        with st.spinner("Running daily pipeline..."):
            full_log, ok = run_daily_pipeline(full_history=False, log_callback=log)

        if ok:
            st.success("Pipeline complete — signals refreshed below.")
        else:
            st.error("Pipeline failed partway through. See details below.")
        with st.expander("Full pipeline log"):
            st.code(full_log)

        load_signals.clear()  # bust the cache so the new signals show immediately

    st.markdown("---")

    signals = load_signals()
    tab1, tab2 = st.tabs(["Today's Suggestions", "Performance Tracker"])
    with tab1:
        render_suggestions_tab(signals)
    with tab2:
        render_performance_tab(signals)


if __name__ == "__main__":
    main()
