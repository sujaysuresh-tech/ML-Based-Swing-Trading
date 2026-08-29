"""
Runs the daily pipeline end-to-end: fetch latest data -> build features ->
score signals. Callable directly (`python run_pipeline.py`) or imported and
used by the Streamlit dashboard so a single button click does all three
steps instead of running each script by hand.
"""
import io
import contextlib

import fetch_data
import build_features
import score_signals


def run_daily_pipeline(full_history: bool = False, log_callback=None):
    """
    Runs fetch_data -> build_features -> score_signals in sequence.
    Captures each step's printed output and returns it as a single string
    (also streamed step-by-step to log_callback if provided, e.g. st.write).
    """
    steps = [
        ("Fetching latest data", lambda: fetch_data.main_args(full_history=full_history)),
        ("Building features", build_features.main),
        ("Scoring signals", score_signals.main),
    ]

    full_log = []
    for label, fn in steps:
        if log_callback:
            log_callback(f"⏳ {label}...")
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                fn()
        except Exception as e:
            msg = f"❌ {label} failed: {e}"
            full_log.append(buf.getvalue())
            full_log.append(msg)
            if log_callback:
                log_callback(msg)
            return "\n".join(full_log), False
        output = buf.getvalue()
        full_log.append(f"--- {label} ---\n{output}")
        if log_callback:
            log_callback(f"✅ {label} done")

    return "\n".join(full_log), True


if __name__ == "__main__":
    log, ok = run_daily_pipeline(log_callback=print)
    print("\nPipeline succeeded." if ok else "\nPipeline failed.")
