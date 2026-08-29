# Nifty 100 ML Swing Trading Suggestion System

A fully free, manually-run pipeline that suggests swing-trading long/short
setups on Nifty 100 stocks. It does **not** place trades — it only generates
ranked suggestions (entry zone, stop-loss, target, confidence, and a SHAP
explanation) for you to act on manually.

## Architecture

```
fetch_data.py        -> pulls OHLCV via yfinance, stores in SQLite
build_features.py    -> pandas-ta + custom + cross-sectional features
label_data.py         -> triple-barrier labeling for training
train_model.py        -> trains LightGBM (with sklearn RF baseline), walk-forward
score_signals.py      -> scores latest data with trained model, applies filters
dashboard.py           -> Streamlit app, reads SQLite, shows suggestions
```

## Daily manual workflow

Just one command:

```bash
streamlit run src/dashboard.py
```

Then click **"🔄 Fetch data + generate today's suggestions"** in the app —
this runs fetch_data → build_features → score_signals automatically and
refreshes the table. (Requires a trained model — see the one-time workflow
below — run at least once first.)

If you ever prefer running the steps by hand instead of the button:
```bash
python src/fetch_data.py
python src/build_features.py
python src/score_signals.py
```

## One-time / periodic workflow (retraining)

```bash
python src/fetch_data.py --full-history     # first time: pulls 10 years
python src/build_features.py
python src/label_data.py
python src/train_model.py                   # walk-forward validated, saves model.pkl
```

## Stack (all free)

| Layer            | Tool                          |
|------------------|--------------------------------|
| Data             | yfinance                      |
| Storage          | SQLite (data/market.db)        |
| Features         | pandas-ta, pandas              |
| Model            | LightGBM (sklearn RF baseline) |
| Interpretability | SHAP                            |
| Backtest         | vectorbt                        |
| Dashboard        | Streamlit                       |
| Automation       | Manual (run scripts yourself)   |

## Notes

- Universe: Nifty 100 (edit `data/nifty100_list.csv` to update constituents).
- This is a **suggestion-only** system. You review and execute trades manually.
- Not financial advice — validate thoroughly with backtesting and paper
  trading before using real capital.
