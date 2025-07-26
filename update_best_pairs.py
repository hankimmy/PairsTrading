import pandas as pd
import os
from signals_engine import SignalsEngine
from datetime import datetime
import sys

log_path = f"logs/update_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.makedirs("logs", exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

best_pairs_df = pd.read_csv("best_pairs.csv")

for _, row in best_pairs_df.iterrows():
    sector = row["sector"]
    ticker_x = row["ticker_x"]
    ticker_y = row["ticker_y"]
    print(f"\n=== Processing {ticker_x}-{ticker_y} in {sector} sector ===")

    summary_path = f"summary/{sector}_summary.csv"
    if not os.path.exists(summary_path):
        print(f"[SKIP] Summary file not found: {summary_path}")
        continue

    summary_df = pd.read_csv(summary_path)
    match = summary_df[
        ((summary_df["Ticker_X"] == ticker_x) & (summary_df["Ticker_Y"] == ticker_y)) |
        ((summary_df["Ticker_X"] == ticker_y) & (summary_df["Ticker_Y"] == ticker_x))
    ]

    if match.empty:
        print(f"[SKIP] No summary match found for {ticker_x}-{ticker_y}")
        continue

    best_window = int(match.iloc[0]["Best_Window"])
    print(f"Using best window: {best_window}")

    signal_csv_path = f"{sector}_signals/{ticker_x}_{ticker_y}.csv"

    try:
        engine = SignalsEngine(
            ticker_x=ticker_x,
            ticker_y=ticker_y,
            csv_path=signal_csv_path,
            start="2024-01-01",
            window=best_window
        )
        eval_results = engine.evaluate()
        print(
            f"Return={eval_results['cum_returns'].iloc[-1] * 100:.2f}% | "
            f"Sharpe={eval_results['sharpe']:.2f} | "
            f"Trades={eval_results['num_trades']}"
        )
        print(f"[DONE] Saved signals to {signal_csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed for {ticker_x}-{ticker_y}: {e}")
