import pandas as pd
import os
from signals_engine import SignalsEngine
import sys
from datetime import datetime

log_path = f"logs/signal_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
os.makedirs("logs", exist_ok=True)
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

summary_df = pd.read_csv("signal_summary.csv")
summary_df["Pair"] = summary_df["Ticker_X"] + "-" + summary_df["Ticker_Y"]

best_pairs = [
    ("NVDA", "TSM"),
    ("NVDA", "ASML"),
    ("AVGO", "TSM"),
    ("AVGO", "AMD"),
    ("QCOM", "ADI")
]

for ticker_x, ticker_y in best_pairs:
    pair_key_1 = f"{ticker_x}-{ticker_y}"
    pair_key_2 = f"{ticker_y}-{ticker_x}"

    match = summary_df[
        (summary_df["Pair"] == pair_key_1) | (summary_df["Pair"] == pair_key_2)
    ]
    if match.empty:
        print(f"[SKIP] No summary found for {ticker_x}-{ticker_y}")
        continue

    best_window = int(match.iloc[0]["Best_Window"])
    print(f"\nProcessing {ticker_x}-{ticker_y} with window {best_window}")

    try:
        csv_path = f"./signals/{ticker_x}_{ticker_y}.csv"
        engine = SignalsEngine(
            ticker_x=ticker_x,
            ticker_y=ticker_y,
            csv_path=csv_path,
            start='2024-01-01',
            window=best_window
        )
        eval_results = engine.evaluate()
        print(
            f"Return={eval_results['cum_returns'].iloc[-1] * 100:.2f}% | "
            f"Sharpe={eval_results['sharpe']:.2f} | "
            f"Trades={eval_results['num_trades']}"
        )
        print(f"[DONE] Saved signals to {csv_path}")
    except Exception as e:
        print(f"[ERROR] Failed for {ticker_x}-{ticker_y}: {e}")
