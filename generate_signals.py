import argparse
import itertools
import dataclasses
import numpy as np
import pandas as pd
from rolling_window import RollingPairsTrader
from signals_engine import SignalsEngine
import yfinance as yf
import os
import sys
from pathlib import Path
from datetime import datetime

parser = argparse.ArgumentParser(description="Generate trading signals for stock pairs in a given sector.")
parser.add_argument("--sector", required=True, help="Sector name (used for folder and summary naming)")
parser.add_argument("--stocks", nargs='+', required=True, help="List of stock tickers")
args = parser.parse_args()

sector = args.sector
stocks = args.stocks

os.makedirs("logs", exist_ok=True)
log_path = f"logs/signal_log_{sector}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
sys.stdout = open(log_path, "w")
sys.stderr = sys.stdout

signal_dir = Path(f"{sector}_signals")
signal_dir.mkdir(exist_ok=True)
summary_dir = Path("summary")
summary_dir.mkdir(exist_ok=True)
summary_path = summary_dir / f"{sector}_summary.csv"

windows_to_try = np.arange(50, 200, 5)

default_entry = 1.5
default_exit = 0.5

entry_grid = np.arange(1.0, 3.05, 0.05)
exit_grid  = np.arange(0.25, 1.25, 0.05)

if summary_path.exists():
    summary_df = pd.read_csv(summary_path)
else:
    summary_df = pd.DataFrame(columns=[
        "Ticker_X", "Ticker_Y", "Best_Window",
        "Sharpe", "Return", "Num_Trades"
    ])
    summary_df.to_csv(summary_path, index=False)

for ticker_x, ticker_y in itertools.combinations(stocks, 2):
    print("\n" + "=" * 80 + "\n")
    print(f"Evaluating pair: {ticker_x}-{ticker_y}")
    csv_path = signal_dir / f"{ticker_x}_{ticker_y}.csv"
    match = summary_df[
        ((summary_df["Ticker_X"] == ticker_x) & (summary_df["Ticker_Y"] == ticker_y)) |
        ((summary_df["Ticker_X"] == ticker_y) & (summary_df["Ticker_Y"] == ticker_x))
    ]
    if not match.empty:
        best_window = int(match.iloc[0]["Best_Window"])
        print(f"[SKIP GRID] {ticker_x}-{ticker_y}: Using saved window {best_window}")
    else:
        window_results = []
        try:
            prices = yf.download(
                [ticker_x, ticker_y],
                start="2024-01-01",
                end=datetime.today().strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False
            )["Adj Close"].dropna()
            trader = RollingPairsTrader(prices_x=prices[ticker_x], prices_y=prices[ticker_y])
        except Exception as e:
            print(f"[ERROR] Failed to fetch data or init trader for {ticker_x}-{ticker_y}: {e}")
            continue

        for window in windows_to_try:
            try:
                res = trader.backtest(window=window, entry_threshold=default_entry, exit_threshold=default_exit)
                sharpe = res["sharpe"]
                ret = res["cum_returns_net"]
                num_trades = res["num_trades"]
                window_results.append((window, sharpe, ret, num_trades))
            except Exception as e:
                print(f"Skipping {ticker_x}-{ticker_y} window={window}: {e}")

        if not window_results:
            continue

        MIN_TRADES = 5
        filtered_results = [r for r in window_results if r[3] >= MIN_TRADES and r[1] > 0]
        if len(filtered_results) < 1:
            print(f"[SKIP] {ticker_x}-{ticker_y} not enough trades or negative sharpes")
            print(filtered_results)
            continue

        def score(sharpe, ret, num_trades):
            penalty = 1.0 if num_trades >= 20 else num_trades / 20.0
            base_score = sharpe + ret
            return base_score * penalty

        top = sorted(filtered_results, key=lambda x: score(x[1], x[2], x[3]), reverse=True)[:3]
        best_window = int(np.round(np.mean([t[0] for t in top])))
        print(f"[INFO] {ticker_x}-{ticker_y}: Top windows: {top} -> best window: {best_window}")

    try:
        engine = SignalsEngine(
            ticker_x=ticker_x,
            ticker_y=ticker_y,
            csv_path=str(csv_path),
            start="2024-01-01",
            window=best_window,
            entry_threshold=default_entry,
            exit_threshold=default_exit
        )

        try:
            best = engine.grid_search_thresholds(
                entry_grid=entry_grid,
                exit_grid=exit_grid,
                n_splits=5,
                min_trades=5,
                optimize="sharpe"
            )
            best_entry, best_exit = best["entry"], best["exit"]
            print(f"[THRESH-CV] {ticker_x}-{ticker_y} entry={best_entry} exit={best_exit} cv_sharpe={best['cv_sharpe']:.2f}")

            engine.cfg = dataclasses.replace(
                engine.cfg,
                entry_threshold=float(best_entry),
                exit_threshold=float(best_exit)
            )
        except Exception as e:
            print(f"[WARN] Threshold CV failed for {ticker_x}-{ticker_y}. Using defaults. Reason: {e}")

        eval_results = engine.evaluate()
        total_return_pct = eval_results["cum_returns"].iloc[-1] * 100.0
        print(f"Return={total_return_pct:.2f}% | Sharpe={eval_results['sharpe']:.2f} | Trades={eval_results['num_trades']}")
        print(f"[DONE] Saved signals to {csv_path}")

        new_row = {
            "Ticker_X": ticker_x,
            "Ticker_Y": ticker_y,
            "Best_Window": best_window,
            "Sharpe": round(eval_results["sharpe"], 4),
            "Return": round(total_return_pct, 4),
            "Num_Trades": eval_results["num_trades"],
        }

        summary_df = pd.concat([summary_df, pd.DataFrame([new_row])], ignore_index=True)
        summary_df.drop_duplicates(subset=["Ticker_X", "Ticker_Y"], keep="last", inplace=True)
        summary_df.to_csv(summary_path, index=False)

    except Exception as e:
        print(f"[ERROR] SignalsEngine failed for {ticker_x}-{ticker_y}: {e}")
