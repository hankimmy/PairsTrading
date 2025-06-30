import os
import logging
from pathlib import Path
from typing   import List

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from helper import generate_positions, calculate_returns

logging.getLogger("yfinance").setLevel(logging.CRITICAL)

class SignalsEngine:
    def __init__(
        self,
        ticker_x: str,
        ticker_y: str,
        csv_path: str | Path,
        *,
        start: str = "2023-01-01",
        end: str | None = None,
        entry_grid: np.ndarray = np.arange(0.5, 2.51, 0.05),
        exit_grid:  np.ndarray = np.arange(0.05, 1.01, 0.05),
        window: int = 100,
        tc: float = 0.001,
        regime_pval: float = 0.05,
        stop_z: float = 3.0,
    ):
        self.ticker_x, self.ticker_y = ticker_x, ticker_y
        self.csv_path   = str(csv_path)
        self.entry_grid = entry_grid
        self.exit_grid  = exit_grid
        self.window     = window
        self.tc         = tc
        self.regime_pval = regime_pval
        self.stop_z      = stop_z

        self._load_prices(start, end)
        self.signals_df: pd.DataFrame | None = None
        self.process()

    def _load_prices(self, start: str, end: str | None):
        tickers = [self.ticker_x, self.ticker_y]
        px = yf.download(
            tickers, start=start, end=end or None,
            auto_adjust=False, progress=False
        )["Adj Close"].dropna()

        if not all(t in px.columns for t in tickers):
            raise ValueError(f"Could not download both tickers {tickers}")

        self.prices_x = px[self.ticker_x]
        self.prices_y = px[self.ticker_y]

    def log_signal_row(self, row: dict):
        """Append one row to the CSV (create if absent)."""
        file_exists = os.path.exists(self.csv_path)
        pd.DataFrame([row]).to_csv(
            self.csv_path, mode="a", header=not file_exists, index=False
        )

    def reload_signals_from_csv(self):
        self.signals_df = (
            pd.read_csv(self.csv_path, parse_dates=["date"])
              .set_index("date")
              .sort_index()
        )

    def process(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_path):
            print("[SignalsEngine] No CSV found → running full walk-forward…")
            signals = self.walkforward_gridsearch()
            signals.reset_index().to_csv(self.csv_path, index=False)
        self.reload_signals_from_csv()
        self.update_with_new_price()          # pulls any missing days
        self.reload_signals_from_csv()
        return self.signals_df

    def evaluate(self):
        if self.signals_df is None:
            raise RuntimeError("No signals yet – run process() first.")

        df = self.signals_df.copy()
        cum_ret = (1 + df["return"]).cumprod() - 1
        sharpe  = (
            df["return"].mean() / df["return"].std() * np.sqrt(252)
            if df["return"].std() > 0 else 0
        )
        n_trades = (df["signal"].diff().abs() > 0).sum()
        return {"cum_returns": cum_ret, "sharpe": sharpe, "num_trades": n_trades}

    def log_signal_row(self, row):
        file_exists = os.path.exists(self.csv_path)
        pd.DataFrame([row]).to_csv(self.csv_path, mode='a', header=not file_exists, index=False)

    def reload_signals_from_csv(self):
        self.signals_df = pd.read_csv(self.csv_path, parse_dates=['date']).set_index('date')

    def evaluate(self):
        if self.signals_df is None:
            raise RuntimeError("Run walkforward_gridsearch() first!")
        df = self.signals_df
        cum_returns = (1 + df['return']).cumprod() - 1
        sharpe = (
            df['return'].mean() / df['return'].std() * np.sqrt(252)
            if df['return'].std() > 0 else 0
        )

        num_trades = (df['signal'].diff().abs() > 0).sum()
        return {'cum_returns': cum_returns, 'sharpe': sharpe, 'num_trades': num_trades}

    def compute_regime_mask(self, px: pd.Series, py: pd.Series, window: int):
        pvals: List[float] = []
        for i in range(window, len(px)):
            try:
                _, pval, _ = coint(px.iloc[i - window:i], py.iloc[i - window:i])
            except Exception:
                pval = 1.0
            pvals.append(pval)
        return (pd.Series(pvals, index=px.index[window:]) < self.regime_pval)

    def walkforward_gridsearch(self) -> pd.DataFrame:
        px, py = self.prices_x, self.prices_y
        regime_mask = self.compute_regime_mask(px, py, self.window)

        rows: list[dict] = []
        for j in range(self.window, len(px) - 1):
            prev_sig = rows[-1]["signal"] if rows else 0
            if not regime_mask.iloc[j - self.window]:
                row = self._generate_exit_row(px, py, j, prev_sig, forced_exit=True)
            else:
                row = self._generate_signal_row(px, py, j, prev_sig)
            rows.append(row)
            self.log_signal_row(row)

        return pd.DataFrame(rows).set_index("date")

    def update_with_new_price(self):
        self.reload_signals_from_csv()
        last_date = self.signals_df.index[-1]
        print(f"[update] last logged date: {last_date.date()}")

        today   = pd.Timestamp.today().normalize()
        new_px  = yf.download(
            [self.ticker_x, self.ticker_y],
            start=(last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            end  =(today + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=False, progress=False,
        )["Adj Close"].dropna()

        if new_px.empty:
            print("[update] nothing new to add.")
            return

        self.prices_x = pd.concat([self.prices_x, new_px[self.ticker_x]]).sort_index()
        self.prices_y = pd.concat([self.prices_y, new_px[self.ticker_y]]).sort_index()

        all_dates    = self.prices_x.index
        missing_idx  = all_dates[all_dates > last_date]
        regime_mask  = self.compute_regime_mask(self.prices_x, self.prices_y, self.window)

        rows_added = 0
        for d in missing_idx:
            i = all_dates.get_loc(d)
            if isinstance(i, slice):          # safety for pandas >=2.2
                i = i.start
            if i is None or i < self.window:
                continue

            prev_sig = self.signals_df.iloc[-1]["signal"]
            prev_ok  = regime_mask.iloc[i - self.window]
            if not prev_ok:
                row = self._generate_exit_row(self.prices_x, self.prices_y, i - 1, prev_sig, forced_exit=True)
            else:
                row = self._generate_signal_row(self.prices_x, self.prices_y, i - 1, prev_sig)

            self.log_signal_row(row)
            rows_added += 1

        if rows_added:
            print(f"[update] appended {rows_added} new rows.")

    @staticmethod
    def _bar_pnl(pos, alpha, beta, px_t, py_t, px_t1, py_t1):
        eps_t  = px_t  - (alpha + beta * py_t)
        eps_t1 = px_t1 - (alpha + beta * py_t1)
        return pos * (eps_t1 - eps_t)

    def _generate_exit_row(self, px, py, i, prev_signal, forced_exit=False):
        date = px.index[i + 1]
        if prev_signal == 0:
            return {
                'date': date, 'signal': 0, 'entry': np.nan, 'exit': np.nan,
                'sharpe': np.nan, 'return': 0, 'forced_exit': False,
                'px': px.iloc[i + 1], 'py': py.iloc[i + 1]
            }

        alpha, beta = sm.OLS(px.iloc[i - self.window + 1:i + 1],
                             sm.add_constant(py.iloc[i - self.window + 1:i + 1])).fit().params

        bar_pnl  = self._bar_pnl(prev_signal, alpha, beta,
                                 px.iloc[i],   py.iloc[i],
                                 px.iloc[i+1], py.iloc[i+1])

        gross_notional = abs(px.iloc[i]) + abs(beta * py.iloc[i])
        tc_cost = 2 * self.tc * gross_notional if prev_signal != 0 else 0
        bar_pnl_net = bar_pnl - tc_cost

        pct_ret = bar_pnl_net / gross_notional if gross_notional else 0.0
        return {
            "date": date,
            "signal": 0,
            "entry": np.nan,
            "exit":  np.nan,
            "sharpe": np.nan,
            "return": pct_ret,
            "forced_exit": forced_exit,
            "px": px.iloc[i + 1],
            "py": py.iloc[i + 1],
        }

    def _generate_signal_row(self, px, py, i, prev_signal):
        w_px = px.iloc[i - self.window:i]
        w_py = py.iloc[i - self.window:i]
        alpha, beta = sm.OLS(w_px, sm.add_constant(w_py)).fit().params

        spread = w_px - (alpha + beta * w_py)
        z_win  = (spread - spread.mean()) / spread.std()

        best_sharpe = -np.inf
        best_e, best_x = None, None
        for e in self.entry_grid:
            for x in self.exit_grid:
                if x >= e:
                    continue
                pos = generate_positions(z_win, e, x, self.stop_z)
                stats = calculate_returns(
                    w_px, w_py, pos,
                    pd.Series(beta, index=w_px.index),
                    self.tc,
                    pd.Series(alpha, index=w_px.index),
                )
                if stats["sharpe_net"] > best_sharpe:
                    best_sharpe, best_e, best_x = stats["sharpe_net"], e, x

        z_now = z_win.iloc[-1]
        if z_now > best_e:
            signal = -1
        elif z_now < -best_e:
            signal = 1
        elif abs(z_now) < best_x:
            signal = 0
        else:
            signal = prev_signal

        bar_pnl = self._bar_pnl(prev_signal, alpha, beta,
                                px.iloc[i],   py.iloc[i],
                                px.iloc[i+1], py.iloc[i+1])

        gross_notional = abs(px.iloc[i]) + abs(beta * py.iloc[i])
        if signal != prev_signal:
            bar_pnl -= 2 * self.tc * gross_notional

        pct_ret = bar_pnl / gross_notional if gross_notional else 0.0
        return {
            "date": px.index[i + 1],
            "signal": signal,
            "entry":  best_e,
            "exit":   best_x,
            "sharpe": best_sharpe,
            "return": pct_ret,
            "forced_exit": False,
            "px": px.iloc[i + 1],
            "py": py.iloc[i + 1],
        }
