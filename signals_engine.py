import numpy as np
import pandas as pd
import yfinance as yf
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from helper import generate_positions, calculate_returns

class SignalsEngine:
    def __init__(self, ticker_x, ticker_y, csv_path, start='2023-01-01', end=None,
                 entry_grid=np.arange(0.5, 2.51, 0.05), exit_grid=np.arange(0.05, 1.01, 0.05),
                 window=100, tc=0.001, regime_pval=0.05, stop_z=3.0):
        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.csv_path = csv_path
        self.entry_grid = entry_grid
        self.exit_grid = exit_grid
        self.window = window
        self.tc = tc
        self.regime_pval = regime_pval
        self.stop_z = stop_z
        self.signals_df = None

        self._load_prices(start, end)
        self.process()

    def _load_prices(self, start, end):
        tickers = [self.ticker_x, self.ticker_y]
        print(f"[SignalsEngine] Downloading prices for {tickers} from {start} to {end or 'today'}...")
        px = yf.download(tickers, start=start, end=end, auto_adjust=False)['Adj Close'].dropna()
        if not all(t in px.columns for t in tickers):
            raise ValueError(f"Could not download both tickers {tickers}: only got {list(px.columns)}")
        self.prices_x = px[self.ticker_x]
        self.prices_y = px[self.ticker_y]

    def process(self):
        if not os.path.exists(self.csv_path):
            print("[SignalsEngine] CSV not found, running full walkforward grid search...")
            signals = self.walkforward_gridsearch()
            signals.reset_index().to_csv(self.csv_path, index=False)
            self.signals_df = signals
        else:
            self.reload_signals_from_csv()

        self.update_with_new_price()
        self.reload_signals_from_csv()
        return self.signals_df

    def compute_regime_mask(self, prices_x, prices_y, window):
        pvals = []
        for i in range(window, len(prices_x)):
            try:
                _, pval, _ = coint(prices_x.iloc[i-window:i], prices_y.iloc[i-window:i])
            except Exception:
                pval = 1
            pvals.append(pval)
        return pd.Series(pvals, index=prices_x.index[window:]) < self.regime_pval

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
        sharpe = df['return'].mean() / df['return'].std() * np.sqrt(252)
        num_trades = (df['signal'].diff().abs() > 0).sum()
        return {'cum_returns': cum_returns, 'sharpe': sharpe, 'num_trades': num_trades}

    def walkforward_gridsearch(self):
        prices_x, prices_y = self.prices_x, self.prices_y
        regime_mask = self.compute_regime_mask(prices_x, prices_y, self.window)
        results = []

        for j in range(self.window, len(prices_x) - 1):
            prev_signal = results[-1]['signal'] if results else 0
            date = prices_x.index[j + 1]

            if not regime_mask.iloc[j - self.window]:
                row = self._generate_exit_row(prices_x, prices_y, j, prev_signal, forced_exit=True)
                results.append(row)
                self.log_signal_row(row)
                continue

            row = self._generate_signal_row(prices_x, prices_y, j, prev_signal)
            results.append(row)
            self.log_signal_row(row)

        self.signals_df = pd.DataFrame(results).set_index('date')
        return self.signals_df

    def update_with_new_price(self):
        self.reload_signals_from_csv()
        last_date = self.signals_df.index[-1]
        print(f"[update_with_new_price] Last recorded signal: {last_date.date()}")

        today = pd.Timestamp.today().normalize()
        end_date = today + pd.Timedelta(days=1)
        new_px = yf.download([self.ticker_x, self.ticker_y],
                             start=(last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),
                             end=end_date.strftime('%Y-%m-%d'),
                             auto_adjust=False)['Adj Close'].dropna()

        if new_px.empty:
            print("[update_with_new_price] No new price data available.")
            return

        self.prices_x = pd.concat([self.prices_x, new_px[self.ticker_x]]).sort_index()
        self.prices_y = pd.concat([self.prices_y, new_px[self.ticker_y]]).sort_index()

        all_dates = self.prices_x.index
        missing_dates = all_dates[all_dates > last_date]
        regime_mask = self.compute_regime_mask(self.prices_x, self.prices_y, self.window)
        results = []

        for date in missing_dates:
            try:
                i = all_dates.get_loc(date)
                if i < self.window:
                    continue
                prev_signal = self.signals_df.iloc[-1]['signal'] if len(self.signals_df) > 0 else 0
                prev_date = all_dates[i - 1]

                if not regime_mask.loc[prev_date]:
                    row = self._generate_exit_row(self.prices_x, self.prices_y, i - 1, prev_signal, forced_exit=True)
                else:
                    row = self._generate_signal_row(self.prices_x, self.prices_y, i - 1, prev_signal)

                results.append(row)
                self.log_signal_row(row)
            except Exception as e:
                print(f"[update_with_new_price] Skipped {date.date()}: {e}")

        if results:
            print(f"[update_with_new_price] Added {len(results)} rows.")

    def _generate_exit_row(self, px, py, i, prev_signal, forced_exit=False):
        date = px.index[i + 1]
        if prev_signal == 0:
            return {
                'date': date, 'signal': 0, 'entry': np.nan, 'exit': np.nan,
                'sharpe': np.nan, 'return': 0, 'forced_exit': False,
                'px': px.iloc[i + 1], 'py': py.iloc[i + 1]
            }

        window_px = px.iloc[i - self.window + 1:i + 1]
        window_py = py.iloc[i - self.window + 1:i + 1]
        reg = sm.OLS(window_px, sm.add_constant(window_py)).fit()
        _, beta = reg.params

        ret_x = px.iloc[i + 1] / px.iloc[i] - 1
        ret_y = py.iloc[i + 1] / py.iloc[i] - 1
        trade_ret = (ret_x - beta * ret_y) if prev_signal == 1 else (-ret_x + beta * ret_y)
        trade_ret -= self.tc

        return {
            'date': date, 'signal': 0, 'entry': np.nan, 'exit': np.nan,
            'sharpe': np.nan, 'return': trade_ret, 'forced_exit': forced_exit,
            'px': px.iloc[i + 1], 'py': py.iloc[i + 1]
        }

    def _generate_signal_row(self, px, py, i, prev_signal):
        window_px = px.iloc[i - self.window:i]
        window_py = py.iloc[i - self.window:i]
        reg = sm.OLS(window_px, sm.add_constant(window_py)).fit()
        alpha, beta = reg.params
        spread = window_px - (alpha + beta * window_py)
        z_win = (spread - spread.mean()) / spread.std()

        best_sharpe = -np.inf
        best_entry, best_exit = None, None
        for entry in self.entry_grid:
            for exit in self.exit_grid:
                if exit >= entry:
                    continue
                pos = generate_positions(z_win, entry, exit, self.stop_z)
                ret_dict = calculate_returns(window_px, window_py, pos, pd.Series(beta, index=window_px.index), self.tc)
                if ret_dict['sharpe_net'] > best_sharpe:
                    best_sharpe = ret_dict['sharpe_net']
                    best_entry, best_exit = entry, exit

        zscore = z_win.iloc[-1]
        if zscore > best_entry:
            signal = -1
        elif zscore < -best_entry:
            signal = 1
        elif abs(zscore) < best_exit:
            signal = 0
        else:
            signal = prev_signal

        ret_x = px.iloc[i + 1] / px.iloc[i] - 1
        ret_y = py.iloc[i + 1] / py.iloc[i] - 1
        trade_ret = 0
        if signal == 1:
            trade_ret = ret_x - beta * ret_y
        elif signal == -1:
            trade_ret = -ret_x + beta * ret_y
        trade_ret -= self.tc * (signal != prev_signal)

        return {
            'date': px.index[i + 1], 'signal': signal, 'entry': best_entry, 'exit': best_exit,
            'sharpe': best_sharpe, 'return': trade_ret, 'forced_exit': False,
            'px': px.iloc[i + 1], 'py': py.iloc[i + 1]
        }
