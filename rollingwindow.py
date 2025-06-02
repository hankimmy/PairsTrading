import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class RollingPairsTrader:
    def __init__(self, prices_x, prices_y, entry_threshold=1, exit_threshold=0.2, transaction_cost=0.001):
        self.prices_x = prices_x
        self.prices_y = prices_y
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.results = {}  # To store results for each window size and static

    def rolling_params(self, window):
        alphas, betas, means, stds = [], [], [], []
        idx = self.prices_x.index
        for i in range(window, len(self.prices_x)):
            X = self.prices_x.iloc[i - window:i]
            Y = self.prices_y.iloc[i - window:i]
            reg = sm.OLS(X, sm.add_constant(Y)).fit()
            alpha, beta = reg.params
            spread = X - (alpha + beta * Y)
            alphas.append(alpha)
            betas.append(beta)
            means.append(spread.mean())
            stds.append(spread.std())
        return (
            pd.Series(alphas, index=idx[window:]),
            pd.Series(betas, index=idx[window:]),
            pd.Series(means, index=idx[window:]),
            pd.Series(stds, index=idx[window:])
        )

    def generate_positions(self, zscore):
        positions = []
        in_trade = 0
        for i in range(len(zscore)):
            if in_trade == 0:
                if zscore.iloc[i] > self.entry_threshold:
                    positions.append(-1)
                    in_trade = -1
                elif zscore.iloc[i] < -self.entry_threshold:
                    positions.append(1)
                    in_trade = 1
                else:
                    positions.append(0)
            elif in_trade == 1:
                if abs(zscore.iloc[i]) < self.exit_threshold:
                    positions.append(0)
                    in_trade = 0
                else:
                    positions.append(1)
            elif in_trade == -1:
                if abs(zscore.iloc[i]) < self.exit_threshold:
                    positions.append(0)
                    in_trade = 0
                else:
                    positions.append(-1)
        return pd.Series(positions, index=zscore.index)

    def backtest(self, window):
        alphas, betas, means, stds = self.rolling_params(window)
        px = self.prices_x.iloc[window:]
        py = self.prices_y.iloc[window:]
        rets_x = px.pct_change().fillna(0)
        rets_y = py.pct_change().fillna(0)
        betas_used = betas.values
        spread = px - (alphas + betas * py)
        zscore = (spread - means) / stds
        positions = self.generate_positions(zscore)
        strategy_returns = []
        for i in range(1, len(positions)):
            if positions.iloc[i - 1] == 1:
                strat_ret = rets_x.iloc[i] - betas_used[i] * rets_y.iloc[i]
            elif positions.iloc[i - 1] == -1:
                strat_ret = -rets_x.iloc[i] + betas_used[i] * rets_y.iloc[i]
            else:
                strat_ret = 0
            strategy_returns.append(strat_ret)
        strategy_returns = pd.Series(strategy_returns, index=px.index[1:])
        position_changes = positions.diff().abs()
        daily_costs = self.transaction_cost * position_changes[1:]
        strategy_returns_net = strategy_returns - daily_costs
        cum_returns = (1 + strategy_returns).cumprod() - 1
        cum_returns_net = (1 + strategy_returns_net).cumprod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        sharpe_net = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
        # Save results
        self.results[f'rolling_{window}'] = {
            'cum_returns': cum_returns,
            'cum_returns_net': cum_returns_net,
            'sharpe': sharpe,
            'sharpe_net': sharpe_net,
            'positions': positions,
            'strategy_returns': strategy_returns,
            'strategy_returns_net': strategy_returns_net
        }
        return self.results[f'rolling_{window}']

    def static_backtest(self):
        # Static regression and params
        reg = sm.OLS(self.prices_x, sm.add_constant(self.prices_y)).fit()
        alpha, beta = reg.params
        spread = self.prices_x - (alpha + beta * self.prices_y)
        spread_mean = spread.mean()
        spread_std = spread.std()
        zscore = (spread - spread_mean) / spread_std
        positions = self.generate_positions(zscore)
        rets_x = self.prices_x.pct_change().fillna(0)
        rets_y = self.prices_y.pct_change().fillna(0)
        strategy_returns = []
        for i in range(1, len(positions)):
            if positions.iloc[i - 1] == 1:
                strat_ret = rets_x.iloc[i] - beta * rets_y.iloc[i]
            elif positions.iloc[i - 1] == -1:
                strat_ret = -rets_x.iloc[i] + beta * rets_y.iloc[i]
            else:
                strat_ret = 0
            strategy_returns.append(strat_ret)
        strategy_returns = pd.Series(strategy_returns, index=self.prices_x.index[1:])
        position_changes = positions.diff().abs()
        daily_costs = self.transaction_cost * position_changes[1:]
        strategy_returns_net = strategy_returns - daily_costs
        cum_returns = (1 + strategy_returns).cumprod() - 1
        cum_returns_net = (1 + strategy_returns_net).cumprod() - 1
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        sharpe_net = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
        self.results['static'] = {
            'cum_returns': cum_returns,
            'cum_returns_net': cum_returns_net,
            'sharpe': sharpe,
            'sharpe_net': sharpe_net,
            'positions': positions,
            'strategy_returns': strategy_returns,
            'strategy_returns_net': strategy_returns_net
        }
        return self.results['static']

    def compare_windows(self, window_sizes, include_static=True, plot=True):
        if include_static:
            self.static_backtest()
            res = self.results['static']
            print(f"Static: Sharpe={res['sharpe']:.2f}, Sharpe (net)={res['sharpe_net']:.2f}, "
                  f"Total Return={res['cum_returns'].iloc[-1]:.2%}, Total Return (net)={res['cum_returns_net'].iloc[-1]:.2%}")
        for window in window_sizes:
            self.backtest(window)
            res = self.results[f'rolling_{window}']
            print(f"Rolling {window}: Sharpe={res['sharpe']:.2f}, Sharpe (net)={res['sharpe_net']:.2f}, "
                  f"Total Return={res['cum_returns'].iloc[-1]:.2%}, Total Return (net)={res['cum_returns_net'].iloc[-1]:.2%}")
        if plot:
            plt.figure(figsize=(14, 7))
            if include_static:
                plt.plot(self.results['static']['cum_returns_net'], label='Static Net')
            for window in window_sizes:
                plt.plot(self.results[f'rolling_{window}']['cum_returns_net'], label=f'Rolling {window} Net')
            plt.title('Cumulative Returns: Static vs Rolling Window (Net of Tx Costs)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.show()
