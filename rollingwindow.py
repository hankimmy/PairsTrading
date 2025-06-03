import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from thresholdoptimizer import ThresholdOptimizer

def _fit_rolling_params(px, py, window):
    idx = px.index
    alphas, betas, means, stds = [], [], [], []
    for i in range(window, len(px)):
        X, Y = px.iloc[i - window:i], py.iloc[i - window:i]
        reg = sm.OLS(X, sm.add_constant(Y)).fit()
        alpha, beta = reg.params
        spread = X - (alpha + beta * Y)
        alphas.append(alpha)
        betas.append(beta)
        means.append(spread.mean())
        stds.append(spread.std())
    idx = idx[window:]
    return pd.Series(alphas, idx), pd.Series(betas, idx), pd.Series(means, idx), pd.Series(stds, idx)


def _calculate_returns(px, py, positions, betas, tc):
    rets_x = px.pct_change().fillna(0)
    rets_y = py.pct_change().fillna(0)
    betas = betas.values

    strategy_returns = []
    for i in range(1, len(positions)):
        beta = betas[i - 1]
        if positions.iloc[i - 1] == 1:
            strat_ret = rets_x.iloc[i] - beta * rets_y.iloc[i]
        elif positions.iloc[i - 1] == -1:
            strat_ret = -rets_x.iloc[i] + beta * rets_y.iloc[i]
        else:
            strat_ret = 0
        strategy_returns.append(strat_ret)
    strategy_returns = pd.Series(strategy_returns, index=px.index[1:])
    position_changes = positions.diff().abs()
    daily_costs = tc * position_changes[1:]
    strategy_returns_net = strategy_returns - daily_costs
    cum_returns = (1 + strategy_returns).cumprod() - 1
    cum_returns_net = (1 + strategy_returns_net).cumprod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    sharpe_net = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
    return dict(
        cum_returns=cum_returns,
        cum_returns_net=cum_returns_net,
        sharpe=sharpe,
        sharpe_net=sharpe_net,
        positions=positions,
        strategy_returns=strategy_returns,
        strategy_returns_net=strategy_returns_net,
    )


class RollingPairsTrader:
    def __init__(self, prices_x, prices_y, entry_threshold=1, exit_threshold=0.2, transaction_cost=0.001):
        self.prices_x = prices_x
        self.prices_y = prices_y
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.results = {}

    def _generate_positions(self, zscore, entry=None, exit=None):
        entry = entry if entry is not None else self.entry_threshold
        exit = exit if exit is not None else self.exit_threshold
        positions, in_trade = [], 0
        for val in zscore:
            if in_trade == 0:
                if val > entry:
                    positions.append(-1); in_trade = -1
                elif val < -entry:
                    positions.append(1); in_trade = 1
                else:
                    positions.append(0)
            elif in_trade == 1:
                if abs(val) < exit:
                    positions.append(0); in_trade = 0
                else:
                    positions.append(1)
            elif in_trade == -1:
                if abs(val) < exit:
                    positions.append(0); in_trade = 0
                else:
                    positions.append(-1)
        return pd.Series(positions, index=zscore.index)

    def backtest(self, window, optimize_thresholds=False, optimizer_kwargs=None):
        px, py = self.prices_x, self.prices_y
        alphas, betas, means, stds = _fit_rolling_params(px, py, window)
        px, py = px.iloc[window:], py.iloc[window:]
        spread = px - (alphas + betas * py)
        zscore = (spread - means) / stds

        if optimize_thresholds:
            optimizer = ThresholdOptimizer(**(optimizer_kwargs or {}))
            result = optimizer.find_optimal_thresholds(zscore, px.pct_change().fillna(0), py.pct_change().fillna(0),
                                                       betas, self.transaction_cost)
            metric_key = optimizer.metric
            positions = result['positions']
            entry_thr = result['entry']
            exit_thr = result['exit']
            metric = result[metric_key]
        else:
            positions = self._generate_positions(zscore)
            entry_thr, exit_thr, metric = self.entry_threshold, self.exit_threshold, None

        out = _calculate_returns(px, py, positions, betas, self.transaction_cost)
        out.update({'entry_threshold': entry_thr, 'exit_threshold': exit_thr, 'optimized_metric': metric})
        self.results[f'rolling_{window}_{"opt" if optimize_thresholds else "fixed"}'] = out
        return out

    def static_backtest(self):
        reg = sm.OLS(self.prices_x, sm.add_constant(self.prices_y)).fit()
        alpha, beta = reg.params
        spread = self.prices_x - (alpha + beta * self.prices_y)
        zscore = (spread - spread.mean()) / spread.std()
        positions = self._generate_positions(zscore)
        out = _calculate_returns(self.prices_x, self.prices_y, positions, pd.Series(beta, index=self.prices_x.index), self.transaction_cost)
        self.results['static'] = out
        return out

    def compare_windows(self, window_sizes, include_static=True, plot=True):
        if include_static:
            res = self.static_backtest()
            print(f"Static: Sharpe={res['sharpe']:.2f}, Sharpe (net)={res['sharpe_net']:.2f}, "
                  f"Total Return={res['cum_returns'].iloc[-1]:.2%}, Total Return (net)={res['cum_returns_net'].iloc[-1]:.2%}")
        for window in window_sizes:
            res = self.backtest(window)
            print(f"Rolling {window}: Sharpe={res['sharpe']:.2f}, Sharpe (net)={res['sharpe_net']:.2f}, "
                  f"Total Return={res['cum_returns'].iloc[-1]:.2%}, Total Return (net)={res['cum_returns_net'].iloc[-1]:.2%}")
        if plot:
            plt.figure(figsize=(14, 7))
            if include_static:
                plt.plot(self.results['static']['cum_returns_net'], label='Static Net')
            for window in window_sizes:
                plt.plot(self.results[f'rolling_{window}_fixed']['cum_returns_net'], label=f'Rolling {window} Net')
            plt.title('Cumulative Returns: Static vs Rolling Window (Net of Tx Costs)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.show()

    def backtest_test_with_rolling(self, train_prices_x, train_prices_y, test_prices_x, test_prices_y, window):
        prices_x_all, prices_y_all = pd.concat([train_prices_x, test_prices_x]), pd.concat([train_prices_y, test_prices_y])
        test_index = test_prices_x.index
        alphas, betas, means, stds = [], [], [], []
        for date in test_index:
            end_loc = prices_x_all.index.get_loc(date)
            start_loc = end_loc - window
            if start_loc < 0:
                alphas.append(np.nan); betas.append(np.nan); means.append(np.nan); stds.append(np.nan); continue
            X_window, Y_window = prices_x_all.iloc[start_loc:end_loc], prices_y_all.iloc[start_loc:end_loc]
            reg = sm.OLS(X_window, sm.add_constant(Y_window)).fit()
            alpha, beta = reg.params
            spread = X_window - (alpha + beta * Y_window)
            alphas.append(alpha); betas.append(beta); means.append(spread.mean()); stds.append(spread.std())
        alphas, betas, means, stds = map(lambda x: pd.Series(x, index=test_index), [alphas, betas, means, stds])
        spread = test_prices_x - (alphas + betas * test_prices_y)
        zscore = (spread - means) / stds
        positions = self._generate_positions(zscore)
        return _calculate_returns(test_prices_x, test_prices_y, positions, betas, self.transaction_cost)

    def compare_test_windows_with_rolling(self, train_prices_x, train_prices_y, test_prices_x, test_prices_y, window_sizes, plot=True):
        test_results = {}
        for window in window_sizes:
            res = self.backtest_test_with_rolling(train_prices_x, train_prices_y, test_prices_x, test_prices_y, window)
            test_results[window] = res
            print(
                f"OOS Rolling {window}: Sharpe={res['sharpe']:.2f}, "
                f"Sharpe (net)={res['sharpe_net']:.2f}, "
                f"Total Return={res['cum_returns'].iloc[-1]:.2%}, "
                f"Total Return (net)={res['cum_returns_net'].iloc[-1]:.2%}"
            )
        if plot:
            plt.figure(figsize=(14,7))
            for window in window_sizes:
                plt.plot(test_results[window]['cum_returns_net'], label=f'Win {window} Net')
            plt.title('Out-of-Sample (Test) Cumulative Returns by Rolling Window Size (Net of Tx Costs)')
            plt.xlabel('Date')
            plt.ylabel('Cumulative Return')
            plt.legend()
            plt.show()
        return test_results
