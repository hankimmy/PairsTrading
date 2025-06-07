import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from thresholdoptimizer import ThresholdOptimizer
from statsmodels.tsa.stattools import coint
from helper import generate_positions, calculate_returns

def _fit_rolling_params(px, py, window):
    idx = px.index
    alphas, betas, means, stds, pvals = [], [], [], [], []
    for i in range(window, len(px)):
        X, Y = px.iloc[i - window:i], py.iloc[i - window:i]
        reg = sm.OLS(X, sm.add_constant(Y)).fit()
        alpha, beta = reg.params
        spread = X - (alpha + beta * Y)
        alphas.append(alpha)
        betas.append(beta)
        means.append(spread.mean())
        stds.append(spread.std())
        try:
            _, pval, _ = coint(X, Y)
        except Exception:
            pval = 1  # If fail, assume not cointegrated
        pvals.append(pval)
    idx = idx[window:]
    return pd.Series(alphas, idx), pd.Series(betas, idx), pd.Series(means, idx), pd.Series(stds, idx), pd.Series(pvals, idx)

class RollingPairsTrader:
    def __init__(self, prices_x, prices_y, entry_threshold=1, exit_threshold=0.2, transaction_cost=0.001):
        self.prices_x = prices_x
        self.prices_y = prices_y
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.results = {}

    def backtest(self, window, optimize_thresholds=False, optimizer_kwargs=None, entry=1.0, exit= 0.5, regime_pval=0.05):
        px, py = self.prices_x, self.prices_y
        alphas, betas, means, stds, pvals = _fit_rolling_params(px, py, window)
        px, py = px.iloc[window:], py.iloc[window:]
        spread = px - (alphas + betas * py)
        zscore = (spread - means) / stds
        regime_mask = (pvals < regime_pval)

        if optimize_thresholds:
            optimizer = ThresholdOptimizer(**(optimizer_kwargs or {}))
            result = optimizer.find_optimal_thresholds(zscore, px, py, betas, self.transaction_cost, regime_mask=regime_mask)
            metric_key = optimizer.metric
            positions = result['positions']
            entry_thr = result['entry']
            exit_thr = result['exit']
            metric = result[metric_key]
        else:
            positions = generate_positions(zscore, entry, exit)
            # Set positions to 0 when regime is off
            positions = positions.where(regime_mask, other=0)
            entry_thr, exit_thr, metric = self.entry_threshold, self.exit_threshold, None

        out = calculate_returns(px, py, positions, betas, self.transaction_cost)
        out.update({'entry_threshold': entry_thr, 'exit_threshold': exit_thr, 'optimized_metric': metric})
        self.results[f'rolling_{window}_{"opt" if optimize_thresholds else "fixed"}'] = out
        return out

    def static_backtest(self, entry_threshold, exit_threshold):
        reg = sm.OLS(self.prices_x, sm.add_constant(self.prices_y)).fit()
        alpha, beta = reg.params
        spread = self.prices_x - (alpha + beta * self.prices_y)
        mu, sigma = spread.mean(), spread.std()

        zscore = (spread - mu) / sigma
        positions = generate_positions(zscore, entry_threshold, exit_threshold)

        out = calculate_returns(
            self.prices_x, self.prices_y,
            positions,
            pd.Series(beta, index=self.prices_x.index),
            self.transaction_cost,
        )
        out.update({'alpha': alpha, 'beta': beta, 'mu': mu, 'sigma': sigma})
        self.results['static'] = out
        return out

    def static_forecast(self, px_test, py_test,
                                entry=None, exit=None):
        reg   = sm.OLS(self.prices_x, sm.add_constant(self.prices_y)).fit()
        a, b  = reg.params
        mu, s = (self.prices_x - (a + b*self.prices_y)).agg(['mean', 'std'])

        spread  = px_test - (a + b*py_test)
        zscore  = (spread - mu) / s
        posit   = generate_positions(zscore, entry, exit)

        return calculate_returns(px_test, py_test, posit,
                                 pd.Series(b, index=px_test.index),
                                 self.transaction_cost)
