import numpy as np
import pandas as pd
import statsmodels.api as sm
from threshold_search import ThresholdSearch
from helper import generate_positions, calculate_returns, fit_rolling_params

class RollingPairsTrader:
    def __init__(self, prices_x, prices_y, entry_threshold=1, exit_threshold=0.2, transaction_cost=0.001):
        self.prices_x = prices_x
        self.prices_y = prices_y
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.results = {}

    def backtest(self, window,
                 optimize_thresholds=False, optimizer_kwargs=None,
                 entry_threshold=None, exit_threshold=None,
                 regime_pval=0.05):
        entry_threshold = entry_threshold or self.entry_threshold
        exit_threshold  = exit_threshold  or self.exit_threshold

        px0, py0 = self.prices_x, self.prices_y
        alphas, betas, means, stds, pvals = fit_rolling_params(px0, py0, window)

        px, py        = px0.iloc[window:], py0.iloc[window:]
        alphas_, betas_ = alphas.copy(), betas.copy()

        spread  = px - (alphas_ + betas_ * py)
        zscore  = (spread - means) / stds
        regime_mask = (pvals < regime_pval)

        if optimize_thresholds:
            opt  = ThresholdSearch(**(optimizer_kwargs or {}))
            res  = opt.find_optimal_thresholds(zscore, px, py,
                                               betas_, alphas_,
                                               self.transaction_cost,
                                               regime_mask=regime_mask)
            positions = res["positions"]
            entry_thr = res["entry"]
            exit_thr  = res["exit"]
            metric    = res[opt.metric]
        else:
            positions   = generate_positions(zscore, entry_threshold, exit_threshold)
            positions   = positions.where(regime_mask.reindex_like(positions), other=0)
            entry_thr, exit_thr, metric = entry_threshold, exit_threshold, None

        out = calculate_returns(px, py, positions,
                                betas_, self.transaction_cost, alphas_)
        out.update({"entry_threshold": entry_thr,
                    "exit_threshold":  exit_thr,
                    "optimized_metric": metric})

        tag = f"rolling_{window}_{'opt' if optimize_thresholds else 'fixed'}"
        self.results[tag] = out
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
            pd.Series(alpha, index=self.prices_x.index)
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
                                 self.transaction_cost,
                                 pd.Series(a, index=px_test.index))
