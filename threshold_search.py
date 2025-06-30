import numpy as np
import pandas as pd
from itertools import product
from helper import generate_positions, calculate_returns, fit_rolling_params

class ThresholdSearch:
    def __init__(
        self,
        metric='return',
        entry_grid=np.arange(0.5, 2.51, 0.05),
        exit_grid=np.arange(0.05, 1.01, 0.05),
        verbose=False
    ):
        self.best_result = None
        self.metric = metric
        self.entry_grid = entry_grid
        self.exit_grid = exit_grid
        self.verbose = verbose

    def evaluate(self, positions, px, py, betas, alphas, transaction_cost=0.001):
        result = calculate_returns(px, py, positions, betas, transaction_cost, alphas)

        daily = result["daily_return_net"]
        if daily.std() == 0 or daily.isnull().all():
            return -np.inf

        if self.metric == "sharpe":
            return result["sharpe_net"]
        return result["cum_returns_net"].iloc[-1]

    def find_optimal_thresholds(self, zscore, px, py, betas, alphas, transaction_cost=0.001, regime_mask=None):
        best_value = -np.inf
        best_entry, best_exit = None, None
        best_positions = None
        grid = []

        for entry, exit_ in product(self.entry_grid, self.exit_grid):
            if exit_ + 0.2 >= entry or np.isclose(exit_, entry):
                continue
            positions = generate_positions(zscore, entry, exit_)
            if regime_mask is not None:
                positions = positions.where(regime_mask, other=0)
            metric_value = self.evaluate(positions, px, py, betas, alphas, transaction_cost)
            grid.append({'entry': entry, 'exit': exit_, self.metric: metric_value})
            if self.verbose:
                print(f"Entry={entry:.2f}, Exit={exit_:.2f}, {self.metric.capitalize()}={metric_value:.4f}")
            if metric_value > best_value:
                best_value = metric_value
                best_entry = entry
                best_exit = exit_
                best_positions = positions

        if best_positions is None:
            print(f"No valid set of thresholds found (all grid points returned -inf). Returning flat positions.")
            best_entry = np.nan
            best_exit = np.nan
            best_positions = pd.Series(0, index=zscore.index)
            best_value = -np.inf

        self.best_result = {
            'entry': best_entry,
            'exit': best_exit,
            'positions': best_positions,
            self.metric: best_value,
            'grid': grid
        }
        return self.best_result
