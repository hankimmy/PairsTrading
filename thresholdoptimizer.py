import numpy as np
import pandas as pd
from itertools import product

class ThresholdOptimizer:
    def __init__(
        self,
        metric='sharpe',
        entry_grid=np.arange(0.5, 2.51, 0.05),
        exit_grid=np.arange(0.05, 1.01, 0.05),
        verbose=False
    ):
        self.best_result = None
        self.metric = metric
        self.entry_grid = entry_grid
        self.exit_grid = exit_grid
        self.verbose = verbose

    def generate_positions(self, zscore, entry_threshold, exit_threshold):
        positions = []
        in_trade = 0
        for i in range(len(zscore)):
            if in_trade == 0:
                if zscore.iloc[i] > entry_threshold:
                    positions.append(-1)
                    in_trade = -1
                elif zscore.iloc[i] < -entry_threshold:
                    positions.append(1)
                    in_trade = 1
                else:
                    positions.append(0)
            elif in_trade == 1:
                if abs(zscore.iloc[i]) < exit_threshold:
                    positions.append(0)
                    in_trade = 0
                else:
                    positions.append(1)
            elif in_trade == -1:
                if abs(zscore.iloc[i]) < exit_threshold:
                    positions.append(0)
                    in_trade = 0
                else:
                    positions.append(-1)
        return pd.Series(positions, index=zscore.index)

    def evaluate(self, positions, rets_x, rets_y, betas, transaction_cost=0.001):
        strategy_returns = []
        betas = betas if isinstance(betas, pd.Series) else pd.Series(betas, index=rets_x.index)
        for i in range(1, len(positions)):
            if positions.iloc[i - 1] == 1:
                strat_ret = rets_x.iloc[i] - betas.iloc[i] * rets_y.iloc[i]
            elif positions.iloc[i - 1] == -1:
                strat_ret = -rets_x.iloc[i] + betas.iloc[i] * rets_y.iloc[i]
            else:
                strat_ret = 0
            strategy_returns.append(strat_ret)
        strategy_returns = pd.Series(strategy_returns, index=rets_x.index[1:])
        position_changes = positions.diff().abs()
        daily_costs = transaction_cost * position_changes[1:]
        strategy_returns_net = strategy_returns - daily_costs

        if strategy_returns_net.std() == 0 or strategy_returns_net.isnull().all():
            return -np.inf

        if self.metric == 'sharpe':
            return strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
        elif self.metric == 'return':
            return ((1 + strategy_returns_net).cumprod() - 1).iloc[-1]
        else:
            raise ValueError('Unknown metric')

    def find_optimal_thresholds(self, zscore, rets_x, rets_y, betas, transaction_cost=0.001):
        best_value = -np.inf
        best_entry, best_exit = None, None
        best_positions = None
        grid = []

        for entry, exit_ in product(self.entry_grid, self.exit_grid):
            if exit_ >= entry or np.isclose(exit_, entry):
                continue
            positions = self.generate_positions(zscore, entry, exit_)
            metric_value = self.evaluate(positions, rets_x, rets_y, betas, transaction_cost)
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
