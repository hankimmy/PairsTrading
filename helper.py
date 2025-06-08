import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def generate_positions(zscore, entry_threshold, exit_threshold, stop_z = 3.0):
    """
    stop_z: z-score level for stop-loss (e.g. 3.0 = exit if z-score moves 3 stddevs against the trade)
    """
    positions = []
    in_trade = 0
    entry_side = 0 

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if in_trade == 0:
            if z > entry_threshold:
                positions.append(-1)
                in_trade = -1
                entry_side = -1
            elif z < -entry_threshold:
                positions.append(1)
                in_trade = 1
                entry_side = 1
            else:
                positions.append(0)
        elif in_trade == 1:
            if abs(z) < exit_threshold:
                positions.append(0)
                in_trade = 0
                entry_side = 0
            elif z < -stop_z:
                positions.append(0)
                in_trade = 0
                entry_side = 0
            else:
                positions.append(1)
        elif in_trade == -1:
            if abs(z) < exit_threshold:
                positions.append(0)
                in_trade = 0
                entry_side = 0
            elif z > stop_z:
                positions.append(0)
                in_trade = 0
                entry_side = 0
            else:
                positions.append(-1)

    return pd.Series(positions, index=zscore.index)


def calculate_returns(px, py, positions, betas, tc):
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

def fit_rolling_params(px, py, window):
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
            pval = 1
        pvals.append(pval)
    idx = idx[window:]
    return pd.Series(alphas, idx), pd.Series(betas, idx), pd.Series(means, idx), pd.Series(stds, idx), pd.Series(pvals, idx)
