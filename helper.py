import pandas as pd
import numpy as np

def generate_positions(zscore, entry_threshold, exit_threshold):
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
