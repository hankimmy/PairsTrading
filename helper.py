import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm


def generate_positions(zscore, entry_threshold, exit_threshold, stop_z = 3.0):
    positions = []
    in_trade = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if in_trade == 0:
            if z > entry_threshold:
                positions.append(-1)
                in_trade = -1
            elif z < -entry_threshold:
                positions.append(1)
                in_trade = 1
            else:
                positions.append(0)
        elif in_trade == 1:
            if abs(z) < exit_threshold:
                positions.append(0)
                in_trade = 0
            elif z < -stop_z:
                positions.append(0)
                in_trade = 0
            else:
                positions.append(1)
        elif in_trade == -1:
            if abs(z) < exit_threshold:
                positions.append(0)
                in_trade = 0
            elif z > stop_z:
                positions.append(0)
                in_trade = 0
            else:
                positions.append(-1)

    return pd.Series(positions, index=zscore.index)


def calculate_returns(
    px: pd.Series,
    py: pd.Series,
    positions: pd.Series,
    betas: pd.Series,
    tc: float,
    alphas: pd.Series,
    notional_per_trade: float = 10_000.0,
):
    df = pd.concat(
        {"px": px, "py": py, "pos": positions, "beta": betas, "alpha": alphas},
        axis=1,
    ).dropna()
    df = df.sort_index()

    prev_pos = df["pos"].shift(fill_value=0)
    entry_mask = (prev_pos == 0) & (df["pos"] != 0)
    exit_mask  = (prev_pos != 0) & (df["pos"] == 0)

    entry_idx = df.index[entry_mask]
    exit_idx  = df.index[exit_mask]

    pnl = pd.Series(0.0, index=df.index)
    n_trades = 0

    for t_entry in entry_idx:
        future_exits = exit_idx[exit_idx > t_entry]
        if len(future_exits) == 0:
            continue
        t_exit = future_exits[0]

        sign = int(df.loc[t_entry, "pos"])  # +1 long spread, -1 short spread
        beta_entry = float(df.loc[t_entry, "beta"])

        px_e = float(df.loc[t_entry, "px"])
        py_e = float(df.loc[t_entry, "py"])
        px_x = float(df.loc[t_exit,  "px"])
        py_x = float(df.loc[t_exit,  "py"])

        gross_per_unit_entry = abs(px_e) + abs(beta_entry) * abs(py_e)
        if gross_per_unit_entry == 0:
            continue  # degenerate guard

        u = min(notional_per_trade / gross_per_unit_entry, 1.0)
        qx =  sign * u
        qy = -sign * beta_entry * u

        leg_pnl = qx * (px_x - px_e) + qy * (py_x - py_e)

        entry_turnover = abs(qx) * px_e + abs(qy) * py_e
        exit_turnover  = abs(qx) * px_x + abs(qy) * py_x
        tc_cost = tc * (entry_turnover + exit_turnover)

        realized = leg_pnl - tc_cost

        pnl.loc[t_exit] += realized
        n_trades += 1

    returns = pnl / float(notional_per_trade)
    cum_returns = returns.cumsum()

    nz = returns[returns != 0.0]
    if len(nz) >= 2 and nz.std(ddof=1) > 0:
        sharpe = nz.mean() / nz.std(ddof=1)
    else:
        sharpe = float("nan")

    return pnl, returns, cum_returns, sharpe, n_trades

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