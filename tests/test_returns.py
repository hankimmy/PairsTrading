import math
import numpy as np
import pandas as pd
import pytest

from helper import calculate_returns  # update this import path


def _s(vals):
    idx = pd.date_range("2024-01-01", periods=len(vals), freq="D")
    return pd.Series(vals, index=idx)


def test_single_long_trade_no_cap():
    px = _s([100, 102, 103, 104])      # X
    py = _s([50, 51, 52, 53])          # Y
    beta = _s([0.5, 0.5, 0.5, 0.5])
    alpha = _s([0, 0, 0, 0])
    pos = _s([0, 1, 1, 0])              # enter day 1, exit day 3
    tc = 0.001
    notional = 10_000.0

    t_entry = px.index[1]
    t_exit = px.index[3]
    px_e, py_e = px.loc[t_entry], py.loc[t_entry]
    px_x, py_x = px.loc[t_exit], py.loc[t_exit]
    beta_e = beta.loc[t_entry]

    u = 1.0
    qx = 1 * u
    qy = -beta_e * u

    leg_pnl = qx * (px_x - px_e) + qy * (py_x - py_e)
    entry_turnover = abs(qx) * px_e + abs(qy) * py_e
    exit_turnover = abs(qx) * px_x + abs(qy) * py_x
    tc_cost = tc * (entry_turnover + exit_turnover)
    expected = leg_pnl - tc_cost

    pnl, rets, cum, sharpe = calculate_returns(px, py, pos, beta, tc, alpha, notional)

    assert pnl.loc[t_exit] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert pnl.drop(t_exit).sum() == 0.0
    assert rets.loc[t_exit] == pytest.approx(expected / notional, rel=1e-12, abs=1e-12)


def test_single_short_trade_with_cap():
    px = _s([8_000, 8_010, 8_020])
    py = _s([9_000, 9_010, 9_020])
    beta = _s([1.2, 1.2, 1.2])
    alpha = _s([0, 0, 0])
    pos = _s([0, -1, 0])                # enter day 1, exit day 2
    tc = 0.002
    notional = 10_000.0

    t_entry = px.index[1]
    t_exit = px.index[2]
    px_e, py_e = px.loc[t_entry], py.loc[t_entry]
    px_x, py_x = px.loc[t_exit], py.loc[t_exit]
    beta_e = beta.loc[t_entry]

    gross_per_unit_entry = abs(px_e) + abs(beta_e) * abs(py_e)
    assert gross_per_unit_entry > notional
    u = notional / gross_per_unit_entry

    qx = -1 * u
    qy = beta_e * u

    leg_pnl = qx * (px_x - px_e) + qy * (py_x - py_e)
    entry_turnover = abs(qx) * px_e + abs(qy) * py_e
    exit_turnover = abs(qx) * px_x + abs(qy) * py_x
    tc_cost = tc * (entry_turnover + exit_turnover)
    expected = leg_pnl - tc_cost

    pnl, rets, cum, sharpe = calculate_returns(px, py, pos, beta, tc, alpha, notional)

    assert u < 1.0
    assert pnl.loc[t_exit] == pytest.approx(expected, rel=1e-12, abs=1e-6)
    assert (pnl != 0).sum() == 1 and pnl.index[pnl != 0][0] == t_exit


def test_multiple_trades_and_open_trade_ignored():
    # Two trades with the second left open at the end
    px = _s([100, 101, 102, 103, 104, 105])
    py = _s([50, 51, 52, 53, 54, 55])
    beta = _s([1, 1, 1, 1, 1, 1])
    alpha = _s([0, 0, 0, 0, 0, 0])
    pos = _s([0, 1, 1, 0, -1, -1])       # trade 1 completed, trade 2 open
    tc = 0.001
    notional = 10_000.0

    pnl, rets, cum, sharpe = calculate_returns(px, py, pos, beta, tc, alpha, notional)

    assert (pnl != 0).sum() == 1
    t_exit1 = px.index[3]
    assert pnl.loc[t_exit1] != 0.0
    assert pnl.loc[px.index[-1]] == 0.0


def test_sharpe_uses_only_non_zero_trade_returns():
    px = _s([100, 101, 102, 103, 104, 105, 106])
    py = _s([50, 49, 48, 47, 46, 45, 44])
    beta = _s([0.5] * 7)
    alpha = _s([0] * 7)
    pos = _s([0, 1, 1, 0, -1, -1, 0])
    tc = 0.0
    notional = 10_000.0

    pnl, rets, cum, sharpe = calculate_returns(px, py, pos, beta, tc, alpha, notional)

    nz = rets[rets != 0]
    mean = nz.mean()
    std = nz.std(ddof=1)
    expected = mean / std if std > 0 else float("nan")

    if math.isnan(expected):
        assert math.isnan(sharpe)
    else:
        assert sharpe == pytest.approx(expected, rel=1e-12)
