import numpy as np
import pandas as pd
from helper import calculate_returns

def test_basic_spread_long():
    px = pd.Series([100, 105], index=pd.date_range("2025-01-01", periods=2))
    py = pd.Series([50,  51],  index=px.index)

    betas = pd.Series([2, 2], index=px.index)
    positions = pd.Series([1, 1], index=px.index)

    tc = 0.0

    out = calculate_returns(px, py, positions, betas, tc)

    # Day-1 P&L should be   +1 * (3 − 0) = +3
    assert out["strategy_returns"].iloc[1] == 3.0
    assert out["cum_pnl"].iloc[-1] == 3.0
    assert out["positions"].iloc[1] == 1