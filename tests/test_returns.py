import numpy as np
import pandas as pd
import pytest
from helper import calculate_returns


def test_calculate_returns_basic():
    px = pd.Series([100, 101, 102, 103, 104])
    py = pd.Series([100, 100, 100, 100, 100])
    positions = pd.Series([1, 1, 1, 1, 1])
    betas = pd.Series([0, 0, 0, 0, 0])
    result = calculate_returns(px, py, positions, betas, tc=0)

    expected_returns = px.pct_change().fillna(0)[1:]
    expected_cum = (1 + expected_returns).cumprod() - 1
    expected_sharpe = expected_returns.mean() / expected_returns.std() * np.sqrt(252)

    assert np.allclose(result['cum_returns'].values, expected_cum.values)
    assert np.isclose(result['sharpe'], expected_sharpe)
