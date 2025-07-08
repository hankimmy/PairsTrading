import numpy as np
import pandas as pd
import pytest
from helper import calculate_returns

@pytest.fixture
def synthetic_data():
    dates = pd.date_range(start="2024-01-01", periods=5)
    px = pd.Series([100, 102, 101, 103, 102], index=dates)
    py = pd.Series([50, 51, 50, 52, 51], index=dates)
    betas = pd.Series([1.0] * 5, index=dates)
    positions = pd.Series([0, 1, 1, -1, -1], index=dates)
    return px, py, positions, betas

def test_returns_no_errors(synthetic_data):
    px, py, positions, betas = synthetic_data
    result = calculate_returns(px, py, positions, betas, tc=0.001)
    assert isinstance(result, dict)
    for key in ['strategy_pnl', 'strategy_pnl_net', 'cum_returns_net']:
        assert key in result
        assert isinstance(result[key], pd.Series)

def test_returns_nonzero_trade(synthetic_data):
    dates = pd.date_range("2024-01-01", periods=5)
    px = pd.Series([100, 102, 101, 104, 106], index=dates)
    py = pd.Series([50, 51, 50.5, 51.5, 52], index=dates)
    positions = pd.Series([0, 1, 1, 1, 0], index=dates)
    betas = pd.Series([2.0] * 5, index=dates)

    result = calculate_returns(px, py, positions, betas, tc=0.001)

    assert not np.allclose(result['strategy_pnl'].values, 0), "Expected non-zero PnL"
    assert not np.allclose(result['strategy_pnl_net'].values, 0), "Expected non-zero net PnL"
    assert result['cum_pnl_net'].iloc[-1] != 0, "Cumulative net PnL should not be zero"

def test_zero_positions():
    dates = pd.date_range(start="2024-01-01", periods=5)
    px = pd.Series([100, 100, 100, 100, 100], index=dates)
    py = pd.Series([50, 50, 50, 50, 50], index=dates)
    betas = pd.Series([1.0] * 5, index=dates)
    positions = pd.Series([0] * 5, index=dates)

    result = calculate_returns(px, py, positions, betas, tc=0.001)
    assert np.allclose(result['strategy_pnl'], 0)
    assert np.allclose(result['cum_returns_net'], 0)

def test_constant_prices_with_position():
    dates = pd.date_range(start="2024-01-01", periods=5)
    px = pd.Series([100] * 5, index=dates)
    py = pd.Series([50] * 5, index=dates)
    betas = pd.Series([1.0] * 5, index=dates)
    positions = pd.Series([1] * 5, index=dates)

    result = calculate_returns(px, py, positions, betas, tc=0.001)
    assert np.allclose(result['strategy_pnl'], 0)
    assert np.allclose(result['cum_returns_net'], 0)

def test_return_scaling():
    dates = pd.date_range(start="2024-01-01", periods=5)
    px = pd.Series([100, 105, 110, 100, 95], index=dates)
    py = pd.Series([50, 51, 52, 53, 52], index=dates)
    betas = pd.Series([1.0] * 5, index=dates)
    positions = pd.Series([0, 1, 1, -1, -1], index=dates)

    result = calculate_returns(px, py, positions, betas, tc=0.001, notional_per_trade=10000)
    assert result['cum_pnl_net'].iloc[-1] != 0
    assert result['cum_returns_net'].iloc[-1] != 0
