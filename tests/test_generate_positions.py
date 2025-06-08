import pandas as pd
from helper import generate_positions

def test_long_entry_and_stop_loss():
    z = pd.Series([0, -1.2, -1.5, -2.0, -3.2, 0])
    pos = generate_positions(z, entry_threshold=1.0, exit_threshold=0.5, stop_z=3.0)
    print("Long stop-loss test:", pos.values)
    assert (pos.values == [0, 1, 1, 1, 0, 0]).all(), "Long stop-loss failed"

def test_short_entry_and_stop_loss():
    z = pd.Series([0, 1.2, 1.5, 2.0, 3.2, 0])
    pos = generate_positions(z, entry_threshold=1.0, exit_threshold=0.5, stop_z=3.0)
    print("Short stop-loss test:", pos.values)
    assert (pos.values == [0, -1, -1, -1, 0, 0]).all(), "Short stop-loss failed"

def test_entry_exit_normal():
    z = pd.Series([0, -1.5, -0.3, 0.2, 1.5, 0.4])
    pos = generate_positions(z, entry_threshold=1.0, exit_threshold=0.5, stop_z=3.0)
    print("Normal entry/exit test:", pos.values)
    assert (pos.values == [0, 1, 0, 0, -1, 0]).all(), "Normal entry/exit failed"

def test_no_trade():
    z = pd.Series([0.2, 0.4, 0.3, -0.2, 0.1, 0.2])
    pos = generate_positions(z, entry_threshold=1.0, exit_threshold=0.5, stop_z=3.0)
    print("No trade test:", pos.values)
    assert (pos.values == [0, 0, 0, 0, 0, 0]).all(), "No trade failed"
