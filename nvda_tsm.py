import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

stock_y = 'TSM'
stock_x = 'NVDA'
tickers = [stock_x, stock_y]

prices_all = yf.download(tickers, start='2024-01-01', end='2025-12-31', auto_adjust=False)['Adj Close'].dropna()
prices_2024 = prices_all.loc[:'2024-12-31']
prices_2025 = prices_all.loc['2025-01-01':]

print("\n=== Parameter Fit: 2024 ===")
score, pvalue, _ = coint(prices_2024[stock_x], prices_2024[stock_y])
print(f"Cointegration p-value (2024): {pvalue:.4f}")
if pvalue > 0.05:
    print("Warning: These stocks are likely not cointegrated in this window.")

Y = prices_2024[stock_y]
X = prices_2024[stock_x]
X_const = sm.add_constant(Y)
model = sm.OLS(X, X_const)
result = model.fit()
alpha = result.params[0]
beta = result.params[1]
print(f"Hedge ratio (beta): {beta:.4f}")
print(f"Intercept (alpha): {alpha:.4f}")

entry_threshold = 1
exit_threshold = 0.2
transaction_cost = 0.001

spread_2024 = prices_2024[stock_x] - (alpha + beta * prices_2024[stock_y])
spread_mean_2024 = spread_2024.mean()
spread_std_2024 = spread_2024.std()
zscore_2024 = (spread_2024 - spread_mean_2024) / spread_std_2024

positions_2024 = []
in_trade = 0
for i in range(len(zscore_2024)):
    if in_trade == 0:
        if zscore_2024.iloc[i] > entry_threshold:
            positions_2024.append(-1)
            in_trade = -1
        elif zscore_2024.iloc[i] < -entry_threshold:
            positions_2024.append(1)
            in_trade = 1
        else:
            positions_2024.append(0)
    elif in_trade == 1:
        if abs(zscore_2024.iloc[i]) < exit_threshold:
            positions_2024.append(0)
            in_trade = 0
        else:
            positions_2024.append(1)
    elif in_trade == -1:
        if abs(zscore_2024.iloc[i]) < exit_threshold:
            positions_2024.append(0)
            in_trade = 0
        else:
            positions_2024.append(-1)
positions_2024 = pd.Series(positions_2024, index=zscore_2024.index)

ret_NVDA_2024 = prices_2024[stock_x].pct_change().fillna(0)
ret_TSM_2024 = prices_2024[stock_y].pct_change().fillna(0)
strategy_returns_2024 = []
for i in range(1, len(positions_2024)):
    if positions_2024.iloc[i-1] == 1:
        strat_ret = ret_NVDA_2024.iloc[i] - beta * ret_TSM_2024.iloc[i]
    elif positions_2024.iloc[i-1] == -1:
        strat_ret = -ret_NVDA_2024.iloc[i] + beta * ret_TSM_2024.iloc[i]
    else:
        strat_ret = 0
    strategy_returns_2024.append(strat_ret)
strategy_returns_2024 = pd.Series(strategy_returns_2024, index=prices_2024.index[1:])

position_changes_2024 = positions_2024.diff().abs()
daily_costs_2024 = transaction_cost * position_changes_2024[1:]
strategy_returns_net_2024 = strategy_returns_2024 - daily_costs_2024

cum_returns_2024 = (1 + strategy_returns_2024).cumprod() - 1
cum_returns_net_2024 = (1 + strategy_returns_net_2024).cumprod() - 1
total_return_2024 = cum_returns_2024.iloc[-1]
sharpe_2024 = strategy_returns_2024.mean() / strategy_returns_2024.std() * np.sqrt(252)
total_return_net_2024 = cum_returns_net_2024.iloc[-1]
sharpe_net_2024 = strategy_returns_net_2024.mean() / strategy_returns_net_2024.std() * np.sqrt(252)

print("\n=== 2024 In-Sample Results ===")
print(f"Total return: {total_return_2024:.2%}")
print(f"Annualized Sharpe ratio: {sharpe_2024:.2f}")
print(f"Total return (net): {total_return_net_2024:.2%}")
print(f"Annualized Sharpe (net): {sharpe_net_2024:.2f}")

# === Out-of-sample backtest on 2025 ===
spread_2025 = prices_2025[stock_x] - (alpha + beta * prices_2025[stock_y])
spread_mean_2025 = spread_2025.mean()
spread_std_2025 = spread_2025.std()
zscore_2025 = (spread_2025 - spread_mean_2025) / spread_std_2025

positions_2025 = []
in_trade = 0
for i in range(len(zscore_2025)):
    if in_trade == 0:
        if zscore_2025.iloc[i] > entry_threshold:
            positions_2025.append(-1)
            in_trade = -1
        elif zscore_2025.iloc[i] < -entry_threshold:
            positions_2025.append(1)
            in_trade = 1
        else:
            positions_2025.append(0)
    elif in_trade == 1:
        if abs(zscore_2025.iloc[i]) < exit_threshold:
            positions_2025.append(0)
            in_trade = 0
        else:
            positions_2025.append(1)
    elif in_trade == -1:
        if abs(zscore_2025.iloc[i]) < exit_threshold:
            positions_2025.append(0)
            in_trade = 0
        else:
            positions_2025.append(-1)
positions_2025 = pd.Series(positions_2025, index=zscore_2025.index)

ret_NVDA_2025 = prices_2025[stock_x].pct_change().fillna(0)
ret_TSM_2025 = prices_2025[stock_y].pct_change().fillna(0)
strategy_returns_2025 = []
for i in range(1, len(positions_2025)):
    if positions_2025.iloc[i-1] == 1:
        strat_ret = ret_NVDA_2025.iloc[i] - beta * ret_TSM_2025.iloc[i]
    elif positions_2025.iloc[i-1] == -1:
        strat_ret = -ret_NVDA_2025.iloc[i] + beta * ret_TSM_2025.iloc[i]
    else:
        strat_ret = 0
    strategy_returns_2025.append(strat_ret)
strategy_returns_2025 = pd.Series(strategy_returns_2025, index=prices_2025.index[1:])

position_changes_2025 = positions_2025.diff().abs()
daily_costs_2025 = transaction_cost * position_changes_2025[1:]
strategy_returns_net_2025 = strategy_returns_2025 - daily_costs_2025

cum_returns_2025 = (1 + strategy_returns_2025).cumprod() - 1
cum_returns_net_2025 = (1 + strategy_returns_net_2025).cumprod() - 1
total_return_2025 = cum_returns_2025.iloc[-1]
sharpe_2025 = strategy_returns_2025.mean() / strategy_returns_2025.std() * np.sqrt(252)
total_return_net_2025 = cum_returns_net_2025.iloc[-1]
sharpe_net_2025 = strategy_returns_net_2025.mean() / strategy_returns_net_2025.std() * np.sqrt(252)

print("\n=== 2025 Out-of-Sample Results ===")
print(f"Total return: {total_return_2025:.2%}")
print(f"Annualized Sharpe ratio: {sharpe_2025:.2f}")
print(f"Total return (net): {total_return_net_2025:.2%}")
print(f"Annualized Sharpe (net): {sharpe_net_2025:.2f}")

plt.figure(figsize=(14, 6))
plt.plot(cum_returns_2024, label='2024 No Tx Cost')
plt.plot(cum_returns_net_2024, label='2024 With Tx Cost')
plt.plot(cum_returns_2025, label='2025 No Tx Cost')
plt.plot(cum_returns_net_2025, label='2025 With Tx Cost')
plt.axhline(0, color='k', linestyle=':')
plt.title('Cumulative Return: 2024 (In-sample) vs 2025 (Out-of-sample)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()
