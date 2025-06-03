import numpy as np
import yfinance as yf
from rollingwindow import RollingPairsTrader
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings("ignore")

tickers = ['NVDA', 'TSM']
prices_all = yf.download(tickers, start='2024-01-01', end='2025-05-02', auto_adjust=False)['Adj Close'].dropna()
_, pvalue_2025, _ = coint(prices_all['NVDA'].loc['2025-01-01':], prices_all['TSM'].loc['2025-01-01':])
print(f"pvalue: {pvalue_2025:.4f}")

train_prices = prices_all.loc['2024-01-01':]
test_prices = prices_all.loc['2024-10-01':]

trader = RollingPairsTrader(prices_x=train_prices['NVDA'], prices_y=train_prices['TSM'])

window_candidates = np.arange(20, 101, 5)
window_results = []

for window in window_candidates:
    res = trader.backtest(window, optimize_thresholds=True, optimizer_kwargs={
        'metric': 'sharpe',
        'entry_grid': np.arange(0.5, 2.6, 0.2),
        'exit_grid': np.arange(0.05, 1.05, 0.1),
        'verbose': False,
    })
    window_results.append({
        'window': window,
        'sharpe': res['sharpe_net'],
        'return': res['cum_returns_net'].iloc[-1],
        'entry': res['entry_threshold'],
        'exit': res['exit_threshold'],
    })
    print(
        f"[SCAN] Window={window:2d} | Entry={res['entry_threshold']:.2f} | "
        f"Exit={res['exit_threshold']:.2f} | "
        f"Return={res['cum_returns_net'].iloc[-1] * 100:.2f}% | "
        f"Sharpe={res['sharpe_net']:.2f}"
    )

top5 = sorted(window_results, key=lambda x: x['sharpe'], reverse=True)[:5]
print("\n[TOP 5 WINDOWS BY SHARPE]")
for r in top5:
    print(f"Window={r['window']} | Sharpe={r['sharpe']:.2f} | Return={r['return'] * 100:.2f}% | Entry={r['entry']:.2f} | Exit={r['exit']:.2f}")

print("\n[GRID SEARCH IN TOP 5 WINDOWS]")
for r in top5:
    win = r['window']
    res_fine = trader.backtest(win, optimize_thresholds=True, optimizer_kwargs={
        'metric': 'sharpe',
        'entry_grid': np.arange(0.5, 2.51, 0.05),
        'exit_grid': np.arange(0.05, 1.01, 0.05),
        'verbose': False,
    })
    print(
        f"Window={win:2d} | Entry={res_fine['entry_threshold']:.2f} | "
        f"Exit={res_fine['exit_threshold']:.2f} | "
        f"Return={res_fine['cum_returns_net'].iloc[-1] * 100:.2f}% | "
        f"Sharpe={res_fine['sharpe_net']:.2f}"
    )

test_prices_x = test_prices['NVDA']
test_prices_y = test_prices['TSM']

for params in top5:
    trader_test = RollingPairsTrader(prices_x=test_prices_x, prices_y=test_prices_y,
                                     entry_threshold=params['entry'],
                                     exit_threshold=params['exit'],
                                     transaction_cost=0.001)

    res = trader_test.backtest(params['window'], optimize_thresholds=False)
    total_return = res['cum_returns_net'].iloc[-1]
    sharpe = res['sharpe_net']
    print(f"[OOS] Window={params['window']} | Entry={params['entry']} | Exit={params['exit']} | "
          f"Return={total_return * 100:.2f}% | Sharpe={sharpe:.2f}")
