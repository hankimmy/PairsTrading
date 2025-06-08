from rolling_window import RollingPairsTrader
import yfinance as yf

tickers = ['NVDA', 'TSM']
prices = yf.download(tickers, start='2024-01-01', end='2025-06-08', auto_adjust=False)['Adj Close'].dropna()

trader = RollingPairsTrader(prices_x=prices['NVDA'], prices_y=prices['TSM'])

window_candidates = np.arange(70, 300, 5)
window_results = []

for window in window_candidates:
    res = trader.backtest(window)
    window_results.append({
        'window': window,
        'sharpe': res['sharpe_net'],
        'return': res['cum_returns_net'].iloc[-1],
        'entry': res['entry_threshold'],
        'exit': res['exit_threshold'],
        'num_trades': (res['positions'].diff().abs() > 0).sum()
    })
    print(
        f"Window={window:2d} | Entry={res['entry_threshold']:.2f} | "
        f"Exit={res['exit_threshold']:.2f} | "
        f"Return={res['cum_returns_net'].iloc[-1] * 100:.2f}% | "
        f"Sharpe={res['sharpe_net']:.2f} | "
        f"Trades={np.sum(res['positions'].diff().abs() > 0)}"
    )

top5 = sorted(window_results, key=lambda x: x['sharpe'], reverse=True)[:5]
print("\n[TOP 5 WINDOWS BY SHARPE]")
for r in top5:
    print(f"Window={r['window']} | Sharpe={r['sharpe']:.2f} | Return={r['return'] * 100:.2f}% | Entry={r['entry']:.2f} | Exit={r['exit']:.2f} | Trades={r['num_trades']}")
