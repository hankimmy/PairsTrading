import yfinance as yf
from rollingwindow import RollingPairsTrader

# Download prices for your chosen period
tickers = ['NVDA', 'TSM']
prices = yf.download(tickers, start='2024-01-01', end='2024-12-31', auto_adjust=False)['Adj Close'].dropna()

trader = RollingPairsTrader(
    prices_x=prices['NVDA'],
    prices_y=prices['TSM'],
    entry_threshold=1,
    exit_threshold=0.2,
    transaction_cost=0.001
)

window_sizes = [30, 60, 90, 120, 180]
trader.compare_windows(window_sizes, include_static=True)
