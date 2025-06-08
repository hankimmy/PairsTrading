from statsmodels.tsa.stattools import coint
from signals_engine import SignalsEngine
import warnings
warnings.filterwarnings("ignore")


ticker_x = 'NVDA'
ticker_y = 'TSM'
engine = SignalsEngine(
    ticker_x=ticker_x,
    ticker_y=ticker_y,
    csv_path=f'{ticker_x}_{ticker_y}_signals_log.csv',
    start='2024-01-01',
    end='2025-06-05',
    window=150
)
eval_results = engine.evaluate()
print(
    f"Return={eval_results['cum_returns'].iloc[-1] * 100:.2f}% | "
    f"Sharpe={eval_results['sharpe']:.2f} | "
    f"Trades={eval_results['num_trades']}"
)
