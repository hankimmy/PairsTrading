import pandas as pd
from statsmodels.tsa.stattools import coint
import heapq

def find_top_k_coint_windows(px, py, k=5, min_days=60, max_days=None, freq=1):
    """
    px, py: pd.Series (with DateTime index)
    k: number of top windows to keep (lowest p-values)
    min_days: minimum window size for searching
    max_days: maximum window size (defaults to full length)
    freq: step size in days for window sliding (1=every day)
    Returns: pd.DataFrame of top k windows by p-value
    """
    if max_days is None:
        max_days = len(px)
    dates = px.index
    n = len(dates)
    heap = []
    
    for start in range(0, n - min_days + 1, freq):
        for end in range(start + min_days, min(start + max_days, n) + 1, freq):
            px_window = px.iloc[start:end]
            py_window = py.iloc[start:end]
            try:
                _, pval, _ = coint(px_window, py_window)
            except Exception:
                continue

            result = {
                'start': dates[start],
                'end': dates[end-1],
                'n_days': end-start,
                'pval': pval
            }
            if len(heap) < k:
                heapq.heappush(heap, (-pval, result))
            else:
                if pval < -heap[0][0]:
                    heapq.heappushpop(heap, (-pval, result))

    top_results = [item for (_, item) in heap]
    top_results.sort(key=lambda x: x['pval'])
    return pd.DataFrame(top_results)

def is_coint_significant(px, py):
    """
    px, py: pd.Series (with DateTime index)
    Returns: True if p-val is significant
    """
    try:
        _, pval, _ = coint(px, py)
    except Exception:
        return False

    return True if pval < 0.05 else False
