import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

TRADING_DAYS = 252
START_DATE = '2025-04-05'

def load_best_pairs():
    return pd.read_csv('best_pairs.csv')

def load_pair_returns(sector, ticker_x, ticker_y):
    filename = f"{sector}_signals/{ticker_x}_{ticker_y}.csv"
    if not os.path.exists(filename):
        print(f"Warning: File {filename} not found")
        return None
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        df_2025 = df[df['date'] >= START_DATE].copy()
        df_2025 = df_2025.sort_values('date').drop_duplicates(subset='date', keep='first')
        df_2025 = df_2025[df_2025['return'] != 0]
        if df_2025.empty:
            return None
        df_2025['cumulative_return'] = (1 + df_2025['return']).cumprod() - 1
        return df_2025[['date', 'return', 'cumulative_return']]
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

def plot_all_returns(capital_per_pair=10_000):
    best_pairs = load_best_pairs()
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(best_pairs)))
    successful_pairs = 0
    all_returns = []

    for idx, row in best_pairs.iterrows():
        sector, x, y = row['sector'], row['ticker_x'], row['ticker_y']
        data = load_pair_returns(sector, x, y)
        if data is None:
            continue
        dates, ret, cum = data['date'], data['return'], data['cumulative_return']
        plt.plot(dates, cum, color=colors[idx], lw=1.5, alpha=0.8,
                 label=f"{x}_{y} ({sector})")
        successful_pairs += 1
        all_returns.append(pd.Series(data=ret.values, index=dates))

    if all_returns:
        combined = pd.concat(all_returns, axis=1)
        full_idx = pd.date_range(combined.index.min(), combined.index.max(), freq='D')
        combined = combined.reindex(full_idx).fillna(0)
        combined.columns = [f"pair_{i}" for i in range(len(all_returns))]
        daily_ret = combined.mean(axis=1)
        cum_ret = (1 + daily_ret).cumprod() - 1

        sharpe = daily_ret.mean() / daily_ret.std(ddof=0) * np.sqrt(TRADING_DAYS)
        peak = cum_ret.cummax()
        max_dd = (peak - cum_ret).max()

        print(f"Final Return: {cum_ret.iloc[-1]*100:.2f}%")
        print(f"P&L: ${cum_ret.iloc[-1]*capital_per_pair*successful_pairs:,.2f}")
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_dd*100:.2f}%")

        plt.plot(cum_ret.index, cum_ret.values, color='black', lw=2.5, ls='--',
                 label='Portfolio (Equal-Weighted)')
        plt.annotate(f"Sharpe: {sharpe:.2f}\nMax DD: {max_dd*100:.2f}%",
                     xy=(0.02, 0.90), xycoords='axes fraction', fontsize=12,
                     bbox=dict(boxstyle='round,pad=0.3', alpha=0.3))

    plt.title('Cumulative Returns for Best Pairs (2025-04 YTD)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.axhline(0, color='black', ls='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig('returns.png', dpi=300, bbox_inches='tight')
    plt.show()


def print_summary_stats(capital_per_pair=10_000):
    best_pairs = load_best_pairs()
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR BEST PAIRS (2025-04 YTD)")
    print("="*80)
    rows = []
    for _, r in best_pairs.iterrows():
        data = load_pair_returns(r['sector'], r['ticker_x'], r['ticker_y'])
        if data is None:
            continue
        dates, ret, cum = data['date'], data['return'], data['cumulative_return']
        peak = cum.cummax()
        dd = (peak - cum).max()
        rows.append({
            'Pair': f"{r['ticker_x']}_{r['ticker_y']}",
            'Sector': r['sector'],
            'Return %': cum.iloc[-1]*100,
            'Profit $': cum.iloc[-1]*capital_per_pair,
            'Max %': cum.max()*100,
            'Min %': cum.min()*100,
            'Max DD %': dd*100
        })
    df_sum = pd.DataFrame(rows).sort_values('Return %', ascending=False)
    print(df_sum.to_string(index=False, float_format='%.2f'))


if __name__ == '__main__':
    plot_all_returns()
    print_summary_stats()
