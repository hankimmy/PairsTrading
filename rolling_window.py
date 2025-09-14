from helper import generate_positions, calculate_returns, fit_rolling_params


class RollingPairsTrader:
    def __init__(self, prices_x, prices_y, entry_threshold=1, exit_threshold=0.2, transaction_cost=0.001):
        self.prices_x = prices_x
        self.prices_y = prices_y
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.transaction_cost = transaction_cost
        self.results = {}

    def backtest(self, window, entry_threshold=None, exit_threshold=None, regime_pval=0.05):
        entry_threshold = entry_threshold or self.entry_threshold
        exit_threshold  = exit_threshold  or self.exit_threshold

        px0, py0 = self.prices_x, self.prices_y
        alphas, betas, means, stds, pvals = fit_rolling_params(px0, py0, window)

        px, py        = px0.iloc[window:], py0.iloc[window:]
        alphas_, betas_ = alphas.copy(), betas.copy()

        spread  = px - (alphas_ + betas_ * py)
        zscore  = (spread - means) / stds
        regime_mask = (pvals < regime_pval)

        positions   = generate_positions(zscore, entry_threshold, exit_threshold)
        positions   = positions.where(regime_mask.reindex_like(positions), other=0)
        entry_thr, exit_thr = entry_threshold, exit_threshold

        pnl, _, cum, sharpe, n_trades = calculate_returns(px, py, positions, betas_, self.transaction_cost, alphas_)
        res = dict({
            "sharpe": sharpe,
            "pnl": pnl,
            "cum_returns_net": cum.iloc[-1] * 100,
            "num_trades": n_trades,
            "entry_threshold": entry_thr,
            "exit_threshold":  exit_thr
        })
        return res
