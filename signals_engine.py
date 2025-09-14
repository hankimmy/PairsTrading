# signals_engine.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class EngineConfig:
    start: str = "2024-01-01"
    window: int = 120
    entry_threshold: float = 1.5
    exit_threshold: float = 0.5
    min_bars: int = 200

class SignalsEngine:
    def __init__(
        self,
        ticker_x: str,
        ticker_y: str,
        csv_path: str,
        start: str = "2024-01-01",
        window: int = 120,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        min_bars: int = 200,
    ) -> None:
        self.ticker_x = ticker_x
        self.ticker_y = ticker_y
        self.csv_path = csv_path
        self.cfg = EngineConfig(
            start=start,
            window=int(window),
            entry_threshold=float(entry_threshold),
            exit_threshold=float(exit_threshold),
            min_bars=int(min_bars),
        )

        self._cached_prices: Optional[Tuple[pd.Series, pd.Series]] = None

    def evaluate(self) -> Dict[str, object]:
        px, py = self._get_prices()
        results = self._walk_forward(px, py, self.cfg.entry_threshold, self.cfg.exit_threshold)

        daily = results["ret"]
        equity = (1.0 + daily).cumprod()
        cum = equity - 1.0

        out = pd.DataFrame({
            "date": results.index.strftime("%Y-%m-%d"),
            "signal": results["signal"].astype(int),
            "entry": np.full(len(results), self.cfg.entry_threshold, dtype=float),
            "exit":  np.full(len(results), self.cfg.exit_threshold, dtype=float),
            "sharpe": results["running_sharpe"],
            "return": daily, # % daily PnL
            "forced_exit": results["forced_exit"].fillna(False).astype(bool),
            "px": results["px"], "py": results["py"],
            "qty_x": results["qty_x"],
            "qty_y": results["qty_y"],
            "order_x": results["order_x"],
            "order_y": results["order_y"],
        })
        out.to_csv(self.csv_path, index=False)

        sharpe = float(self._safe_sharpe(daily))
        num_trades = int((results["entry_flag"] == 1).sum())

        return {
            "cum_returns": cum,
            "daily_returns": daily,
            "sharpe": sharpe,
            "num_trades": num_trades
        }


    def grid_search_thresholds(
        self,
        entry_grid: Iterable[float],
        exit_grid: Iterable[float],
        n_splits: int = 5,
        min_trades: int = 5,
        optimize: str = "sharpe",
        score_weight: float = 1.0,
    ) -> Dict[str, float]:
        px, py = self._get_prices()
        idx = px.index.intersection(py.index)
        w = self.cfg.window

        if len(idx) < max(self.cfg.min_bars, 3 * w):
            raise ValueError("Not enough data for threshold CV. Increase history or reduce window.")

        eps = 0.04
        cand_pairs = [(float(e), float(x)) for e in entry_grid for x in exit_grid if float(x) < float(e) - eps]
        if not cand_pairs:
            raise ValueError("Empty candidate set. Ensure exit < entry for at least one pair.")

        all_results: dict[Tuple[float, float], pd.DataFrame] = {}
        for e, x in cand_pairs:
            all_results[(e, x)] = self._walk_forward(px, py, e, x)[["ret", "entry_flag"]]

        start_pos = np.searchsorted(idx, idx[w])  # ensure indicators warmed
        split_bounds = np.linspace(start_pos + 1, len(idx), num=n_splits + 1, dtype=int)
        agg_val_score = {p: 0.0 for p in cand_pairs}
        for k in range(1, len(split_bounds)):
            val_start = idx[split_bounds[k - 1]]
            val_end = idx[split_bounds[k] - 1]
            train_scores = []
            for p, df in all_results.items():
                train_mask = df.index < val_start
                tr = df.loc[train_mask, "ret"]
                trades = int(df.loc[train_mask, "entry_flag"].sum())
                if trades < min_trades or tr.std(ddof=1) == 0 or tr.empty:
                    score = -np.inf
                else:
                    s = self._safe_sharpe(tr)
                    if optimize == "combo":
                        s = float(s + score_weight * tr.sum())
                    score = float(s)
                train_scores.append((p, score))
            train_scores.sort(key=lambda t: t[1], reverse=True)
            best_p = train_scores[0][0]
            df_best = all_results[best_p]
            val_mask = (df_best.index >= val_start) & (df_best.index <= val_end)
            vr = df_best.loc[val_mask, "ret"]
            if not vr.empty and vr.std(ddof=1) > 0:
                s_val = self._safe_sharpe(vr)
                if optimize == "combo":
                    s_val = float(s_val + score_weight * vr.sum())
            else:
                s_val = -np.inf
            agg_val_score[best_p] += float(s_val)

        winner = max(agg_val_score.items(), key=lambda t: t[1])[0]
        stitched = []
        for k in range(1, len(split_bounds)):
            val_start = idx[split_bounds[k - 1]]
            val_end = idx[split_bounds[k] - 1]
            dfw = all_results[winner]
            vm = (dfw.index >= val_start) & (dfw.index <= val_end)
            stitched.append(dfw.loc[vm, "ret"])
        stitched = pd.concat(stitched) if stitched else pd.Series(dtype=float)
        cv_sharpe = float(self._safe_sharpe(stitched)) if not stitched.empty else float("-inf")

        return {"entry": winner[0], "exit": winner[1], "cv_sharpe": cv_sharpe}

    def _get_prices(self) -> tuple[pd.Series, pd.Series]:
        if self._cached_prices is not None:
            return self._cached_prices
        end = datetime.today().strftime("%Y-%m-%d")
        data = yf.download([self.ticker_x, self.ticker_y], start=self.cfg.start, end=end, auto_adjust=False, progress=False)
        if not isinstance(data.columns, pd.MultiIndex):
            raise ValueError("Unexpected data shape from yfinance. Expected MultiIndex with OHLC fields.")
        adj = data["Adj Close"].copy()
        if self.ticker_x not in adj or self.ticker_y not in adj:
            missing = [t for t in [self.ticker_x, self.ticker_y] if t not in adj]
            raise ValueError(f"Missing tickers in downloaded data: {missing}")
        df = adj[[self.ticker_x, self.ticker_y]].dropna().astype(float)
        df = df.loc[~df.index.duplicated(keep="last")].sort_index()
        px, py = df[self.ticker_x], df[self.ticker_y]
        self._cached_prices = (px, py)
        return px, py

    def _walk_forward(self, px: pd.Series, py: pd.Series, entry: float, exit: float) -> pd.DataFrame:
        w = self.cfg.window
        lx, ly = np.log(px), np.log(py)

        r_mean_x = lx.rolling(w, min_periods=w).mean()
        r_mean_y = ly.rolling(w, min_periods=w).mean()
        r_mean_xy = (lx * ly).rolling(w, min_periods=w).mean()
        r_mean_yy = (ly * ly).rolling(w, min_periods=w).mean()
        cov_xy = r_mean_xy - r_mean_x * r_mean_y
        var_y = r_mean_yy - r_mean_y * r_mean_y
        beta = (cov_xy / var_y).replace([np.inf, -np.inf], np.nan)

        spread = lx - beta * ly
        r_spread_mean = spread.rolling(w, min_periods=w).mean()
        r_spread_std = spread.rolling(w, min_periods=w).std(ddof=1)
        z = (spread - r_spread_mean) / r_spread_std

        signal = np.zeros(len(z), dtype=int)
        entry_flag = np.zeros(len(z), dtype=int)
        forced_exit = np.zeros(len(z), dtype=bool)
        in_pos = 0
        for i in range(len(z)):
            zi = z.iat[i]
            if np.isnan(zi):
                signal[i] = in_pos
                continue
            if in_pos == 0:
                if zi <= -entry:
                    in_pos = 1; entry_flag[i] = 1
                elif zi >= entry:
                    in_pos = -1; entry_flag[i] = 1
            else:
                if abs(zi) <= exit:
                    in_pos = 0
            signal[i] = in_pos

        if signal[-1] != 0:
            signal[-1] = 0
            forced_exit[-1] = True

        dpx = px.pct_change().fillna(0.0)
        dpy = py.pct_change().fillna(0.0)
        beta_lag = beta.shift(1).fillna(0.0)
        sig_lag = pd.Series(signal, index=px.index).shift(1).fillna(0.0)
        notional = 1.0 + beta_lag.abs()
        ret = sig_lag * (dpx - beta_lag * dpy) / notional
        running_sharpe = self._running_sharpe(ret)

        # long spread (signal=+1): +1 share X, -beta_lag shares Y
        # short spread (signal=-1): -1 share X, +beta_lag shares Y
        qty_x = pd.Series(signal, index=px.index).astype(float)               # ±1 or 0
        qty_y = (-pd.Series(signal, index=px.index) * beta_lag).astype(float) # ∓β or 0

        order_x = qty_x.diff().fillna(qty_x)
        order_y = qty_y.diff().fillna(qty_y)

        results = pd.DataFrame({
            "px": px, "py": py, "beta": beta, "spread": spread, "z": z,
            "signal": signal, "entry_flag": entry_flag, "ret": ret,
            "running_sharpe": running_sharpe, "forced_exit": forced_exit,
            "qty_x": qty_x, "qty_y": qty_y,
            "order_x": order_x, "order_y": order_y,
        }).dropna(subset=["px", "py"])

        return results


    def _safe_sharpe(self, daily_returns: pd.Series) -> float:
        r = daily_returns.dropna()
        if r.empty:
            return 0.0
        vol = r.std(ddof=1)
        if vol == 0 or np.isnan(vol):
            return 0.0
        return float(np.sqrt(252) * r.mean() / vol)

    def _running_sharpe(self, daily_returns: pd.Series) -> pd.Series:
        r = daily_returns.fillna(0.0).to_numpy()
        csum = np.cumsum(r)
        csum2 = np.cumsum(r * r)
        n = np.arange(1, len(r) + 1)
        mean = csum / n
        var = np.maximum((csum2 - n * mean * mean) / np.maximum(n - 1, 1), 0.0)
        std = np.sqrt(var)
        sharpe = np.zeros_like(r)
        m = std > 0
        sharpe[m] = np.sqrt(252) * mean[m] / std[m]
        return pd.Series(sharpe, index=daily_returns.index)
