"""
Side-by-Side Benchmark: Baseline Uncalibrated Single Model vs New Dual-Model Calibrated Purged Ensemble.

Standardized Evaluation Methodology:
- Same synthetic market dataset with dynamic VIX
- Same 13 zero-lookahead features
- Same friction-adjusted labels (Target +2.18%, Stop -1.18%, Friction 0.18%, Window 120)
- Same Purged Walk-Forward Split
- Same ML confidence gate threshold (0.65)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.ml.ensemble import CalibratedEnsemble
from src.ml.feature_engineering import (
    FEATURE_COLS,
    engineer_extended_features,
    generate_friction_adjusted_labels,
)
from src.wfo import PurgedWalkForwardSplitter

logger = logging.getLogger(__name__)


def generate_benchmark_market_data(n_candles: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic market data with intraday trend & dynamic VIX."""
    np.random.seed(seed)
    dates = pd.date_range("2026-01-01 09:15", periods=n_candles, freq="1min")
    returns = np.random.normal(0.0001, 0.0025, n_candles)
    price = 24000.0 * np.exp(np.cumsum(returns))
    high = price * (1 + np.abs(np.random.normal(0, 0.0015, n_candles)))
    low = price * (1 - np.abs(np.random.normal(0, 0.0005, n_candles)))
    open_p = price * (1 + np.random.normal(0, 0.0005, n_candles))

    # Dynamic mean-reverting VIX
    vix_vals = [14.5]
    for _ in range(n_candles - 1):
        v_next = vix_vals[-1] + 0.05 * (15.0 - vix_vals[-1]) + np.random.normal(0, 0.25)
        vix_vals.append(round(float(np.clip(v_next, 10.5, 28.0)), 2))

    return pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": price,
        "volume": np.random.randint(500, 5000, n_candles),
        "vix": vix_vals,
    })


def calc_brier_score(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | float) -> float:
    """Pure numpy Brier Score calculation: mean((y_prob - y_true)^2)."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    return float(np.mean((y_prob_arr - y_true_arr) ** 2))


def calc_metrics(
    probs: np.ndarray | float,
    y_true: pd.Series,
    threshold: float = 0.65,
    target_pct: float = 2.0,
    stop_pct: float = 1.0,
    friction_pct: float = 0.18,
    risk_per_trade: float = 0.02,
) -> dict:
    """Evaluate strategy metrics on out-of-sample data with 2% risk-rule compounded equity curve."""
    probs_arr = np.asarray(probs)
    selected_mask = (probs_arr >= threshold)
    trades = y_true[selected_mask]

    net_reward = target_pct - friction_pct
    net_risk = stop_pct + friction_pct
    win_mult = net_reward / max(1e-6, net_risk)

    if len(trades) == 0:
        return {
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_dd": 0.0,
            "ev": 0.0,
            "brier": calc_brier_score(y_true, probs_arr),
            "n_trades": 0,
            "win_mult": win_mult,
        }

    win_rate = float(trades.mean())
    wins = int(np.sum(trades == 1))
    losses = int(np.sum(trades == 0))

    gross_profit = wins * net_reward
    gross_loss = losses * net_risk
    pf = gross_profit / max(1e-6, gross_loss)

    # 2% Risk Rule compounding (Win: +2% * win_mult, Loss: -2%)
    win_return = risk_per_trade * win_mult
    loss_return = -risk_per_trade
    returns = np.where(trades == 1, win_return, loss_return)

    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd)) * 100 if len(dd) > 0 else 0.0

    ev = (win_rate * net_reward) - ((1 - win_rate) * net_risk)
    brier = calc_brier_score(y_true, probs_arr)

    return {
        "win_rate": win_rate * 100,
        "profit_factor": pf,
        "max_dd": max_dd,
        "ev": ev,
        "brier": brier,
        "n_trades": len(trades),
        "win_mult": win_mult,
    }


def run_benchmark(use_real_data: bool = True, ticker_symbol: str = "^NSEI") -> dict:
    """
    Run OOS benchmark on real historical market data (yfinance) or synthetic data.
    Strictly uses forward_window = 20 candles for 1-minute intraday trade labeling.
    """
    if use_real_data:
        try:
            from src.ml.data_fetcher import fetch_historical_candles_yfinance, fetch_india_vix
            df_pd = fetch_historical_candles_yfinance(ticker_symbol, days=7, interval="1m")
            if not df_pd.empty:
                df_pd["vix"] = fetch_india_vix()
                logger.info("Using real historical %s data (%d candles)", ticker_symbol, len(df_pd))
            else:
                df_pd = generate_benchmark_market_data(3000)
                logger.warning("Real data fetch empty, falling back to synthetic dataset.")
        except Exception as exc:
            logger.warning("Real data fetch failed (%s), falling back to synthetic dataset.", exc)
            df_pd = generate_benchmark_market_data(3000)
    else:
        df_pd = generate_benchmark_market_data(3000)

    is_real = use_real_data and ("^" in ticker_symbol or ".NS" in ticker_symbol)
    target_p = 0.10 if is_real else 2.0
    stop_p = 0.05 if is_real else 1.0
    fric_p = 0.03 if is_real else 0.18

    # Extract features & friction-adjusted labels (strictly forward_window = 20)
    feat_pl = engineer_extended_features(df_pd)
    X = feat_pl.select(FEATURE_COLS).to_pandas()
    X = X.replace([np.inf, -np.inf], np.nan)

    y_arr = generate_friction_adjusted_labels(
        df_pd, target_pct=target_p, stop_pct=stop_p, friction_pct=fric_p, forward_window=20
    )
    y = pd.Series(y_arr)

    # Clean NaNs
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].astype(int).reset_index(drop=True)

    # Purged Walk-Forward Splitter (forward_window = 20)
    splitter = PurgedWalkForwardSplitter(n_splits=5, forward_window=20, embargo_pct=0.02)
    train_idx, test_idx = next(splitter.split(X))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # ── 1. Baseline Model: Uncalibrated Single XGBoost ──
    base_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.03,
        random_state=42,
        verbosity=0,
    )
    base_xgb.fit(X_train.values, y_train.values)
    probs_base = base_xgb.predict_proba(X_test.values)[:, 1]

    # ── 2. New Dual-Model Calibrated Ensemble ──
    ensemble = CalibratedEnsemble(use_lightgbm=True)
    ensemble.fit(X_train.values, y_train.values, feature_names=FEATURE_COLS)
    probs_new = ensemble.predict_proba(X_test.values)

    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
    sweep_results = []

    for th in thresholds:
        m_base = calc_metrics(probs_base, y_test, threshold=th, target_pct=target_p, stop_pct=stop_p, friction_pct=fric_p)
        m_new = calc_metrics(probs_new, y_test, threshold=th, target_pct=target_p, stop_pct=stop_p, friction_pct=fric_p)
        sweep_results.append({
            "threshold": th,
            "base": m_base,
            "ensemble": m_new,
        })

    return {
        "ticker": ticker_symbol,
        "is_real": is_real,
        "n_candles": len(X),
        "pos_label_pct": float(y.mean()) * 100,
        "sweep": sweep_results,
        "probs_base": probs_base,
        "probs_new": probs_new,
        "brier_base": calc_brier_score(y_test, probs_base),
        "brier_new": calc_brier_score(y_test, probs_new),
    }


if __name__ == "__main__":
    for ticker, name in [("^NSEI", "REAL NIFTY 50"), ("^NSEBANK", "REAL BANK NIFTY")]:
        results = run_benchmark(use_real_data=True, ticker_symbol=ticker)
        print("\n" + "=" * 90)
        print(f"      OUT-OF-SAMPLE BENCHMARK REPORT: {name} (forward_window = 20)")
        print("=" * 90)
        print(f"Candles: {results['n_candles']} | Base Win Label %: {results['pos_label_pct']:.2f}%")
        print(f"Brier Score — Baseline: {results['brier_base']:.4f} | New Calibrated Ensemble: {results['brier_new']:.4f}")
        print("-" * 90)
        print(
            f"{'Thresh':<8} | {'Model':<12} | {'Win %':<8} | {'PF':<7} | {'EV %':<8} | {'Max DD %':<10} | {'Signals':<8}"
        )
        print("=" * 90)
        for row in results["sweep"]:
            th = row["threshold"]
            b = row["base"]
            e = row["ensemble"]
            print(
                f"{th:<8.2f} | {'Baseline':<12} | {b['win_rate']:<8.2f} | {b['profit_factor']:<7.2f} | "
                f"{b['ev']:<8.2f} | {b['max_dd']:<10.2f} | {b['n_trades']:<8d}"
            )
            print(
                f"{'':<8} | {'Ensemble':<12} | {e['win_rate']:<8.2f} | {e['profit_factor']:<7.2f} | "
                f"{e['ev']:<8.2f} | {e['max_dd']:<10.2f} | {e['n_trades']:<8d}"
            )
            print("-" * 90)
        print("=" * 90 + "\n")
