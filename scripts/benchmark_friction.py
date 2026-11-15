"""
Friction Sensitivity Benchmark Script.

Standardized Evaluation Methodology across friction levels:
- 0.10% (Low friction / baseline assumption)
- 0.18% (Default realistic trading cost assumption)
- 0.25% (High friction / slippage stress test)
"""

from __future__ import annotations

import logging

import pandas as pd

from src.ml.benchmark_oos import calc_metrics, generate_benchmark_market_data
from src.ml.ensemble import CalibratedEnsemble
from src.ml.feature_engineering import (
    FEATURE_COLS,
    engineer_extended_features,
    generate_friction_adjusted_labels,
)
from src.wfo import PurgedWalkForwardSplitter

logger = logging.getLogger(__name__)


def run_friction_sensitivity():
    df_pd = generate_benchmark_market_data(3000)
    feat_pl = engineer_extended_features(df_pd)
    X = feat_pl.select(FEATURE_COLS).to_pandas()

    frictions = [0.10, 0.18, 0.25]
    results = {}

    for f in frictions:
        y_arr = generate_friction_adjusted_labels(
            df_pd, target_pct=2.0, stop_pct=1.0, friction_pct=f, forward_window=120
        )
        y = pd.Series(y_arr)

        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[valid_idx].reset_index(drop=True)
        y_clean = y.loc[valid_idx].astype(int).reset_index(drop=True)

        splitter = PurgedWalkForwardSplitter(n_splits=5, forward_window=20, embargo_pct=0.02)
        train_idx, test_idx = next(splitter.split(X_clean))

        X_train, y_train = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
        X_test, y_test = X_clean.iloc[test_idx], y_clean.iloc[test_idx]

        ensemble = CalibratedEnsemble(use_lightgbm=True)
        ensemble.fit(X_train.values, y_train.values, feature_names=FEATURE_COLS)
        probs = ensemble.predict_proba(X_test.values)

        metrics = calc_metrics(probs, y_test, threshold=0.60, friction_pct=f)

        net_reward = 2.0 - f
        net_risk = 1.0 + f

        results[f] = {
            "friction_pct": f,
            "net_reward_pct": net_reward,
            "net_risk_pct": net_risk,
            "positive_label_pct": float(y_clean.mean()) * 100,
            "oos_win_rate": metrics["win_rate"],
            "profit_factor": metrics["profit_factor"],
            "ev": metrics["ev"],
            "max_dd": metrics["max_dd"],
            "brier": metrics["brier"],
            "n_signals": metrics["n_trades"],
        }

    return results


if __name__ == "__main__":
    res = run_friction_sensitivity()
    print("\n" + "=" * 90)
    print("           FRICTION SENSITIVITY BENCHMARK REPORT (Threshold = 0.60)")
    print("=" * 90)
    print(f"{'Friction %':<12} | {'Net Rew %':<10} | {'Net Risk %':<10} | {'Win Label %':<12} | {'OOS Win %':<10} | {'PF':<7} | {'EV %':<8} | {'Signals':<8}")
    print("-" * 90)
    for f, metrics in res.items():
        print(
            f"{metrics['friction_pct']:<12.2f} | "
            f"+{metrics['net_reward_pct']:<9.2f} | "
            f"-{metrics['net_risk_pct']:<9.2f} | "
            f"{metrics['positive_label_pct']:<12.2f} | "
            f"{metrics['oos_win_rate']:<10.2f} | "
            f"{metrics['profit_factor']:<7.2f} | "
            f"{metrics['ev']:<8.2f} | "
            f"{metrics['n_signals']:<8d}"
        )
    print("=" * 90 + "\n")
