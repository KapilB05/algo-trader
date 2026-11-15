#!/usr/bin/env python3
"""
Retrospective benchmark-relative reality-check of every recorded promotion.

For each version in config/version_history.csv (plus the current live config),
re-simulates the exact parameters on the SAME out-of-sample window the
auto-learner evaluates, and reports alpha vs the intraday naive-long benchmark.

This is the "alpha truth" layer: it distinguishes a real edge (statistically
significant excess return over the market's own open-to-close drift) from
plain beta riding a trending index.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

from src.analysis.benchmark import post_compute_alpha, session_benchmark_returns
from src.backtest.diagnostic_optimizer import precompute_data, simulate_strategy

HIST_PATH = Path("data/historical/banknifty_5min.csv")
VERSION_CSV = Path("config/version_history.csv")
CONFIG_PATH = Path("config/optimized_strategy_config.yaml")
ACTIVE_STRATS = ["ORB", "SUPERTREND", "FLOW"]
MAX_VIX = 24.0


def _sim(params: dict, df_oos: pd.DataFrame):
    return simulate_strategy(
        df=df_oos,
        symbol="BANKNIFTY",
        sl_atr_mult=float(params["sl"]),
        tp_atr_mult=float(params["tp"]),
        min_vix=float(params["vix"]),
        max_vix=MAX_VIX,
        active_strategies=ACTIVE_STRATS,
        consensus_rule=str(params["rule"]),
        label=str(params["version"]),
    )


def main() -> None:
    df_full = precompute_data(HIST_PATH)
    split_idx = int(len(df_full) * 0.75)
    df_oos = df_full.iloc[split_idx:].copy().reset_index(drop=True)
    bench = session_benchmark_returns(df_oos, "NAIVE_LONG")
    bench_cum = (float((1.0 + bench["ret_pct"] / 100.0).prod()) - 1.0) * 100.0

    print("=" * 120)
    print("ALPHA TRUTH — RETROSPECTIVE PROMOTION AUDIT")
    print("=" * 120)
    print(f"OOS window : {df_oos['timestamp'].min()}  ->  {df_oos['timestamp'].max()}")
    print(f"Sessions   : {len(bench)}  (need >= 40 for significance)")
    print(f"BENCHMARK  : NAIVE_LONG (buy open/sell close) cumulative {bench_cum:+.2f}%")
    print("-" * 120)

    rows: list[dict] = []
    if VERSION_CSV.exists():
        with open(VERSION_CSV, encoding="utf-8") as f:
            for rec in csv.DictReader(f):
                try:
                    rows.append({
                        "version": rec["version"],
                        "sl": rec["sl_atr"], "tp": rec["tp_atr"],
                        "vix": rec["vix_gate"], "rule": rec["consensus_rule"],
                        "recorded_sharpe": rec["new_sharpe"],
                    })
                except KeyError:
                    continue

    # Always append the current live config
    import yaml
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        rp = cfg.get("risk_and_execution_parameters", {})
        rvf = cfg.get("regime_and_volatility_filters", {})
        rows.append({
            "version": cfg.get("version", "LIVE"),
            "sl": rp.get("stop_loss_atr_multiplier"), "tp": rp.get("take_profit_atr_multiplier"),
            "vix": rvf.get("min_india_vix_entry"),
            "rule": cfg.get("ensemble_configuration", {}).get("voting_rule", "BENCHMARK"),
            "recorded_sharpe": cfg.get("backtested_performance", {}).get("out_of_sample_sharpe"),
        })

    hdr = (f"{'VERSION':<30} {'SL':>4} {'TP':>4} {'VIX':>5} | {'rec.Sharpe':>9} "
           f"{'reSim.Sharpe':>11} | {'alpha%/yr':>9} {'beta':>5} {'excess%':>8} {'IR':>5} "
           f"{'n':>3} {'beat%':>6} {'sig':>5} | VERDICT")
    print(hdr)
    print("-" * 120)

    for r in rows:
        try:
            res = _sim(r, df_oos)
        except Exception as exc:
            print(f"{r['version']:<30}  SKIP ({exc})")
            continue
        a = post_compute_alpha(res, df_oos, bench_df=bench)
        if a is None:
            print(f"{r['version']:<30}  no aligned daily sessions — cannot attribute alpha")
            continue
        verdict = "REAL EDGE" if a.significant else (
            "BETA/NOISE (not significant)" if a.excess_return_pct > 0 else "UNDERPERFORMS MARKET")
        print(
            f"{r['version']:<30} {r['sl']:>4} {r['tp']:>4} {r['vix']:>5} | "
            f"{r['recorded_sharpe']:>9} {res.sharpe_ratio:>11.2f} | "
            f"{a.alpha_annual_pct:>+9.1f} {a.beta:>5.2f} {a.excess_return_pct:>+8.2f} {a.information_ratio:>5.2f} "
            f"{a.n_sessions:>3} {a.days_beat_pct:>6.1f} {'YES' if a.significant else 'no':>5} | {verdict}"
        )

    print("-" * 120)
    print("Interpretation: a promotion only proves an edge if 'sig' is YES and")
    print("alpha%/yr is positive. Everything else is self-relative beta noise.")


if __name__ == "__main__":
    main()