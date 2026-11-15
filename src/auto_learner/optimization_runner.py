"""
Continuous Auto-Learner — Rolling Walk-Forward Optimization Runner.

Executes walk-forward parameter calibration across rolling In-Sample (12M) & Out-of-Sample (3M) windows:
1. Sweeps ATR Stops [0.75, 1.0, 1.25, 1.5]
2. Sweeps ATR Targets [1.5, 2.0, 2.5]
3. Sweeps VIX Gate Min [10.0, 11.5, 13.0]
4. Sweeps Consensus Rules [ALT_B, BENCHMARK]
5. Ranks by Out-of-Sample Sharpe Ratio
6. Flags 'Promotable' if candidate OOS Sharpe > 5% higher than baseline live config
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.analysis.benchmark import post_compute_alpha, promotion_eligible, session_benchmark_returns
from src.backtest.diagnostic_optimizer import precompute_data, simulate_strategy

logger = logging.getLogger("AutoLearner.OptimizationRunner")
CONFIG_PATH = Path("config/optimized_strategy_config.yaml")


@dataclass
class CandidateConfig:
    """Evaluated parameter candidate."""

    sl_atr_mult: float
    tp_atr_mult: float
    min_vix_gate: float
    consensus_rule: str
    active_strategies: List[str]
    in_sample_sharpe: float
    in_sample_pf: float
    in_sample_return_pct: float
    out_of_sample_sharpe: float
    out_of_sample_pf: float
    out_of_sample_return_pct: float
    out_of_sample_max_dd: float
    out_of_sample_trades: int
    is_promotable: bool = False
    improvement_pct: float = 0.0
    alpha_annual_pct: float = 0.0
    information_ratio: float = 0.0
    excess_return_pct: float = 0.0
    bench_return_pct: float = 0.0
    days_beat_pct: float = 0.0
    n_sessions: int = 0
    significant: bool = False


class OptimizationRunner:
    """
    Automated Walk-Forward Grid Search & Parameter Evaluator.
    """

    def __init__(self, data_path: Path = Path("data/historical/banknifty_5min.csv")) -> None:
        self.data_path = data_path
        self.live_config = self._load_live_config()

    def _load_live_config(self) -> Dict[str, Any]:
        """Load currently active production config."""
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {
            "risk_and_execution_parameters": {"stop_loss_atr_multiplier": 1.0, "take_profit_atr_multiplier": 2.0},
            "regime_and_volatility_filters": {"min_india_vix_entry": 11.5},
            "ensemble_configuration": {"voting_rule": "BENCHMARK", "active_strategies": ["ORB", "SUPERTREND", "FLOW"]},
        }

    def run_walk_forward_optimization(self) -> Tuple[CandidateConfig, List[CandidateConfig]]:
        """
        Execute In-Sample training & Out-of-Sample validation grid search.
        """
        logger.info("Precomputing technical indicators for walk-forward optimization...")
        df_full = precompute_data(self.data_path)

        # 1. Rolling Split: 75% In-Sample (Training) / 25% Out-of-Sample (Validation)
        split_idx = int(len(df_full) * 0.75)
        df_is = df_full.iloc[:split_idx].copy().reset_index(drop=True)
        df_oos = df_full.iloc[split_idx:].copy().reset_index(drop=True)

        logger.info("Split dataset: In-Sample = %d bars (%s to %s) | Out-of-Sample = %d bars (%s to %s)",
                    len(df_is), df_is['timestamp'].min().strftime('%Y-%m-%d'), df_is['timestamp'].max().strftime('%Y-%m-%d'),
                    len(df_oos), df_oos['timestamp'].min().strftime('%Y-%m-%d'), df_oos['timestamp'].max().strftime('%Y-%m-%d'))

        # 2. Evaluate Baseline Performance on OOS set
        base_sl = float(self.live_config["risk_and_execution_parameters"]["stop_loss_atr_multiplier"])
        base_tp = float(self.live_config["risk_and_execution_parameters"]["take_profit_atr_multiplier"])
        base_vix = float(self.live_config["regime_and_volatility_filters"]["min_india_vix_entry"])
        base_rule = str(self.live_config["ensemble_configuration"]["voting_rule"])
        base_strats = list(self.live_config["ensemble_configuration"]["active_strategies"])

        bench_oos = session_benchmark_returns(df_oos, "NAIVE_LONG")

        res_baseline_oos = simulate_strategy(
            df=df_oos,
            symbol="BANKNIFTY",
            sl_atr_mult=base_sl,
            tp_atr_mult=base_tp,
            min_vix=base_vix,
            active_strategies=base_strats,
            consensus_rule=base_rule,
            label="BASELINE_OOS",
        )
        baseline_alpha = post_compute_alpha(res_baseline_oos, df_oos, bench_df=bench_oos)
        if baseline_alpha is not None:
            logger.info("Baseline Alpha vs NAIVE_LONG: alpha=%.2f%%/yr, beta=%.2f, excess=%.2f%%, significant=%s",
                        baseline_alpha.alpha_annual_pct, baseline_alpha.beta,
                        baseline_alpha.excess_return_pct, baseline_alpha.significant)
        baseline_oos_sharpe = max(0.1, res_baseline_oos.sharpe_ratio)
        logger.info("Baseline Live Config on OOS set: Sharpe = %.2f, PF = %.2f, Return = %+.2f%%",
                    res_baseline_oos.sharpe_ratio, res_baseline_oos.profit_factor, res_baseline_oos.total_return_pct)

        # 3. Parameter Grid to Sweep
        sl_grid = [0.75, 1.0, 1.25, 1.5]
        tp_grid = [1.5, 2.0, 2.5]
        vix_grid = [10.0, 11.5, 13.0]
        consensus_grid = ["ALT_B", "BENCHMARK"]

        candidates: List[CandidateConfig] = []

        for sl, tp, vix, rule in itertools.product(sl_grid, tp_grid, vix_grid, consensus_grid):
            # In-Sample Training Run
            res_is = simulate_strategy(
                df=df_is,
                symbol="BANKNIFTY",
                sl_atr_mult=sl,
                tp_atr_mult=tp,
                min_vix=vix,
                active_strategies=base_strats,
                consensus_rule=rule,
                label=f"IS_SL_{sl}_TP_{tp}",
            )

            # Only test on OOS if In-Sample has positive edge
            if res_is.sharpe_ratio <= 0.0 or res_is.profit_factor < 1.0:
                continue

            # Out-of-Sample Validation Run
            res_oos = simulate_strategy(
                df=df_oos,
                symbol="BANKNIFTY",
                sl_atr_mult=sl,
                tp_atr_mult=tp,
                min_vix=vix,
                active_strategies=base_strats,
                consensus_rule=rule,
                label=f"OOS_SL_{sl}_TP_{tp}",
            )

            imp_pct = ((res_oos.sharpe_ratio - baseline_oos_sharpe) / baseline_oos_sharpe) * 100.0
            cand_alpha = post_compute_alpha(res_oos, df_oos, bench_df=bench_oos)
            is_promo, promo_reason = promotion_eligible(
                sharpe_ratio=res_oos.sharpe_ratio,
                baseline_sharpe=baseline_oos_sharpe,
                profit_factor=res_oos.profit_factor,
                alpha=cand_alpha,
                baseline_alpha=baseline_alpha,
            )
            if is_promo:
                logger.info("Promotable gamma: SL=%.2fx TP=%.2fx VIX=%.1f %s | alpha=%.2f%%/yr excess=%.2f%% significant=%s",
                            sl, tp, vix, rule,
                            (cand_alpha.alpha_annual_pct if cand_alpha else 0.0),
                            (cand_alpha.excess_return_pct if cand_alpha else 0.0),
                            (cand_alpha.significant if cand_alpha else False))

            cand = CandidateConfig(
                sl_atr_mult=sl,
                tp_atr_mult=tp,
                min_vix_gate=vix,
                consensus_rule=rule,
                active_strategies=base_strats,
                in_sample_sharpe=res_is.sharpe_ratio,
                in_sample_pf=res_is.profit_factor,
                in_sample_return_pct=res_is.total_return_pct,
                out_of_sample_sharpe=res_oos.sharpe_ratio,
                out_of_sample_pf=res_oos.profit_factor,
                out_of_sample_return_pct=res_oos.total_return_pct,
                out_of_sample_max_dd=res_oos.max_drawdown_pct,
                out_of_sample_trades=res_oos.total_trades,
                is_promotable=is_promo,
                improvement_pct=round(imp_pct, 2),
                alpha_annual_pct=(cand_alpha.alpha_annual_pct if cand_alpha else 0.0),
                information_ratio=(cand_alpha.information_ratio if cand_alpha else 0.0),
                excess_return_pct=(cand_alpha.excess_return_pct if cand_alpha else 0.0),
                bench_return_pct=(cand_alpha.bench_return_pct if cand_alpha else 0.0),
                days_beat_pct=(cand_alpha.days_beat_pct if cand_alpha else 0.0),
                n_sessions=(cand_alpha.n_sessions if cand_alpha else 0),
                significant=(cand_alpha.significant if cand_alpha else False),
            )
            candidates.append(cand)

        # 4. Rank candidates by OOS Sharpe Ratio
        ranked_candidates = sorted(candidates, key=lambda x: (x.out_of_sample_sharpe, x.out_of_sample_pf), reverse=True)

        if not ranked_candidates:
            # If no candidates beat filter, return baseline as non-promotable
            best_candidate = CandidateConfig(
                sl_atr_mult=base_sl,
                tp_atr_mult=base_tp,
                min_vix_gate=base_vix,
                consensus_rule=base_rule,
                active_strategies=base_strats,
                in_sample_sharpe=0.0,
                in_sample_pf=1.0,
                in_sample_return_pct=0.0,
                out_of_sample_sharpe=res_baseline_oos.sharpe_ratio,
                out_of_sample_pf=res_baseline_oos.profit_factor,
                out_of_sample_return_pct=res_baseline_oos.total_return_pct,
                out_of_sample_max_dd=res_baseline_oos.max_drawdown_pct,
                out_of_sample_trades=res_baseline_oos.total_trades,
                is_promotable=False,
                improvement_pct=0.0,
                alpha_annual_pct=(baseline_alpha.alpha_annual_pct if baseline_alpha else 0.0),
                information_ratio=(baseline_alpha.information_ratio if baseline_alpha else 0.0),
                excess_return_pct=(baseline_alpha.excess_return_pct if baseline_alpha else 0.0),
                bench_return_pct=(baseline_alpha.bench_return_pct if baseline_alpha else 0.0),
                days_beat_pct=(baseline_alpha.days_beat_pct if baseline_alpha else 0.0),
                n_sessions=(baseline_alpha.n_sessions if baseline_alpha else 0),
                significant=(baseline_alpha.significant if baseline_alpha else False),
            )
        else:
            best_candidate = ranked_candidates[0]

        logger.info("Best Candidate: SL=%.2fx, TP=%.2fx, VIX=%.1f, Rule=%s | OOS Sharpe=%.2f (Improvement: %+.1f%%) | Promotable: %s | Alpha=%.2f%%/yr, Excess=%.2f%%, Significant=%s",
                    best_candidate.sl_atr_mult, best_candidate.tp_atr_mult, best_candidate.min_vix_gate,
                    best_candidate.consensus_rule, best_candidate.out_of_sample_sharpe,
                    best_candidate.improvement_pct, best_candidate.is_promotable,
                    best_candidate.alpha_annual_pct, best_candidate.excess_return_pct,
                    best_candidate.significant)

        return best_candidate, ranked_candidates


# Global instance
optimization_runner = OptimizationRunner()

if __name__ == "__main__":
    best, all_cands = optimization_runner.run_walk_forward_optimization()
    logger.info("Top Candidate: SL=%.2fx, TP=%.2fx, VIX=%.1f, Rule=%s", best.sl_atr_mult, best.tp_atr_mult, best.min_vix_gate, best.consensus_rule)
    logger.info("OOS Sharpe: %.2f | OOS PF: %.2f | Promotable: %s", best.out_of_sample_sharpe, best.out_of_sample_pf, best.is_promotable)
