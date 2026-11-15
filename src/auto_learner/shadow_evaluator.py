"""
Continuous Auto-Learner — Shadow Mode Safety Verification.

Executes side-by-side shadow backtesting before promoting any parameter changes:
1. Runs candidate configuration on recent 3-month validation window
2. Runs current live configuration on identical window
3. Compares Sharpe Ratio, Profit Factor, and Max Drawdown
4. Enforces strict safety gates: candidate must NOT increase Max Drawdown beyond safety threshold
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yaml

from src.analysis.benchmark import post_compute_alpha, session_benchmark_returns
from src.backtest.diagnostic_optimizer import precompute_data, simulate_strategy

logger = logging.getLogger("AutoLearner.ShadowEvaluator")
CONFIG_PATH = Path("config/optimized_strategy_config.yaml")

# Shadow window in sessions. Alpha significance requires MIN_SESSIONS (40) aligned
# sessions — the old "recent 25% of bars" window held only ~35 sessions, so the
# significance gate could NEVER pass and every candidate was rejected by design.
# 60 sessions (~3 months intraday) keeps the recency intent while letting a real
# edge clear the bar.
SHADOW_SESSIONS = 60


@dataclass
class ShadowEvaluationResult:
    """Result of shadow mode safety comparison."""

    approved: bool
    reason: str
    live_sharpe: float
    candidate_sharpe: float
    live_return_pct: float
    candidate_return_pct: float
    live_max_dd: float
    candidate_max_dd: float
    improvement_sharpe_pct: float
    benchmark: str = "NAIVE_LONG"
    n_sessions: int = 0
    benchmark_total_return_pct: float = 0.0
    live_alpha_annual_pct: float = 0.0
    live_excess_return_pct: float = 0.0
    live_significant: bool = False
    candidate_alpha_annual_pct: float = 0.0
    candidate_excess_return_pct: float = 0.0
    candidate_information_ratio: float = 0.0
    candidate_significant: bool = False


class ShadowEvaluator:
    """
    Shadow Mode Safety Verification Gateway.
    """

    def __init__(self, data_path: Path = Path("data/historical/banknifty_5min.csv")) -> None:
        self.data_path = data_path

    @staticmethod
    def _recent_sessions(df_full: pd.DataFrame, n_sessions: int = SHADOW_SESSIONS) -> pd.DataFrame:
        """Slice the most recent N sessions (fallback: most recent half of bars)."""
        try:
            session_keys = pd.to_datetime(df_full["date"])
            ordered = session_keys.drop_duplicates(keep="last")
            cutoff = ordered.iloc[-n_sessions] if len(ordered) > n_sessions else ordered.iloc[0]
            mask = session_keys >= cutoff
            df_shadow = df_full.loc[mask].copy().reset_index(drop=True)
            if len(df_shadow) >= 100:
                return df_shadow
        except (KeyError, IndexError, TypeError):
            pass
        split_idx = int(len(df_full) * 0.50)
        return df_full.iloc[split_idx:].copy().reset_index(drop=True)

    def evaluate_candidate(self, candidate_params: Dict[str, Any]) -> ShadowEvaluationResult:
        """
        Execute shadow backtest comparing candidate against live config on recent data.
        """
        df_full = precompute_data(self.data_path)
        # Recent session window (see SHADOW_SESSIONS) — long enough for the
        # benchmark-significance gate to be achievable, short enough to reflect
        # the current regime.
        df_shadow = self._recent_sessions(df_full)
        bench_df = session_benchmark_returns(df_shadow, "NAIVE_LONG")

        # 1. Run Live Config
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            live_cfg = yaml.safe_load(f)

        live_sl = float(live_cfg["risk_and_execution_parameters"]["stop_loss_atr_multiplier"])
        live_tp = float(live_cfg["risk_and_execution_parameters"]["take_profit_atr_multiplier"])
        live_vix = float(live_cfg["regime_and_volatility_filters"]["min_india_vix_entry"])
        live_rule = str(live_cfg["ensemble_configuration"]["voting_rule"])
        active_strats = list(live_cfg["ensemble_configuration"]["active_strategies"])

        res_live = simulate_strategy(
            df=df_shadow,
            symbol="BANKNIFTY",
            sl_atr_mult=live_sl,
            tp_atr_mult=live_tp,
            min_vix=live_vix,
            active_strategies=active_strats,
            consensus_rule=live_rule,
            label="SHADOW_LIVE",
        )

        # 2. Run Candidate Config
        cand_sl = float(candidate_params.get("sl_atr_mult", live_sl))
        cand_tp = float(candidate_params.get("tp_atr_mult", live_tp))
        cand_vix = float(candidate_params.get("min_vix_gate", live_vix))
        cand_rule = str(candidate_params.get("consensus_rule", live_rule))

        res_cand = simulate_strategy(
            df=df_shadow,
            symbol="BANKNIFTY",
            sl_atr_mult=cand_sl,
            tp_atr_mult=cand_tp,
            min_vix=cand_vix,
            active_strategies=active_strats,
            consensus_rule=cand_rule,
            label="SHADOW_CANDIDATE",
        )

        sharpe_imp = ((res_cand.sharpe_ratio - res_live.sharpe_ratio) / max(0.1, abs(res_live.sharpe_ratio))) * 100.0

        live_alpha = post_compute_alpha(res_live, df_shadow, bench_df=bench_df)
        cand_alpha = post_compute_alpha(res_cand, df_shadow, bench_df=bench_df)

        # Safety Gate Checks
        # Rule 1: Candidate Sharpe must exceed Live Sharpe
        # Rule 2: Candidate Max Drawdown must NOT exceed 10.0%
        # Rule 3: Candidate Profit Factor must be >= 1.15
        # Rule 4: Candidate must show positive, statistically significant alpha
        #         (excess over naive-long intraday benchmark) that also beats the
        #         live config's excess — i.e. the candidate must beat the MARKET,
        #         not just the previous config.
        # Rule 5: Candidate recent-window Sharpe must be POSITIVE — the strategy's
        #         edge must hold in the CURRENT regime, not just in the long-run
        #         historical window (protects against regime decay).
        approved = True
        reasons = []

        if res_cand.sharpe_ratio < res_live.sharpe_ratio:
            approved = False
            reasons.append(f"Candidate Sharpe ({res_cand.sharpe_ratio:.2f}) < Live Sharpe ({res_live.sharpe_ratio:.2f})")

        if res_cand.sharpe_ratio <= 0.0:
            approved = False
            reasons.append(
                f"Candidate recent-window Sharpe ({res_cand.sharpe_ratio:.2f}) is not positive — "
                "edge has decayed in the current regime, refusing deployment")

        if res_cand.max_drawdown_pct > 10.0:
            approved = False
            reasons.append(f"Candidate Drawdown ({res_cand.max_drawdown_pct:.2f}%) exceeds 10% safety cap")

        if res_cand.profit_factor < 1.15:
            approved = False
            reasons.append(f"Candidate Profit Factor ({res_cand.profit_factor:.2f}) < 1.15 minimum edge threshold")

        if cand_alpha is None or live_alpha is None:
            approved = False
            reasons.append("Insufficient sessions to compute benchmark-relative alpha")
        else:
            if not cand_alpha.significant:
                approved = False
                reasons.append(
                    f"Candidate alpha NOT significant (p={cand_alpha.p_value:.3f}, n={cand_alpha.n_sessions}) — "
                    f"excess over {cand_alpha.benchmark} could be noise")
            if cand_alpha.alpha_annual_pct <= 0.0:
                approved = False
                reasons.append(f"Candidate annualized alpha not positive ({cand_alpha.alpha_annual_pct:+.1f}%)")
            if cand_alpha.excess_return_pct <= 0.0:
                approved = False
                reasons.append(f"Candidate shows no excess return vs {cand_alpha.benchmark} ({cand_alpha.excess_return_pct:+.2f}%)")
            if cand_alpha.excess_return_pct <= live_alpha.excess_return_pct:
                approved = False
                reasons.append(
                    f"Candidate excess ({cand_alpha.excess_return_pct:+.2f}%) not above live excess "
                    f"({live_alpha.excess_return_pct:+.2f}%)")

        summary_reason = "Approved: Candidate outperforms live config under shadow testing" if approved else f"Rejected: {'; '.join(reasons)}"

        if approved:
            logger.info("SHADOW SAFETY CHECK PASSED: %s", summary_reason)
        else:
            logger.warning("SHADOW SAFETY CHECK REJECTED: %s", summary_reason)

        return ShadowEvaluationResult(
            approved=approved,
            reason=summary_reason,
            live_sharpe=res_live.sharpe_ratio,
            candidate_sharpe=res_cand.sharpe_ratio,
            live_return_pct=res_live.total_return_pct,
            candidate_return_pct=res_cand.total_return_pct,
            live_max_dd=res_live.max_drawdown_pct,
            candidate_max_dd=res_cand.max_drawdown_pct,
            improvement_sharpe_pct=round(sharpe_imp, 2),
            benchmark=(cand_alpha.benchmark if cand_alpha else "NAIVE_LONG"),
            n_sessions=(cand_alpha.n_sessions if cand_alpha else 0),
            benchmark_total_return_pct=(live_alpha.bench_return_pct if live_alpha else 0.0),
            live_alpha_annual_pct=(live_alpha.alpha_annual_pct if live_alpha else 0.0),
            live_excess_return_pct=(live_alpha.excess_return_pct if live_alpha else 0.0),
            live_significant=(live_alpha.significant if live_alpha else False),
            candidate_alpha_annual_pct=(cand_alpha.alpha_annual_pct if cand_alpha else 0.0),
            candidate_excess_return_pct=(cand_alpha.excess_return_pct if cand_alpha else 0.0),
            candidate_information_ratio=(cand_alpha.information_ratio if cand_alpha else 0.0),
            candidate_significant=(cand_alpha.significant if cand_alpha else False),
        )


# Global instance
shadow_evaluator = ShadowEvaluator()
