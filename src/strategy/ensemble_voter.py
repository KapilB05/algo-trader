"""
Ensemble Voting Engine — Algo-Trader 4-Strategy Ensemble (Layer 4).

Combines votes from 4 Base Strategies:
1. ORB (Opening Range Breakout)
2. VWAP Reclaim
3. EMA-Supertrend
4. Institutional Flow Following

Gating & Sizing Rules:
- 3 of 4 agree on Direction → Full Position (100% of calculated size)
- 2 of 4 agree on Direction → Half Position (50% of calculated size)
- 1 or 0 agree (or split)   → NO TRADE (0% size)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.strategy.institutional_flow_strategy import institutional_flow_strategy
from src.strategy.orb_strategy import StrategySignal, orb_strategy
from src.strategy.regime_detector import MarketRegime, regime_detector
from src.strategy.supertrend_strategy import supertrend_strategy
from src.strategy.vwap_reclaim_strategy import vwap_reclaim_strategy

logger = logging.getLogger("EnsembleVoter")


@dataclass
class EnsembleDecision:
    """Consolidated decision from the multi-strategy ensemble."""

    trade_approved: bool
    direction: str              # LONG / SHORT / NONE
    position_size_multiplier: float  # 1.0 (100%), 0.5 (50%), 0.0 (No Trade)
    agreed_votes: int           # Count of strategies agreeing (0 to 4)
    total_strategies: int
    signals: List[StrategySignal] = field(default_factory=list)
    average_entry: float = 0.0
    recommended_sl: float = 0.0
    recommended_tp: float = 0.0
    regime: str = "CHOP"
    rationale: str = ""


class MultiStrategyEnsembleVoter:
    """
    Evaluates all 4 base strategies in parallel and applies strict consensus voting.
    """

    def __init__(self) -> None:
        self.strategies = [
            orb_strategy,
            vwap_reclaim_strategy,
            supertrend_strategy,
            institutional_flow_strategy,
        ]

    def evaluate_all(
        self,
        df_5m: pd.DataFrame,
        symbol: str = "BANKNIFTY",
        fii_net_cr: float = 0.0,
        dii_net_cr: float = 0.0,
        pcr: float = 1.0,
        max_call_strike: float = 0.0,
        max_put_strike: float = 0.0,
        current_vix: float = 14.0,
    ) -> EnsembleDecision:
        """
        Run all 4 strategies, check regime gating, and aggregate consensus vote.
        """
        # 1. Check Layer 3 Regime Gating
        active_regime = regime_detector.classify_market_state(df_5m, current_vix=current_vix)
        if not regime_detector.is_trade_allowed():
            return EnsembleDecision(
                trade_approved=False,
                direction="NONE",
                position_size_multiplier=0.0,
                agreed_votes=0,
                total_strategies=len(self.strategies),
                regime=active_regime.value,
                rationale=f"Trading gated by regime: {active_regime.value.upper()} (100% Cash)",
            )

        # 2. Collect signals from each strategy
        s1 = orb_strategy.evaluate(df_5m, symbol=symbol)
        s2 = vwap_reclaim_strategy.evaluate(df_5m, symbol=symbol)
        s3 = supertrend_strategy.evaluate(df_5m, symbol=symbol)
        s4 = institutional_flow_strategy.evaluate(
            df_5m,
            symbol=symbol,
            fii_net_cr=fii_net_cr,
            dii_net_cr=dii_net_cr,
            pcr=pcr,
            max_call_strike=max_call_strike if max_call_strike > 0 else (df_5m["close"].iloc[-1] + 500),
            max_put_strike=max_put_strike if max_put_strike > 0 else (df_5m["close"].iloc[-1] - 500),
        )

        all_signals = [s1, s2, s3, s4]
        long_votes = [s for s in all_signals if s.direction == "LONG"]
        short_votes = [s for s in all_signals if s.direction == "SHORT"]

        long_count = len(long_votes)
        short_count = len(short_votes)

        # Determine consensus direction and multiplier
        if long_count >= 3:
            winner_signals = long_votes
            chosen_dir = "LONG"
            multiplier = 1.0  # 100%
        elif short_count >= 3:
            winner_signals = short_votes
            chosen_dir = "SHORT"
            multiplier = 1.0  # 100%
        elif long_count == 2 and short_count == 0:
            winner_signals = long_votes
            chosen_dir = "LONG"
            multiplier = 0.5  # 50%
        elif short_count == 2 and long_count == 0:
            winner_signals = short_votes
            chosen_dir = "SHORT"
            multiplier = 0.5  # 50%
        else:
            return EnsembleDecision(
                trade_approved=False,
                direction="NONE",
                position_size_multiplier=0.0,
                agreed_votes=max(long_count, short_count),
                total_strategies=len(self.strategies),
                signals=all_signals,
                regime=active_regime.value,
                rationale=f"Insufficient consensus: {long_count} Long, {short_count} Short (Need >=2 unanimous)",
            )

        # Compute consensus entry, stop loss, and target
        avg_entry = float(np.mean([s.entry_price for s in winner_signals]))
        if chosen_dir == "LONG":
            sl = float(np.max([s.stop_loss for s in winner_signals]))  # Tightest SL for Long
            tp = float(np.mean([s.target_price for s in winner_signals]))
        else:
            sl = float(np.min([s.stop_loss for s in winner_signals]))  # Tightest SL for Short
            tp = float(np.mean([s.target_price for s in winner_signals]))

        agreed_names = [s.strategy_name for s in winner_signals]
        rationale = f"{len(winner_signals)}/4 Consensus ({', '.join(agreed_names)}) → Sizing {int(multiplier*100)}%"

        return EnsembleDecision(
            trade_approved=True,
            direction=chosen_dir,
            position_size_multiplier=multiplier,
            agreed_votes=len(winner_signals),
            total_strategies=len(self.strategies),
            signals=all_signals,
            average_entry=round(avg_entry, 2),
            recommended_sl=round(sl, 2),
            recommended_tp=round(tp, 2),
            regime=active_regime.value,
            rationale=rationale,
        )


# Global singleton voter
ensemble_voter = MultiStrategyEnsembleVoter()
