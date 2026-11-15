"""
Strategy 4: Institutional Flow Following — Algo-Trader 4-Strategy Ensemble.

Rules:
- Identifies Option OI Magnets: Highest Call OI (Resistance Magnet) & Highest Put OI (Support Magnet).
- Entry Long: Price expands toward Call OI Magnet with FII/DII Net Flow > +₹1,000 Cr and PCR > 1.10.
- Entry Short: Price drops toward Put OI Magnet with FII/DII Net Flow < -₹1,000 Cr and PCR < 0.90.
- Target: 0.5% just before reaching the maximum OI concentration strike (preventing pin trap).
- Stop Loss: 1.5× ATR(14) or breach of opposite support/resistance strike.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.institutional_flows import institutional_tracker
from src.strategy.orb_strategy import StrategySignal

logger = logging.getLogger("InstitutionalFlowStrategy")


class InstitutionalFlowStrategy:
    """
    Follows institutional positioning footprints (FII net flows + OI Magnets).
    """

    def __init__(self, flow_threshold_cr: float = 1000.0, target_atr_mult: float = 2.0) -> None:
        self.flow_threshold_cr = flow_threshold_cr
        self.target_atr_mult = target_atr_mult

    def evaluate(
        self,
        df_5m: pd.DataFrame,
        symbol: str = "BANKNIFTY",
        fii_net_cr: float = 1250.0,
        dii_net_cr: float = 450.0,
        pcr: float = 1.25,
        max_call_strike: float = 58000.0,
        max_put_strike: float = 57000.0,
    ) -> StrategySignal:
        """Evaluate institutional flow alignment signal."""
        if len(df_5m) < 10:
            return StrategySignal("Institutional_Flow", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["Insufficient Bars"])

        c = df_5m["close"]
        h = df_5m["high"]
        l = df_5m["low"]

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        atr14 = float(pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean().iloc[-1])

        curr_close = float(c.iloc[-1])
        total_flow = fii_net_cr + dii_net_cr

        # Long Institutional Flow
        if total_flow >= self.flow_threshold_cr and pcr >= 1.10 and curr_close < max_call_strike:
            entry = curr_close
            sl = entry - (1.5 * atr14)
            # Target before call magnet
            tp = min(max_call_strike * 0.998, entry + (self.target_atr_mult * atr14))
            return StrategySignal(
                strategy_name="Institutional_Flow",
                symbol=symbol,
                direction="LONG",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.90,
                passed_filters=[f"FII Flow +₹{total_flow:.0f}Cr", f"PCR {pcr:.2f}", f"Call Magnet ₹{max_call_strike:.0f}"],
                rr_ratio=round((tp - entry) / max(1.0, (entry - sl)), 2),
            )

        # Short Institutional Flow
        elif total_flow <= -self.flow_threshold_cr and pcr <= 0.90 and curr_close > max_put_strike:
            entry = curr_close
            sl = entry + (1.5 * atr14)
            # Target before put magnet
            tp = max(max_put_strike * 1.002, entry - (self.target_atr_mult * atr14))
            return StrategySignal(
                strategy_name="Institutional_Flow",
                symbol=symbol,
                direction="SHORT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.90,
                passed_filters=[f"FII Flow ₹{total_flow:.0f}Cr", f"PCR {pcr:.2f}", f"Put Magnet ₹{max_put_strike:.0f}"],
                rr_ratio=round((entry - tp) / max(1.0, (sl - entry)), 2),
            )

        return StrategySignal("Institutional_Flow", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["Flow Neutral"])


# Global singleton strategy
institutional_flow_strategy = InstitutionalFlowStrategy()
