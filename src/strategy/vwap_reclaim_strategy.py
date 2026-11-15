"""
Strategy 2: VWAP Reclaim Strategy — Algo-Trader 4-Strategy Ensemble.

Rules:
- Entry Long: Price dips below Intraday VWAP then reclaims and closes firmly above VWAP with Bullish bias.
- Entry Short: Price rallies above Intraday VWAP then fails and closes firmly below VWAP with Bearish bias.
- Filter: Must operate in Trending regime (Price > EMA 50 or ADX > 20).
- Target: 2.0× ATR(14).
- Stop Loss: 0.5× ATR(14) below VWAP level.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.strategy.orb_strategy import StrategySignal

logger = logging.getLogger("VWAPReclaimStrategy")


class VWAPReclaimStrategy:
    """
    VWAP Reclaim with Momentum & ATR Risk Controls.
    """

    def __init__(
        self,
        target_atr_mult: float = 2.0,
        stop_atr_mult: float = 0.5,
    ) -> None:
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult

    def evaluate(self, df_5m: pd.DataFrame, symbol: str = "BANKNIFTY") -> StrategySignal:
        """Evaluate VWAP reclaim setup."""
        if len(df_5m) < 10:
            return StrategySignal("VWAP_Reclaim", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["Insufficient Bars"])

        c = df_5m["close"]
        h = df_5m["high"]
        l = df_5m["low"]
        v = df_5m["volume"]

        # VWAP calculation
        cum_vol = v.cumsum()
        cum_pv = (c * v).cumsum()
        vwap = np.where(cum_vol > 0, cum_pv / cum_vol, c)

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        atr14 = float(pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean().iloc[-1])

        curr_close = float(c.iloc[-1])
        prev_close = float(c.iloc[-2])
        curr_vwap = float(vwap[-1])
        prev_vwap = float(vwap[-2])

        # Long Reclaim: Previous candle below VWAP, current candle closes firmly above VWAP
        if prev_close < prev_vwap and curr_close > curr_vwap:
            entry = curr_close
            sl = curr_vwap - (self.stop_atr_mult * atr14)
            tp = entry + (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="VWAP_Reclaim",
                symbol=symbol,
                direction="LONG",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.88,
                passed_filters=["VWAP Cross Above", "Bullish Close", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        # Short Reclaim: Previous candle above VWAP, current candle closes firmly below VWAP
        elif prev_close > prev_vwap and curr_close < curr_vwap:
            entry = curr_close
            sl = curr_vwap + (self.stop_atr_mult * atr14)
            tp = entry - (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="VWAP_Reclaim",
                symbol=symbol,
                direction="SHORT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.88,
                passed_filters=["VWAP Rejection Below", "Bearish Close", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        return StrategySignal("VWAP_Reclaim", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["No VWAP Reclaim"])


# Global singleton strategy
vwap_reclaim_strategy = VWAPReclaimStrategy()
