"""
Strategy 1: Opening Range Breakout (ORB) — Algo-Trader 4-Strategy Ensemble.

Rules:
- Defines Opening Range using the first 15 minutes of the trading day (09:15 - 09:30 IST).
- Entry Long: Price breaks above 15-min High with Volume > 1.5× 20-period Volume MA.
- Entry Short: Price breaks below 15-min Low with Volume > 1.5× 20-period Volume MA.
- Target: 1.5× ATR(14) from entry price.
- Stop Loss: 1.0× ATR(14) from entry price.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("ORBStrategy")


@dataclass
class StrategySignal:
    """Standardized strategy output signal."""

    strategy_name: str
    symbol: str
    direction: str       # LONG / SHORT / NONE
    entry_price: float
    stop_loss: float
    target_price: float
    confidence: float    # 0.0 to 1.0
    passed_filters: list[str]
    rr_ratio: float = 1.5


class OpeningRangeBreakoutStrategy:
    """
    15-Minute Opening Range Breakout Strategy with Volume & ATR Filters.
    """

    def __init__(
        self,
        range_bars: int = 3,  # 3 x 5-min bars = 15 minutes
        volume_surge_mult: float = 1.5,
        target_atr_mult: float = 1.5,
        stop_atr_mult: float = 1.0,
    ) -> None:
        self.range_bars = range_bars
        self.volume_surge_mult = volume_surge_mult
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult

    def evaluate(self, df_5m: pd.DataFrame, symbol: str = "BANKNIFTY") -> StrategySignal:
        """Evaluate ORB signal from 5-minute candlestick history."""
        if len(df_5m) < self.range_bars + 1:
            return StrategySignal("ORB", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["Insufficient Bars"])

        c = df_5m["close"]
        h = df_5m["high"]
        l = df_5m["low"]
        v = df_5m["volume"]

        # Calculate 15-min opening range from first 3 bars
        orh = float(h.iloc[: self.range_bars].max())
        orl = float(l.iloc[: self.range_bars].min())

        # Volume MA 20
        v_ma20 = float(v.rolling(20, min_periods=1).mean().iloc[-1])
        curr_vol = float(v.iloc[-1])
        curr_close = float(c.iloc[-1])

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        atr14 = float(pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean().iloc[-1])

        vol_passed = curr_vol >= (v_ma20 * self.volume_surge_mult)

        # Long Breakout
        if curr_close > orh and vol_passed:
            entry = curr_close
            sl = entry - (self.stop_atr_mult * atr14)
            tp = entry + (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="ORB_Breakout",
                symbol=symbol,
                direction="LONG",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.85,
                passed_filters=["15-Min ORH Break", f"Volume Surge {curr_vol/v_ma20:.1f}x", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        # Short Breakout
        elif curr_close < orl and vol_passed:
            entry = curr_close
            sl = entry + (self.stop_atr_mult * atr14)
            tp = entry - (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="ORB_Breakout",
                symbol=symbol,
                direction="SHORT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.85,
                passed_filters=["15-Min ORL Break", f"Volume Surge {curr_vol/v_ma20:.1f}x", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        return StrategySignal("ORB_Breakout", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["No Breakout"])


# Global singleton strategy
orb_strategy = OpeningRangeBreakoutStrategy()
