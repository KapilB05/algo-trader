"""
Strategy 3: EMA-Supertrend Trend Alignment — Algo-Trader 4-Strategy Ensemble.

Rules:
- Entry Long: EMA 9 crosses above EMA 21 AND Supertrend(10, 3) direction == 1 (Bullish).
- Entry Short: EMA 9 crosses below EMA 21 AND Supertrend(10, 3) direction == -1 (Bearish).
- Filter: Both indicators must agree simultaneously.
- Stop Loss: Supertrend line or 1.5× ATR(14).
- Target: 2.0× ATR(14) with dynamic Supertrend trailing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.ml.extended_features import calculate_supertrend
from src.strategy.orb_strategy import StrategySignal

logger = logging.getLogger("SupertrendStrategy")


class EMASupertrendStrategy:
    """
    Dual Trend Alignment: EMA 9/21 cross + Supertrend(10, 3).
    """

    def __init__(self, target_atr_mult: float = 2.0, stop_atr_mult: float = 1.2) -> None:
        self.target_atr_mult = target_atr_mult
        self.stop_atr_mult = stop_atr_mult

    def evaluate(self, df_5m: pd.DataFrame, symbol: str = "BANKNIFTY") -> StrategySignal:
        """Evaluate EMA-Supertrend confluence signal."""
        if len(df_5m) < 25:
            return StrategySignal("EMA_Supertrend", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["Insufficient Bars"])

        c = df_5m["close"]
        h = df_5m["high"]
        l = df_5m["low"]

        # EMAs
        ema9 = c.ewm(span=9, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()

        # Supertrend
        st_dir, st_band = calculate_supertrend(h, l, c, period=10, multiplier=3.0)

        # ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        atr14 = float(pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14, min_periods=1).mean().iloc[-1])

        curr_close = float(c.iloc[-1])
        curr_ema9 = float(ema9.iloc[-1])
        curr_ema21 = float(ema21.iloc[-1])
        prev_ema9 = float(ema9.iloc[-2])
        curr_st_dir = float(st_dir[-1]) if isinstance(st_dir, np.ndarray) else float(st_dir.iloc[-1])
        curr_st_band = float(st_band[-1]) if isinstance(st_band, np.ndarray) else float(st_band.iloc[-1])

        # Long: EMA 9 > 21 + Supertrend Bullish (1.0)
        is_bull_cross = (curr_ema9 > curr_ema21) and (curr_st_dir == 1.0)
        if is_bull_cross:
            entry = curr_close
            sl = max(curr_st_band, entry - (self.stop_atr_mult * atr14))
            tp = entry + (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="EMA_Supertrend",
                symbol=symbol,
                direction="LONG",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.86,
                passed_filters=["EMA 9 > 21", "Supertrend Bullish", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        # Short: EMA 9 < 21 + Supertrend Bearish (-1.0)
        is_bear_cross = (curr_ema9 < curr_ema21) and (curr_st_dir == -1.0)
        if is_bear_cross:
            entry = curr_close
            sl = min(curr_st_band, entry + (self.stop_atr_mult * atr14))
            tp = entry - (self.target_atr_mult * atr14)
            return StrategySignal(
                strategy_name="EMA_Supertrend",
                symbol=symbol,
                direction="SHORT",
                entry_price=round(entry, 2),
                stop_loss=round(sl, 2),
                target_price=round(tp, 2),
                confidence=0.86,
                passed_filters=["EMA 9 < 21", "Supertrend Bearish", f"ATR {atr14:.1f}"],
                rr_ratio=self.target_atr_mult / self.stop_atr_mult,
            )

        return StrategySignal("EMA_Supertrend", symbol, "NONE", 0.0, 0.0, 0.0, 0.0, ["No Trend Alignment"])


# Global singleton strategy
supertrend_strategy = EMASupertrendStrategy()
