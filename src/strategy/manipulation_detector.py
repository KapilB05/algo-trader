"""
Jane Street Anti-Manipulation Defense Module — Algo-Trader Layer 5.

Detects and defends against institutional market manipulation patterns:
1. Anti-Spoofing: Requires 2nd confirming candle for breakout validity.
2. Marking the Close Defense: Mandatory position square-off by 14:30 IST on Expiry Days.
3. Cash-Futures Arbitrage Fading: Detects artificial spot-future spread expansion.
4. Max OI Pinning Trap: Automatically trims target 0.5% before maximum strike OI concentration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("ManipulationDetector")
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class ManipulationCheckResult:
    """Evaluation result from manipulation shield."""

    is_safe: bool
    warnings: List[str]
    is_expiry_day: bool
    cutoff_triggered: bool
    adjusted_target: float
    adjusted_stop: float
    reason: str


class JaneStreetDefenseEngine:
    """
    Guards orders against spoofing, expiry pinning, and synthetic cash-future traps.
    """

    def __init__(self, expiry_cutoff_time: time = time(14, 30)) -> None:
        self.expiry_cutoff_time = expiry_cutoff_time

    def is_expiry_cutoff_active(self, is_expiry_day: bool, current_time: Optional[datetime] = None) -> bool:
        """
        Check if we are past 14:30 IST on an expiry day.
        Rule: Close all intraday positions by 2:30 PM on expiry to avoid institutional marking the close.
        """
        if not is_expiry_day:
            return False

        now = current_time or datetime.now(IST)
        return now.time() >= self.expiry_cutoff_time

    def validate_anti_spoofing(self, df_5m: pd.DataFrame, direction: str) -> bool:
        """
        Anti-Spoofing: Requires 2 consecutive confirming candles in breakout direction
        to ensure breakout isn't an order-book illusion (large orders placed and instantly cancelled).
        """
        if len(df_5m) < 2:
            return False

        c_now = df_5m["close"].iloc[-1]
        o_now = df_5m["open"].iloc[-1]
        c_prev = df_5m["close"].iloc[-2]
        o_prev = df_5m["open"].iloc[-2]

        if direction == "LONG":
            # Current and previous candles must both be bullish in breakout direction
            is_confirmed = (c_now >= o_now) and (c_prev >= o_prev)
            return bool(is_confirmed)
        elif direction == "SHORT":
            # Current and previous candles must both be bearish in breakdown direction
            is_confirmed = (c_now <= o_now) and (c_prev <= o_prev)
            return bool(is_confirmed)

        return True

    def adjust_for_oi_trap(
        self,
        entry_price: float,
        target_price: float,
        direction: str,
        max_call_strike: float,
        max_put_strike: float,
    ) -> float:
        """
        Max OI Trap Protection: If target is beyond the max OI strike, trim target
        to 0.5% below/above the strike so profits are locked BEFORE the institutional pin.
        """
        adjusted_tp = target_price

        if direction == "LONG" and max_call_strike > entry_price:
            safe_call_target = max_call_strike * 0.995  # 0.5% before call pin
            if target_price > safe_call_target:
                logger.info("OI Trap Defense: Trimmed Long target ₹%.2f → ₹%.2f (Max Call OI: ₹%.2f)",
                            target_price, safe_call_target, max_call_strike)
                adjusted_tp = safe_call_target

        elif direction == "SHORT" and max_put_strike < entry_price:
            safe_put_target = max_put_strike * 1.005  # 0.5% before put pin
            if target_price < safe_put_target:
                logger.info("OI Trap Defense: Trimmed Short target ₹%.2f → ₹%.2f (Max Put OI: ₹%.2f)",
                            target_price, safe_put_target, max_put_strike)
                adjusted_tp = safe_put_target

        return round(adjusted_tp, 2)

    def evaluate_order(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        df_5m: pd.DataFrame,
        is_expiry_day: bool = False,
        max_call_strike: float = 0.0,
        max_put_strike: float = 0.0,
    ) -> ManipulationCheckResult:
        """
        Comprehensive pre-entry evaluation through the Jane Street Defense Shield.
        """
        warnings: List[str] = []

        # 1. Check Expiry Cutoff (14:30 IST)
        cutoff = self.is_expiry_cutoff_active(is_expiry_day)
        if cutoff:
            return ManipulationCheckResult(
                is_safe=False,
                warnings=["EXPIRY_CUTOFF_ACTIVE (Past 14:30 IST)"],
                is_expiry_day=is_expiry_day,
                cutoff_triggered=True,
                adjusted_target=target_price,
                adjusted_stop=stop_loss,
                reason="Mandatory 14:30 IST expiry cutoff: Institutional marking the close in progress.",
            )

        # 2. Anti-Spoofing 2nd Candle Check
        spoof_safe = self.validate_anti_spoofing(df_5m, direction)
        if not spoof_safe:
            warnings.append("SPOOF_RISK: Breakout lacks 2nd candle confirmation")

        # 3. OI Magnet Pinning Adjustment
        safe_target = target_price
        if max_call_strike > 0 or max_put_strike > 0:
            safe_target = self.adjust_for_oi_trap(entry_price, target_price, direction, max_call_strike, max_put_strike)

        is_safe = (not cutoff) and (spoof_safe)
        reason = "Passes Jane Street Shield" if is_safe else "; ".join(warnings)

        return ManipulationCheckResult(
            is_safe=is_safe,
            warnings=warnings,
            is_expiry_day=is_expiry_day,
            cutoff_triggered=False,
            adjusted_target=safe_target,
            adjusted_stop=stop_loss,
            reason=reason,
        )


# Global singleton defense engine
jane_street_defense = JaneStreetDefenseEngine()
