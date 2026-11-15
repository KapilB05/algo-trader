"""
Hard Stop-Loss Calculator — Pre-trade stop validation.

Validates R:R ≥ 1:2 before any order dispatch.
Uses ATR-based or key-level stop placement.
Blocks order if R:R check fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StopLossSetup:
    """Pre-trade stop-loss configuration."""

    entry_price: float
    stop_loss: float
    target: float
    risk: float
    reward: float
    rr_ratio: float
    method: str  # "atr" | "key_level" | "fixed"
    valid: bool  # True if R:R >= min_rr


class HardStopCalculator:
    """
    Pre-trade stop-loss calculator and R:R validator.

    Blocks order dispatch if R:R ratio < minimum threshold.
    """

    def __init__(self, min_rr_ratio: float = 2.0) -> None:
        self._min_rr = min_rr_ratio

    def set_min_rr(self, ratio: float) -> None:
        self._min_rr = ratio

    def compute_atr_stop(
        self,
        entry_price: float,
        direction: str,  # "LONG" or "SHORT"
        atr: float,
        atr_multiplier: float = 1.5,
    ) -> StopLossSetup:
        """Compute stop-loss and target using ATR."""
        risk_distance = atr * atr_multiplier

        if direction == "LONG":
            stop_loss = entry_price - risk_distance
            target = entry_price + (risk_distance * self._min_rr)
        else:
            stop_loss = entry_price + risk_distance
            target = entry_price - (risk_distance * self._min_rr)

        return self._build_setup(entry_price, stop_loss, target, "atr")

    def compute_level_stop(
        self,
        entry_price: float,
        direction: str,
        support: float,
        resistance: float,
        buffer_pct: float = 0.001,
    ) -> StopLossSetup:
        """Compute stop-loss using key support/resistance levels."""
        buffer = entry_price * buffer_pct

        if direction == "LONG":
            stop_loss = support - buffer
            reward_distance = abs(entry_price - stop_loss) * self._min_rr
            target = entry_price + reward_distance
        else:
            stop_loss = resistance + buffer
            reward_distance = abs(stop_loss - entry_price) * self._min_rr
            target = entry_price - reward_distance

        return self._build_setup(entry_price, stop_loss, target, "key_level")

    def compute_fixed_stop(
        self,
        entry_price: float,
        direction: str,
        stop_points: float,
    ) -> StopLossSetup:
        """Compute stop-loss using fixed point distance."""
        if direction == "LONG":
            stop_loss = entry_price - stop_points
            target = entry_price + (stop_points * self._min_rr)
        else:
            stop_loss = entry_price + stop_points
            target = entry_price - (stop_points * self._min_rr)

        return self._build_setup(entry_price, stop_loss, target, "fixed")

    def validate_custom(
        self,
        entry_price: float,
        stop_loss: float,
        target: float,
    ) -> StopLossSetup:
        """Validate a custom stop-loss/target setup."""
        return self._build_setup(entry_price, stop_loss, target, "custom")

    def _build_setup(
        self,
        entry: float,
        stop: float,
        target: float,
        method: str,
    ) -> StopLossSetup:
        """Build and validate a stop-loss setup."""
        risk = abs(entry - stop)
        reward = abs(target - entry)
        rr = reward / risk if risk > 0 else 0.0
        valid = rr >= self._min_rr

        setup = StopLossSetup(
            entry_price=entry,
            stop_loss=round(stop, 2),
            target=round(target, 2),
            risk=round(risk, 2),
            reward=round(reward, 2),
            rr_ratio=round(rr, 2),
            method=method,
            valid=valid,
        )

        if not valid:
            logger.warning(
                "R:R check FAILED: %.1f < %.1f | Entry=%.2f SL=%.2f TGT=%.2f",
                rr, self._min_rr, entry, stop, target,
            )
        else:
            logger.debug(
                "R:R check PASSED: %.1f | Entry=%.2f SL=%.2f TGT=%.2f",
                rr, entry, stop, target,
            )

        return setup
