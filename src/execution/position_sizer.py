"""
Position Sizer — 2% risk rule with confidence-weighted scaling.

max_quantity = (capital × max_risk_per_trade) / (entry - stop_loss)
Scaled by signal confidence score (0.0–1.0).
Supports auto (formula) and manual (lot override) allocation modes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SizingResult:
    """Result of position sizing computation."""

    quantity: int
    risk_amount: float        # Actual ₹ risk for this trade
    risk_pct: float           # Risk as % of capital
    confidence: float
    allocation_mode: str      # auto | manual
    raw_quantity: float       # Pre-rounding quantity


class PositionSizer:
    """
    Computes optimal position size based on the 2% risk rule.

    In AUTO mode: quantity = (capital × risk%) / |entry - stop_loss| × confidence
    In MANUAL mode: uses fixed lot_size from config.
    """

    def __init__(
        self,
        total_capital: float,
        max_risk_per_trade: float = 0.02,
        allocation_mode: str = "auto",
        manual_lot_size: int = 1,
    ) -> None:
        self._capital = total_capital
        self._max_risk = max_risk_per_trade
        self._mode = allocation_mode
        self._manual_lot = manual_lot_size

    def update_capital(self, capital: float) -> None:
        """Update total capital (e.g., after P&L changes)."""
        self._capital = capital

    def set_mode(self, mode: str, manual_lot: int = 1) -> None:
        """Hot-reload allocation mode."""
        self._mode = mode
        self._manual_lot = manual_lot

    def compute(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        lot_multiplier: int = 1,
    ) -> SizingResult:
        """
        Compute position size.

        Args:
            entry_price: Expected entry price.
            stop_loss: Hard stop-loss price.
            confidence: Signal confidence score (0.0–1.0).
            lot_multiplier: F&O lot size multiplier (e.g., Nifty=25).
        """
        if self._mode == "manual":
            return SizingResult(
                quantity=self._manual_lot * lot_multiplier,
                risk_amount=abs(entry_price - stop_loss) * self._manual_lot * lot_multiplier,
                risk_pct=(abs(entry_price - stop_loss) * self._manual_lot * lot_multiplier) / self._capital if self._capital > 0 else 0,
                confidence=confidence,
                allocation_mode="manual",
                raw_quantity=float(self._manual_lot * lot_multiplier),
            )

        # ── Auto allocation: 2% risk rule ─────────────────────────────
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            logger.warning("Invalid stop distance: entry=%.2f, SL=%.2f", entry_price, stop_loss)
            return SizingResult(
                quantity=0, risk_amount=0, risk_pct=0,
                confidence=confidence, allocation_mode="auto", raw_quantity=0,
            )

        max_risk_amount = self._capital * self._max_risk
        raw_qty = (max_risk_amount / risk_per_share) * confidence

        # Round down to nearest lot size
        if lot_multiplier > 1:
            quantity = max(lot_multiplier, int(raw_qty // lot_multiplier) * lot_multiplier)
        else:
            quantity = max(1, int(raw_qty))

        actual_risk = risk_per_share * quantity
        risk_pct = actual_risk / self._capital if self._capital > 0 else 0

        result = SizingResult(
            quantity=quantity,
            risk_amount=actual_risk,
            risk_pct=risk_pct,
            confidence=confidence,
            allocation_mode="auto",
            raw_quantity=raw_qty,
        )

        logger.info(
            "Position size: %d units | Risk: ₹%.0f (%.1f%%) | Confidence: %.2f",
            quantity, actual_risk, risk_pct * 100, confidence,
        )

        return result
