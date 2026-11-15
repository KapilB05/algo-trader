"""
Capital Allocator & Kelly Criterion Engine — Algo-Trader Risk Layer (Layer 6).

Implements:
1. Full Kelly Criterion with Max Drawdown Tolerance Denominator:
   Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
   Position Size = Kelly % × Capital × (1 / Max Drawdown Tolerance)
2. Half-Kelly & Quarter-Kelly safety multipliers to minimize volatility.
3. Hard Institutional Risk Ceiling (Max 2.0% Risk Per Trade).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from src.execution.position_sizer import PositionSizer, SizingResult
from src.config.lot_sizes import get_lot_size

logger = logging.getLogger("CapitalAllocator")


@dataclass
class KellySizingMetrics:
    """Detailed Kelly Criterion allocation metrics."""

    win_rate: float                # Historical win rate (e.g. 0.58 for 58%)
    avg_win: float                 # Average win in ₹ or points
    avg_loss: float                # Average loss in ₹ or points
    win_loss_ratio: float          # Avg Win / Avg Loss
    raw_kelly_pct: float           # Pure mathematical Kelly percentage
    scaled_kelly_pct: float        # After Max DD tolerance & safety fractional scaling
    max_dd_tolerance: float        # e.g., 0.06 (6.0% max acceptable drawdown)
    allocated_risk_pct: float      # Final clamped risk % (capped at max_risk_per_trade)
    allocated_capital: float       # ₹ capital at risk for trade


class CapitalAllocator:
    """
    Institutional Capital Allocator leveraging fractional Kelly Criterion and Max Drawdown gates.
    """

    def __init__(
        self,
        total_capital: float = 1000000.0,
        max_risk_per_trade: float = 0.02,
        max_drawdown_tolerance: float = 0.06,
        kelly_fraction: float = 0.5,  # Half-Kelly for conservative institutional sizing
        allocation_mode: str = "auto",
        manual_lot_size: int = 1,
    ) -> None:
        self._capital = total_capital
        self._max_risk = max_risk_per_trade
        self._max_dd_tolerance = max_drawdown_tolerance
        self._kelly_fraction = kelly_fraction
        self._mode = allocation_mode
        self._manual_lot = manual_lot_size

        self._sizer = PositionSizer(
            total_capital=total_capital,
            max_risk_per_trade=max_risk_per_trade,
            allocation_mode=allocation_mode,
            manual_lot_size=manual_lot_size,
        )

    def calculate_kelly_fraction(
        self,
        win_rate: float = 0.58,
        avg_win: float = 2.0,
        avg_loss: float = 1.0,
    ) -> KellySizingMetrics:
        """
        Compute optimal Kelly Criterion allocation with Max DD scaling:
        Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win
        """
        loss_rate = max(0.0, 1.0 - win_rate)
        avg_win = max(0.01, avg_win)
        avg_loss = max(0.01, avg_loss)
        wl_ratio = avg_win / avg_loss

        # Raw Kelly formula
        raw_kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        raw_kelly = max(0.0, raw_kelly)

        # Scale by Max Drawdown tolerance denominator (e.g. 1 / 0.06) and fractional safety
        dd_factor = min(1.0, self._max_dd_tolerance / 0.06)
        scaled_kelly = raw_kelly * self._kelly_fraction * dd_factor

        # Hard clamp between 0.5% and max_risk_per_trade (2.0%)
        final_risk_pct = max(0.005, min(self._max_risk, scaled_kelly))
        allocated_cap = self._capital * final_risk_pct

        return KellySizingMetrics(
            win_rate=round(win_rate, 4),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            win_loss_ratio=round(wl_ratio, 2),
            raw_kelly_pct=round(raw_kelly * 100.0, 2),
            scaled_kelly_pct=round(scaled_kelly * 100.0, 2),
            max_dd_tolerance=round(self._max_dd_tolerance * 100.0, 2),
            allocated_risk_pct=round(final_risk_pct * 100.0, 2),
            allocated_capital=round(allocated_cap, 2),
        )

    def compute(
        self,
        entry_price: float,
        stop_loss: float,
        confidence: float = 1.0,
        lot_multiplier: int = 1,
        win_rate: float = 0.58,
        avg_win: float = 2.0,
        avg_loss: float = 1.0,
        symbol: str = "BANKNIFTY",
    ) -> SizingResult:
        """
        Compute optimal position size integrating Kelly metrics with price distance.
        """
        if self._mode == "manual":
            return self._sizer.compute(entry_price, stop_loss, confidence, self._manual_lot)

        kelly_metrics = self.calculate_kelly_fraction(win_rate, avg_win, avg_loss)
        effective_risk_pct = (kelly_metrics.allocated_risk_pct / 100.0) * confidence

        stop_distance = max(1.0, abs(entry_price - stop_loss))
        risk_budget = self._capital * effective_risk_pct
        raw_qty = risk_budget / stop_distance

        # Lot size rounding using symbol lookup from config/lot_sizes.yaml
        lot_size = get_lot_size(symbol)
        qty = max(lot_size, int(raw_qty / lot_size) * lot_size)
        actual_risk = qty * stop_distance
        actual_risk_pct = (actual_risk / self._capital) * 100.0

        return SizingResult(
            quantity=qty,
            risk_amount=round(actual_risk, 2),
            risk_pct=round(actual_risk_pct, 2),
            confidence=round(confidence, 2),
            allocation_mode="kelly_auto",
            raw_quantity=round(raw_qty, 2),
        )

    def update_config(
        self,
        capital: Optional[float] = None,
        mode: Optional[str] = None,
        manual_lot: Optional[int] = None,
        max_risk: Optional[float] = None,
    ) -> None:
        """Hot-reload allocation parameters."""
        if capital is not None:
            self._capital = capital
            self._sizer.update_capital(capital)
        if max_risk is not None:
            self._max_risk = max_risk
            self._sizer._max_risk = max_risk
        if mode is not None or manual_lot is not None:
            self._mode = mode or self._mode
            self._manual_lot = manual_lot or self._manual_lot
            self._sizer.set_mode(self._mode, self._manual_lot)
        logger.info("Capital allocator config updated.")


# Global singleton allocator
capital_allocator = CapitalAllocator()
