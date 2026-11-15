"""
Kelly Criterion Dynamic Position Sizing Engine.
Based on empirical research by Wu et al. (2022) for Indian index options.

Mathematical Framework:
    Full Kelly: f* = p - (1 - p) / b
    Fractional Kelly: f = kappa * f*  (default kappa = 0.25, Quarter-Kelly)
    Hard Capital Constraint: f_clamped = min(f, max_risk_pct)  (Master Arch v2.1: max 2% per trade)

Where:
    p = Model or historical win rate (0.0 to 1.0)
    q = 1 - p (loss probability)
    b = Reward-to-Risk ratio (e.g., 2.0 for 1:2 R:R)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class KellySizingResult:
    """Result of Kelly Criterion position size calculation."""

    win_rate_pct: float
    reward_risk_ratio: float
    full_kelly_fraction: float
    fractional_kelly: float
    clamped_risk_pct: float
    capital_rupees: float
    risk_rupees: float
    option_premium: float
    stop_loss_distance: float
    lot_size: int
    recommended_lots: int
    recommended_qty: int
    total_position_cost: float
    expected_growth_rate: float
    notes: str


class KellyPositionSizer:
    """
    Computes mathematical Kelly Criterion position sizing with institutional risk caps.
    """

    def __init__(
        self,
        default_capital: float = 1000000.0,
        fractional_kelly_mult: float = 0.25,  # Quarter-Kelly for drawdown control
        max_trade_risk_pct: float = 0.02,     # Hard non-negotiable 2% capital cap
    ) -> None:
        self.default_capital = max(1000.0, default_capital)
        self.fractional_kelly_mult = fractional_kelly_mult
        self.max_trade_risk_pct = max_trade_risk_pct

    def compute_size(
        self,
        capital: Optional[float] = None,
        win_rate: float = 0.768,               # 76.8% session win rate
        reward_risk_ratio: float = 2.0,         # 1:2 R:R
        option_premium: float = 300.0,
        stop_loss_pct: float = 0.25,           # 25% Stop loss on premium
        symbol: str = "BANKNIFTY",
    ) -> KellySizingResult:
        """
        Calculate optimal lot allocation using Fractional Kelly with 2% hard clamp.
        """
        cap = capital if capital is not None and capital > 0 else self.default_capital
        p = max(0.01, min(0.99, win_rate))
        b = max(0.5, reward_risk_ratio)
        q = 1.0 - p

        # Official NSE lot size resolution
        sym_upper = symbol.upper()
        if "NIFTY" in sym_upper and "BANK" not in sym_upper:
            lot_size = 65  # Official NSE Nifty 50 lot size post-revision
        else:
            lot_size = 30  # Official NSE Bank Nifty lot size post-revision

        # 1. Full Kelly Fraction: f* = (b*p - q) / b = p - q/b
        full_kelly = max(0.0, (b * p - q) / b)

        # 2. Fractional Kelly (Quarter-Kelly recommended by Wu et al. 2022)
        fractional = full_kelly * self.fractional_kelly_mult

        # 3. Clamped Risk Percentage (Hard 2.0% ceiling)
        clamped_risk_pct = min(self.max_trade_risk_pct, fractional)
        if full_kelly <= 0:
            clamped_risk_pct = 0.0

        risk_rupees = cap * clamped_risk_pct

        # 4. Stop Loss Distance per unit
        sl_distance = max(1.0, option_premium * stop_loss_pct)

        # Risk per 1 lot = lot_size * sl_distance
        risk_per_lot = lot_size * sl_distance

        if risk_per_lot > 0 and risk_rupees > 0:
            raw_lots = risk_rupees / risk_per_lot
            # raw_lots > 0 here, so trunc == floor; avoids stdlib `math`
            # shadowing by src/math package (Pyrefly missing-attribute).
            recommended_lots = max(1, int(raw_lots))
        else:
            recommended_lots = 1

        recommended_qty = recommended_lots * lot_size
        position_cost = recommended_qty * option_premium

        # Expected logarithmic growth rate: g(f) = p*log(1 + f*b) + q*log(1 - f)
        if clamped_risk_pct > 0:
            growth_rate = p * float(np.log(1.0 + clamped_risk_pct * b)) + q * float(np.log(max(0.001, 1.0 - clamped_risk_pct)))
        else:
            growth_rate = 0.0

        notes = (
            f"Wu et al. Kelly: Win Rate {p*100:.1f}%, 1:{b:.1f} R:R. "
            f"Full Kelly {full_kelly*100:.1f}% -> Quarter-Kelly clamped to {clamped_risk_pct*100:.2f}% (Risk: ₹{risk_rupees:,.2f})."
        )

        return KellySizingResult(
            win_rate_pct=round(p * 100.0, 1),
            reward_risk_ratio=round(b, 2),
            full_kelly_fraction=round(full_kelly, 4),
            fractional_kelly=round(fractional, 4),
            clamped_risk_pct=round(clamped_risk_pct * 100.0, 2),
            capital_rupees=round(cap, 2),
            risk_rupees=round(risk_rupees, 2),
            option_premium=round(option_premium, 2),
            stop_loss_distance=round(sl_distance, 2),
            lot_size=lot_size,
            recommended_lots=recommended_lots,
            recommended_qty=recommended_qty,
            total_position_cost=round(position_cost, 2),
            expected_growth_rate=round(growth_rate, 5),
            notes=notes,
        )
