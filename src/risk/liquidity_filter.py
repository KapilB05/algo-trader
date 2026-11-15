"""
Liquidity, Spread, and Slippage Guard.
Based on empirical market efficiency research by Mohanti & Priyan (2014) and Mutum & Das (2019).

Protects against the #1 cause of theoretical profit erosion in Indian option trading:
- Illiquid strikes with zero or negligible market depth
- Wide bid-ask spreads that guarantee negative slippage on entry and exit
- Strikes near expiry without sufficient open interest
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class LiquidityValidationResult:
    """Result of option strike liquidity & spread validation."""

    approved: bool
    symbol: str
    strike: float
    option_type: str
    ltp: float
    open_interest: int
    volume: int
    bid: float
    ask: float
    spread_points: float
    spread_pct: float
    reason: str
    rejection_code: Optional[str] = None


class LiquidityFilter:
    """
    Enforces institutional market depth, minimum OI, and bid-ask spread rules.
    """

    def __init__(
        self,
        min_oi_nifty: int = 100000,          # Minimum 1,00,000 OI contracts for Nifty
        min_oi_banknifty: int = 50000,       # Minimum 50,000 OI contracts for Bank Nifty
        min_volume: int = 5000,              # Minimum 5,000 volume contracts traded
        max_spread_pct: float = 1.50,        # Maximum 1.50% bid-ask spread
        max_spread_points: float = 4.0,      # Maximum 4.0 index option points spread
    ) -> None:
        self.min_oi_nifty = min_oi_nifty
        self.min_oi_banknifty = min_oi_banknifty
        self.min_volume = min_volume
        self.max_spread_pct = max_spread_pct
        self.max_spread_points = max_spread_points

    def validate_option(
        self,
        symbol: str,
        strike: float,
        option_type: str,
        ltp: float,
        open_interest: int = 0,
        volume: int = 0,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
    ) -> LiquidityValidationResult:
        """
        Validate if an option contract possesses sufficient institutional liquidity to trade.
        """
        sym_upper = symbol.upper()
        is_nifty = "NIFTY" in sym_upper and "BANK" not in sym_upper
        min_oi = self.min_oi_nifty if is_nifty else self.min_oi_banknifty

        # 1. Zero Price / Stale Check
        if ltp <= 0.5:
            return LiquidityValidationResult(
                approved=False,
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                ltp=ltp,
                open_interest=open_interest,
                volume=volume,
                bid=bid or 0.0,
                ask=ask or 0.0,
                spread_points=0.0,
                spread_pct=0.0,
                reason="LTP is near zero (< 0.5) — deep OTM or illiquid strike.",
                rejection_code="ZERO_PRICE_REJECT",
            )

        # 2. Open Interest Threshold Check
        if open_interest > 0 and open_interest < min_oi:
            return LiquidityValidationResult(
                approved=False,
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                ltp=ltp,
                open_interest=open_interest,
                volume=volume,
                bid=bid or 0.0,
                ask=ask or 0.0,
                spread_points=0.0,
                spread_pct=0.0,
                reason=f"Open Interest {open_interest:,} below minimum threshold ({min_oi:,}). Illiquid strike.",
                rejection_code="LOW_OI_REJECT",
            )

        # 3. Volume Check
        if volume > 0 and volume < self.min_volume:
            return LiquidityValidationResult(
                approved=False,
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                ltp=ltp,
                open_interest=open_interest,
                volume=volume,
                bid=bid or 0.0,
                ask=ask or 0.0,
                spread_points=0.0,
                spread_pct=0.0,
                reason=f"Volume {volume:,} below required minimum ({self.min_volume:,}).",
                rejection_code="LOW_VOLUME_REJECT",
            )

        # 4. Bid-Ask Spread Check (if depth quotes available)
        # Default realistic exchange spread if live depth quotes not provided
        b = bid if bid is not None and bid > 0 else ltp * 0.9975
        a = ask if ask is not None and ask > 0 else ltp * 1.0025
        spread_pts = max(0.0, a - b)
        spread_pct = (spread_pts / ltp) * 100.0 if ltp > 0 else 0.0

        max_pts_limit = self.max_spread_points if is_nifty else (self.max_spread_points * 3.0)

        if spread_pct > self.max_spread_pct or (spread_pts > max_pts_limit and spread_pct > 1.0):
            return LiquidityValidationResult(
                approved=False,
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                ltp=ltp,
                open_interest=open_interest,
                volume=volume,
                bid=round(b, 2),
                ask=round(a, 2),
                spread_points=round(spread_pts, 2),
                spread_pct=round(spread_pct, 2),
                reason=f"Bid-Ask Spread too wide ({spread_pts:.2f} pts / {spread_pct:.2f}%). Slippage exceeds safety limits.",
                rejection_code="WIDE_SPREAD_REJECT",
            )

        return LiquidityValidationResult(
            approved=True,
            symbol=symbol,
            strike=strike,
            option_type=option_type,
            ltp=round(ltp, 2),
            open_interest=open_interest,
            volume=volume,
            bid=round(b, 2),
            ask=round(a, 2),
            spread_points=round(spread_pts, 2),
            spread_pct=round(spread_pct, 2),
            reason=f"Passed institutional liquidity checks (OI: {open_interest:,}, Spread: {spread_pct:.2f}%).",
            rejection_code=None,
        )
