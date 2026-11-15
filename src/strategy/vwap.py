"""
VWAP Engine — Volume Weighted Average Price computation.

Computes intraday VWAP from cumulative tick volume and typical price.
Resets daily at 9:15 AM IST (market open).
"""

from __future__ import annotations

import logging
from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

from src.data.protobuf_decoder import DecodedTick

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = dt_time(9, 15)


class VWAPEngine:
    """
    Real-time intraday VWAP calculator.

    VWAP = Cumulative(Price × Volume) / Cumulative(Volume)

    Resets at 9:15 AM IST each day. Provides directional signal:
    price above VWAP = bullish, below = bearish.
    """

    def __init__(self) -> None:
        # Per-instrument accumulators: token → (cum_price_vol, cum_vol, vwap, date)
        self._accumulators: dict[str, dict] = {}

    def on_tick(self, tick: DecodedTick) -> None:
        """Update VWAP accumulator with a new tick."""
        token = tick.instrument_token
        now = datetime.now(IST)

        # Reset on new trading day
        if token in self._accumulators:
            last_date = self._accumulators[token].get("date")
            if last_date and last_date != now.date():
                self._reset(token)

        # Initialize accumulator if needed
        if token not in self._accumulators:
            self._accumulators[token] = {
                "cum_price_vol": 0.0,
                "cum_vol": 0,
                "vwap": 0.0,
                "date": now.date(),
            }

        acc = self._accumulators[token]

        # Typical price for the tick
        typical_price = tick.ltp
        trade_volume = tick.ltq if tick.ltq > 0 else 1

        acc["cum_price_vol"] += typical_price * trade_volume
        acc["cum_vol"] += trade_volume

        if acc["cum_vol"] > 0:
            acc["vwap"] = acc["cum_price_vol"] / acc["cum_vol"]

    def get_vwap(self, instrument_token: str) -> float:
        """Get current VWAP for an instrument."""
        acc = self._accumulators.get(instrument_token)
        if acc is None:
            return 0.0
        return acc["vwap"]

    def is_above_vwap(self, instrument_token: str, current_price: float) -> bool:
        """Check if current price is above VWAP (bullish signal)."""
        vwap = self.get_vwap(instrument_token)
        if vwap <= 0:
            return False
        return current_price > vwap

    def is_below_vwap(self, instrument_token: str, current_price: float) -> bool:
        """Check if current price is below VWAP (bearish signal)."""
        vwap = self.get_vwap(instrument_token)
        if vwap <= 0:
            return False
        return current_price < vwap

    def get_vwap_distance_pct(self, instrument_token: str, current_price: float) -> float:
        """Get percentage distance from VWAP. Positive = above, negative = below."""
        vwap = self.get_vwap(instrument_token)
        if vwap <= 0:
            return 0.0
        return ((current_price - vwap) / vwap) * 100

    def _reset(self, instrument_token: str) -> None:
        """Reset VWAP accumulator for a new trading day."""
        if instrument_token in self._accumulators:
            del self._accumulators[instrument_token]
            logger.debug("VWAP reset for %s (new trading day).", instrument_token)

    def reset_all(self) -> None:
        """Reset all accumulators (e.g., at market open)."""
        self._accumulators.clear()
        logger.info("All VWAP accumulators reset.")
