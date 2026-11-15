"""
Options Strategy Engine — VIX-driven option strategy selector.

High VIX (>14): Directional Option Buying (ATM/ITM) for momentum capture.
Low VIX (<12):  Option Selling / Credit Spreads to extract Theta decay.
Supports auto (VIX-driven) and manual lock modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from src.broker.instrument_master import InstrumentMaster
from src.strategy.regime_detector import MarketRegime, RegimeDetector

logger = logging.getLogger(__name__)


class OptionStrategy(str, Enum):
    BUY_ATM_CALL = "buy_atm_call"
    BUY_ATM_PUT = "buy_atm_put"
    BUY_ITM_CALL = "buy_itm_call"
    BUY_ITM_PUT = "buy_itm_put"
    SELL_OTM_CALL = "sell_otm_call"
    SELL_OTM_PUT = "sell_otm_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    SHORT_STRADDLE = "short_straddle"
    SHORT_STRANGLE = "short_strangle"


class StrategyMode(str, Enum):
    AUTO = "auto"
    BUY_ONLY = "buy_only"
    SELL_ONLY = "sell_only"


@dataclass
class OptionTradeSetup:
    """Complete option trade setup ready for execution."""

    strategy: OptionStrategy
    underlying: str
    expiry: date
    legs: list[dict]  # Each leg: {instrument_token, strike, type, action, qty}
    rationale: str
    regime: MarketRegime


class OptionsEngine:
    """
    VIX-driven option strategy selection engine.

    In AUTO mode, strategy is determined by current VIX regime.
    In BUY_ONLY or SELL_ONLY mode, strategy is locked regardless of VIX.
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        instrument_master: InstrumentMaster,
        mode: StrategyMode = StrategyMode.AUTO,
    ) -> None:
        self._regime = regime_detector
        self._instruments = instrument_master
        self._mode = mode

    @property
    def mode(self) -> StrategyMode:
        return self._mode

    def set_mode(self, mode: StrategyMode) -> None:
        """Hot-reload strategy mode."""
        self._mode = mode
        logger.info("Options strategy mode set to: %s", mode.value)

    def select_strategy(
        self,
        underlying: str,
        direction: str,  # "LONG" or "SHORT"
        spot_price: float,
    ) -> Optional[OptionTradeSetup]:
        """
        Select the optimal option strategy based on regime and direction.
        """
        regime = self._regime.current_regime
        expiry = self._instruments.get_next_weekly_expiry(underlying)

        if expiry is None:
            logger.warning("No weekly expiry found for %s", underlying)
            return None

        # Determine strategy based on mode
        if self._mode == StrategyMode.BUY_ONLY:
            return self._build_buying_setup(
                underlying, direction, spot_price, expiry, regime
            )
        elif self._mode == StrategyMode.SELL_ONLY:
            return self._build_selling_setup(
                underlying, direction, spot_price, expiry, regime
            )
        else:
            # AUTO: VIX-driven selection
            if regime == MarketRegime.HIGH_VIX:
                return self._build_buying_setup(
                    underlying, direction, spot_price, expiry, regime
                )
            elif regime == MarketRegime.LOW_VIX:
                return self._build_selling_setup(
                    underlying, direction, spot_price, expiry, regime
                )
            else:
                # Neutral: prefer buying with smaller size
                return self._build_buying_setup(
                    underlying, direction, spot_price, expiry, regime
                )

    def _build_buying_setup(
        self,
        underlying: str,
        direction: str,
        spot: float,
        expiry: date,
        regime: MarketRegime,
    ) -> OptionTradeSetup:
        """Build directional option buying setup (ATM/ITM)."""
        # Round spot to nearest strike
        strike_step = 50 if underlying.upper() == "NIFTY" else 100
        atm_strike = round(spot / strike_step) * strike_step

        if direction == "LONG":
            # Buy ATM Call
            opt_type = "CE"
            strategy = OptionStrategy.BUY_ATM_CALL
            strike = atm_strike
        else:
            # Buy ATM Put
            opt_type = "PE"
            strategy = OptionStrategy.BUY_ATM_PUT
            strike = atm_strike

        token = self._instruments.resolve_option_strike(
            underlying, expiry, strike, opt_type
        )

        legs = [
            {
                "instrument_token": token or f"{underlying}_{expiry}_{strike}_{opt_type}",
                "strike": strike,
                "type": opt_type,
                "action": "BUY",
                "qty": 1,  # Lot size determined by position sizer
            }
        ]

        return OptionTradeSetup(
            strategy=strategy,
            underlying=underlying,
            expiry=expiry,
            legs=legs,
            rationale=f"{regime.value}: Directional {opt_type} buying at ATM {strike}",
            regime=regime,
        )

    def _build_selling_setup(
        self,
        underlying: str,
        direction: str,
        spot: float,
        expiry: date,
        regime: MarketRegime,
    ) -> OptionTradeSetup:
        """Build credit spread / selling setup (OTM)."""
        strike_step = 50 if underlying.upper() == "NIFTY" else 100
        atm_strike = round(spot / strike_step) * strike_step

        if direction == "LONG":
            # Bull Call Spread: sell OTM put
            sell_strike = atm_strike - (2 * strike_step)
            buy_strike = sell_strike - strike_step
            sell_type = "PE"
            buy_type = "PE"
            strategy = OptionStrategy.BEAR_PUT_SPREAD
        else:
            # Bear Put Spread: sell OTM call
            sell_strike = atm_strike + (2 * strike_step)
            buy_strike = sell_strike + strike_step
            sell_type = "CE"
            buy_type = "CE"
            strategy = OptionStrategy.BULL_CALL_SPREAD

        sell_token = self._instruments.resolve_option_strike(
            underlying, expiry, sell_strike, sell_type
        )
        buy_token = self._instruments.resolve_option_strike(
            underlying, expiry, buy_strike, buy_type
        )

        legs = [
            {
                "instrument_token": sell_token or f"{underlying}_{expiry}_{sell_strike}_{sell_type}",
                "strike": sell_strike,
                "type": sell_type,
                "action": "SELL",
                "qty": 1,
            },
            {
                "instrument_token": buy_token or f"{underlying}_{expiry}_{buy_strike}_{buy_type}",
                "strike": buy_strike,
                "type": buy_type,
                "action": "BUY",
                "qty": 1,
            },
        ]

        return OptionTradeSetup(
            strategy=strategy,
            underlying=underlying,
            expiry=expiry,
            legs=legs,
            rationale=f"{regime.value}: Credit spread — sell {sell_strike} {sell_type}, buy {buy_strike} {buy_type}",
            regime=regime,
        )
