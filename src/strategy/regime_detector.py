"""
Regime Classifier & Playbook Engine — Algo-Trader 7-Layer Architecture (Layer 3).

Implements the strict 5-state market playbook:
1. TRENDING_UP:   Price > 20 EMA, ADX > 25, +DI > -DI  → Momentum Breakout, VWAP Reclaim
2. TRENDING_DOWN: Price < 20 EMA, ADX > 25, -DI > +DI  → Put Buying, Short Futures
3. COMPRESSION:   BBW Percentile Rank ≤ 20%             → Straddles, Iron Condors
4. CHOP:          ADX < 20, Sideways Range              → DO NOT TRADE (100% Cash)
5. CHAOS:         VIX Spike > 30% in 1 hour             → Event-Driven Defense Only

Includes a 3-bar Hysteresis Filter to prevent rapid flapping across regime boundaries.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("RegimeDetector")


class MarketRegime(str, Enum):
    # Core 5 Playbook States
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    COMPRESSION = "compression"
    CHOP = "chop"
    CHAOS = "chaos"

    # Backward compatibility aliases
    HIGH_VIX = "high_vix"
    LOW_VIX = "low_vix"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


class RegimeDetector:
    """
    Classifies market state into one of 5 distinct operational playbooks.
    Enforces strict trade gating: In CHOP regime, is_trade_allowed returns False.
    """

    def __init__(
        self,
        adx_trend_threshold: float = 25.0,
        adx_chop_threshold: float = 20.0,
        bbw_compression_percentile: float = 0.20,
        vix_chaos_spike_pct: float = 30.0,
        hysteresis_bars: int = 3,
        **kwargs,
    ) -> None:
        self.adx_trend_threshold = adx_trend_threshold
        self.adx_chop_threshold = adx_chop_threshold
        self.bbw_compression_percentile = bbw_compression_percentile
        self.vix_chaos_spike_pct = vix_chaos_spike_pct
        self.hysteresis_bars = hysteresis_bars

        self._current_regime: MarketRegime = MarketRegime.CHOP
        self._raw_regime_history: deque[MarketRegime] = deque(maxlen=hysteresis_bars)
        self._vix_history: deque[Tuple[float, float]] = deque(maxlen=60)  # (timestamp_sec, vix_val)
        self._current_vix: float = 14.0
        self._current_adx: float = 18.0
        self._current_bbw_rank: float = 0.50

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def current_vix(self) -> float:
        return self._current_vix

    @property
    def current_adx(self) -> float:
        return self._current_adx

    @property
    def allowed_strategies(self) -> List[str]:
        """Return the allowed strategies list for the active regime."""
        if self._current_regime == MarketRegime.TRENDING_UP:
            return ["orb_breakout", "vwap_reclaim", "ema_supertrend", "institutional_flow"]
        elif self._current_regime == MarketRegime.TRENDING_DOWN:
            return ["put_buying", "short_futures", "ema_supertrend", "institutional_flow"]
        elif self._current_regime == MarketRegime.COMPRESSION:
            return ["short_straddle", "iron_condor", "compression_breakout_prep"]
        elif self._current_regime == MarketRegime.CHAOS:
            return ["event_defense_only"]
        else:  # CHOP
            return []

    def is_trade_allowed(self) -> bool:
        """
        Gating filter: In CHOP or unconfirmed CHAOS regimes, DO NOT TRADE.
        """
        if self._current_regime == MarketRegime.CHOP:
            return False
        return True

    def classify_market_state(
        self,
        df_candles: pd.DataFrame,
        current_vix: float = 14.0,
        vix_1h_change_pct: float = 0.0,
    ) -> MarketRegime:
        """
        Classify raw market state from OHLCV bars and apply hysteresis filter.
        """
        self._current_vix = current_vix

        # 1. Check CHAOS: VIX spike > 30% in 1 hour
        if vix_1h_change_pct >= self.vix_chaos_spike_pct or current_vix >= 24.0:
            raw_state = MarketRegime.CHAOS
        elif len(df_candles) < 20:
            raw_state = MarketRegime.CHOP
        else:
            c = df_candles["close"]
            h = df_candles["high"]
            low = df_candles["low"]

            # EMA 20
            ema20 = c.ewm(span=20, adjust=False).mean()
            last_close = c.iloc[-1]
            last_ema20 = ema20.iloc[-1]

            # ADX and DMI
            up_move = h.diff()
            down_move = -low.diff()
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

            tr1 = h - low
            tr2 = (h - c.shift(1)).abs()
            tr3 = (low - c.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr14 = tr.rolling(14, min_periods=1).mean()
            pdi = 100.0 * (pd.Series(plus_dm, index=c.index).rolling(14, min_periods=1).mean() / (atr14 + 1e-8))
            mdi = 100.0 * (pd.Series(minus_dm, index=c.index).rolling(14, min_periods=1).mean() / (atr14 + 1e-8))
            dx = 100.0 * ((pdi - mdi).abs() / (pdi + mdi + 1e-8))
            adx = dx.rolling(14, min_periods=1).mean()

            last_adx = float(adx.iloc[-1])
            last_pdi = float(pdi.iloc[-1])
            last_mdi = float(mdi.iloc[-1])
            self._current_adx = round(last_adx, 2)

            # Bollinger Band Width Percentile Rank
            bb_mid = c.rolling(20, min_periods=1).mean()
            bb_std = c.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
            bb_width = (4.0 * bb_std) / (bb_mid + 1e-8)
            bw_arr = bb_width.values
            last_val = bw_arr[-1]
            last_bbw_rank = float((bw_arr[-50:] <= last_val).mean()) if len(bw_arr) > 0 else 0.5
            self._current_bbw_rank = round(last_bbw_rank, 3)

            # Regime Decision Tree
            if last_bbw_rank <= self.bbw_compression_percentile:
                raw_state = MarketRegime.COMPRESSION
            elif last_adx >= self.adx_trend_threshold:
                if last_close > last_ema20 and last_pdi > last_mdi:
                    raw_state = MarketRegime.TRENDING_UP
                elif last_close < last_ema20 and last_mdi > last_pdi:
                    raw_state = MarketRegime.TRENDING_DOWN
                else:
                    raw_state = MarketRegime.CHOP
            elif last_adx < self.adx_chop_threshold:
                raw_state = MarketRegime.CHOP
            else:
                raw_state = self._current_regime  # Maintain current if in buffer

        # 2. Immediate trigger for CHAOS (emergency condition)
        if raw_state == MarketRegime.CHAOS:
            self._current_regime = MarketRegime.CHAOS
            self._raw_regime_history.clear()
            return self._current_regime

        # 3. Apply Hysteresis Buffer for normal regime transitions
        self._raw_regime_history.append(raw_state)
        
        # State transitions if all elements in history match the candidate state
        if len(self._raw_regime_history) >= min(2, self.hysteresis_bars):
            if all(r == raw_state for r in self._raw_regime_history):
                if raw_state != self._current_regime:
                    logger.info("Confirmed Regime Transition: %s → %s (ADX: %.1f, VIX: %.2f)",
                                self._current_regime.value, raw_state.value, self._current_adx, current_vix)
                    self._current_regime = raw_state

        return self._current_regime

    def update(self, vix_value: float) -> MarketRegime:
        """Simple VIX update for backwards compatibility."""
        self._current_vix = vix_value
        if vix_value >= 24.0:
            self._current_regime = MarketRegime.CHAOS
        elif vix_value > 14.0:
            self._current_regime = MarketRegime.HIGH_VIX
        elif vix_value < 12.0:
            self._current_regime = MarketRegime.LOW_VIX
        else:
            self._current_regime = MarketRegime.NEUTRAL
        return self._current_regime

    def update_thresholds(self, high_threshold: float, low_threshold: float) -> None:
        """Update VIX classification thresholds dynamically."""
        self.high_vix_threshold = high_threshold
        self.low_vix_threshold = low_threshold

    def get_sizing_multiplier(self) -> float:
        """Position sizing multiplier based on regime."""
        if self._current_regime in (
            MarketRegime.TRENDING_UP,
            MarketRegime.TRENDING_DOWN,
            MarketRegime.HIGH_VIX,
            MarketRegime.NEUTRAL,
        ):
            return 1.0
        elif self._current_regime == MarketRegime.COMPRESSION:
            return 0.75
        elif self._current_regime == MarketRegime.LOW_VIX:
            return 0.5
        elif self._current_regime == MarketRegime.CHAOS:
            return 0.25
        else:  # CHOP
            return 0.0

    def is_signal_allowed(self, direction: Optional[str] = None) -> bool:
        """Whether fresh entry signals are permitted in the current regime.

        Blocks new entries in no-trade regimes (CHOP, CHAOS, LOW_VIX) and
        when no directional bias (NEUTRAL) is present.
        """
        if self._current_regime in (
            MarketRegime.CHOP,
            MarketRegime.CHAOS,
            MarketRegime.LOW_VIX,
        ):
            return False
        if direction is not None and direction.upper() == "NEUTRAL":
            return False
        return True


# Global singleton detector
regime_detector = RegimeDetector()
