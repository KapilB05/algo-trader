"""
Live Performance Monitor — Algo-Trader Risk & Degradation Guardrail.

Monitors rolling win rate over the last N trades. Automatically triggers size reduction
or trading pauses if rolling accuracy drops below safety baselines.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    trade_id: str
    symbol: str
    pnl: float
    is_win: bool
    confidence: float


class LivePerformanceMonitor:
    """
    Monitors rolling trade performance to detect model accuracy degradation.
    """

    def __init__(
        self,
        window_size: int = 20,
        min_win_rate_threshold: float = 0.40,
        degradation_size_multiplier: float = 0.50,
    ) -> None:
        self.window_size = window_size
        self.min_win_rate = min_win_rate_threshold
        self.degradation_mult = degradation_size_multiplier

        self._history: deque[TradeOutcome] = deque(maxlen=window_size)
        self._is_degraded: bool = False
        self._is_paused: bool = False

    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        pnl: float,
        is_win: bool,
        confidence: float = 1.0,
    ) -> None:
        """Record a completed trade outcome."""
        outcome = TradeOutcome(
            trade_id=trade_id,
            symbol=symbol,
            pnl=pnl,
            is_win=is_win,
            confidence=confidence,
        )
        self._history.append(outcome)
        self._evaluate_performance()

    def _evaluate_performance(self) -> None:
        """Evaluate rolling metrics and trigger degradation state if needed."""
        if len(self._history) < 5:
            self._is_degraded = False
            self._is_paused = False
            return

        wins = sum(1 for t in self._history if t.is_win)
        win_rate = wins / len(self._history)

        if win_rate < self.min_win_rate:
            if not self._is_degraded:
                logger.warning(
                    "🚨 Performance Degradation Warning! Rolling Win Rate %.1f%% < %.1f%% threshold across last %d trades.",
                    win_rate * 100, self.min_win_rate * 100, len(self._history),
                )
            self._is_degraded = True

            # If win rate falls below 25%, pause trading completely
            if win_rate < 0.25 and len(self._history) >= 10:
                self._is_paused = True
                logger.critical("🚨 TRADING PAUSED! Rolling Win Rate dropped to %.1f%%.", win_rate * 100)
        else:
            if self._is_degraded:
                logger.info("✅ Performance Recovered! Rolling Win Rate: %.1f%%", win_rate * 100)
            self._is_degraded = False
            self._is_paused = False

    @property
    def rolling_win_rate(self) -> float:
        if not self._history:
            return 1.0
        return sum(1 for t in self._history if t.is_win) / len(self._history)

    @property
    def is_degraded(self) -> bool:
        return self._is_degraded

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def get_risk_multiplier(self) -> float:
        """Returns sizing multiplier based on live accuracy state."""
        if self._is_paused:
            return 0.0
        if self._is_degraded:
            return self.degradation_mult
        return 1.0
