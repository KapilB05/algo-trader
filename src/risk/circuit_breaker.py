"""
Circuit Breaker — 5% daily drawdown lockout.

Monitors cumulative daily P&L. Cancels all pending orders and blocks
new entries for the rest of the day if drawdown exceeds threshold.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class CircuitBreaker:
    """
    Daily drawdown circuit breaker.

    Locks out further trading for the day when total account
    drawdown hits the configured threshold (default 5%).
    """

    def __init__(
        self,
        total_capital: float,
        daily_drawdown_limit: float = 0.05,
    ) -> None:
        self._capital = total_capital
        self._limit = daily_drawdown_limit
        self._max_loss = total_capital * daily_drawdown_limit
        self._is_locked = False
        self._lock_date: Optional[date] = None
        self._peak_pnl: float = 0.0
        self._current_pnl: float = 0.0

    @property
    def is_locked(self) -> bool:
        """Check if circuit breaker is active."""
        # Auto-reset at new trading day
        today = datetime.now(IST).date()
        if self._lock_date and self._lock_date != today:
            self._reset()
        return self._is_locked

    @property
    def current_drawdown_pct(self) -> float:
        """Current drawdown as percentage of capital."""
        if self._capital <= 0:
            return 0.0
        return abs(min(0, self._current_pnl)) / self._capital

    @property
    def remaining_loss_budget(self) -> float:
        """Remaining loss before lockout triggers."""
        current_loss = abs(min(0, self._current_pnl))
        return max(0, self._max_loss - current_loss)

    def update_pnl(self, total_pnl: float) -> bool:
        """
        Update with current P&L. Returns True if lockout triggered.

        Args:
            total_pnl: Total unrealized + realized P&L for the day.
        """
        today = datetime.now(IST).date()

        # Auto-reset on new day
        if self._lock_date and self._lock_date != today:
            self._reset()

        self._current_pnl = total_pnl

        # Track peak P&L
        if total_pnl > self._peak_pnl:
            self._peak_pnl = total_pnl

        # Check drawdown from peak
        drawdown = self._peak_pnl - total_pnl
        drawdown_pct = drawdown / self._capital if self._capital > 0 else 0

        # Also check absolute loss
        absolute_loss_pct = abs(min(0, total_pnl)) / self._capital if self._capital > 0 else 0

        if (drawdown_pct >= self._limit or absolute_loss_pct >= self._limit) and not self._is_locked:
            self._is_locked = True
            self._lock_date = today
            logger.critical(
                "🚨 CIRCUIT BREAKER TRIGGERED! "
                "Drawdown: %.1f%% | P&L: ₹%.0f | Limit: %.1f%%",
                max(drawdown_pct, absolute_loss_pct) * 100,
                total_pnl,
                self._limit * 100,
            )
            return True

        return False

    def update_capital(self, capital: float) -> None:
        """Update total capital (e.g., at start of day)."""
        self._capital = capital
        self._max_loss = capital * self._limit

    def force_lock(self) -> None:
        """Manually trigger the circuit breaker."""
        self._is_locked = True
        self._lock_date = datetime.now(IST).date()
        logger.warning("Circuit breaker manually triggered.")

    def _reset(self) -> None:
        """Reset for a new trading day."""
        self._is_locked = False
        self._lock_date = None
        self._peak_pnl = 0.0
        self._current_pnl = 0.0
        logger.info("Circuit breaker reset for new trading day.")
