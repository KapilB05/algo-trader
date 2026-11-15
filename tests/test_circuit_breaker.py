"""Tests for the Circuit Breaker — 5% drawdown lockout."""

from __future__ import annotations

from src.risk.circuit_breaker import CircuitBreaker


class TestCircuitBreaker:
    """Test the daily drawdown circuit breaker."""

    def test_initial_state(self):
        """Circuit breaker starts unlocked."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)
        assert not cb.is_locked
        assert cb.current_drawdown_pct == 0.0

    def test_triggers_at_threshold(self):
        """Lockout triggers when drawdown hits 5%."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)

        # Lose ₹4999 — still under threshold
        triggered = cb.update_pnl(-4999)
        assert not triggered
        assert not cb.is_locked

        # Lose ₹5001 — over threshold
        triggered = cb.update_pnl(-5001)
        assert triggered
        assert cb.is_locked

    def test_stays_locked_after_recovery(self):
        """Once locked, stays locked even if P&L recovers."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)
        cb.update_pnl(-6000)
        assert cb.is_locked

        # P&L recovers but lock persists
        cb.update_pnl(-2000)
        assert cb.is_locked  # Still locked until next day

    def test_remaining_budget(self):
        """Remaining loss budget tracks correctly."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)
        assert cb.remaining_loss_budget == 5000.0

        cb.update_pnl(-2000)
        assert cb.remaining_loss_budget == 3000.0

    def test_positive_pnl_no_trigger(self):
        """Positive P&L should never trigger lockout."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)
        cb.update_pnl(5000)
        assert not cb.is_locked
        cb.update_pnl(10000)
        assert not cb.is_locked

    def test_force_lock(self):
        """Manual force lock should work."""
        cb = CircuitBreaker(total_capital=100000)
        cb.force_lock()
        assert cb.is_locked

    def test_capital_update(self):
        """Updating capital changes the loss threshold."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)
        assert cb.remaining_loss_budget == 5000.0

        cb.update_capital(200000)
        assert cb.remaining_loss_budget == 10000.0

    def test_drawdown_from_peak(self):
        """Drawdown should be tracked from peak P&L."""
        cb = CircuitBreaker(total_capital=100000, daily_drawdown_limit=0.05)

        # First profit, then drawdown from peak
        cb.update_pnl(3000)  # Peak = 3000
        cb.update_pnl(-2000)  # Drawdown from peak = 5000
        # Total drawdown = peak(3000) - current(-2000) = 5000 = 5%
        assert cb.is_locked
