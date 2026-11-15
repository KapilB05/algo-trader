

"""
Unit Tests for Kelly Criterion Capital Allocator (Layer 6).
"""

import pytest
from src.risk.capital_allocator import CapitalAllocator


def test_kelly_formula_positive_edge():
    """Verify Kelly percentage with 60% win rate and 1:2 R:R."""
    allocator = CapitalAllocator(total_capital=1000000.0, max_risk_per_trade=0.02)
    # Win rate = 60%, Avg win = 2.0R, Avg loss = 1.0R
    # Raw Kelly = (0.60 * 2 - 0.40 * 1) / 2 = (1.2 - 0.4) / 2 = 0.8 / 2 = 0.40 (40%)
    metrics = allocator.calculate_kelly_fraction(win_rate=0.60, avg_win=2.0, avg_loss=1.0)

    assert metrics.raw_kelly_pct == 40.0
    # Clamped at max 2.0% per trade ceiling
    assert metrics.allocated_risk_pct <= 2.0
    assert metrics.allocated_capital <= 20000.0


def test_kelly_zero_edge():
    """Verify Kelly returns 0.5% (minimum baseline) if win rate has no statistical edge."""
    allocator = CapitalAllocator(total_capital=1000000.0)
    # Win rate = 30%, Avg win = 1.0R, Avg loss = 1.0R -> Negative edge
    metrics = allocator.calculate_kelly_fraction(win_rate=0.30, avg_win=1.0, avg_loss=1.0)
    assert metrics.raw_kelly_pct == 0.0
    assert metrics.allocated_risk_pct == 0.5  # Clamped to min baseline


def test_position_sizing_banknifty_lot_multiple():
    """Verify BankNifty sizing produces valid multiples of lot size (15)."""
    allocator = CapitalAllocator(total_capital=1000000.0, max_risk_per_trade=0.02)
    res = allocator.compute(entry_price=57500.0, stop_loss=57250.0, confidence=1.0)

    assert res.quantity % 15 == 0, f"Quantity {res.quantity} is not a multiple of 15"
    assert res.risk_amount <= 20000.0, f"Risk {res.risk_amount} exceeds ₹20,000"
