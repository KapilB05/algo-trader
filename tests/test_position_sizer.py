"""Tests for the Position Sizer — 2% risk rule, confidence scaling, manual override."""

from __future__ import annotations

from src.execution.position_sizer import PositionSizer


class TestPositionSizer:
    """Test the 2% risk position sizer."""

    def test_basic_sizing(self):
        """Basic auto sizing: 2% of 100k capital."""
        sizer = PositionSizer(total_capital=100000, max_risk_per_trade=0.02)
        result = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=1.0)

        # Risk per share = 5.0
        # Max risk = 100000 * 0.02 = 2000
        # Quantity = 2000 / 5.0 = 400
        assert result.quantity == 400
        assert result.risk_amount == 2000.0
        assert result.allocation_mode == "auto"

    def test_confidence_scaling(self):
        """Confidence < 1.0 should reduce position size."""
        sizer = PositionSizer(total_capital=100000, max_risk_per_trade=0.02)

        full = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=1.0)
        half = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=0.5)

        assert half.quantity < full.quantity
        assert half.quantity == 200  # 400 * 0.5 = 200

    def test_lot_multiplier(self):
        """F&O lot size rounding."""
        sizer = PositionSizer(total_capital=500000, max_risk_per_trade=0.02)
        result = sizer.compute(
            entry_price=200.0, stop_loss=190.0,
            confidence=1.0, lot_multiplier=25,
        )

        # Risk per share = 10
        # Max risk = 500000 * 0.02 = 10000
        # Raw qty = 10000 / 10 = 1000
        # Rounded to nearest 25 = 1000 (already divisible)
        assert result.quantity % 25 == 0
        assert result.quantity > 0

    def test_manual_mode(self):
        """Manual mode uses fixed lot size."""
        sizer = PositionSizer(
            total_capital=100000,
            allocation_mode="manual",
            manual_lot_size=5,
        )
        result = sizer.compute(entry_price=100.0, stop_loss=95.0)

        assert result.quantity == 5
        assert result.allocation_mode == "manual"

    def test_zero_stop_distance(self):
        """Zero stop distance should return 0 quantity."""
        sizer = PositionSizer(total_capital=100000)
        result = sizer.compute(entry_price=100.0, stop_loss=100.0)

        assert result.quantity == 0

    def test_capital_update(self):
        """Updating capital should change sizing."""
        sizer = PositionSizer(total_capital=100000)
        r1 = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=1.0)

        sizer.update_capital(200000)
        r2 = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=1.0)

        assert r2.quantity == r1.quantity * 2

    def test_risk_percentage(self):
        """Risk should not exceed 2% of capital."""
        sizer = PositionSizer(total_capital=100000, max_risk_per_trade=0.02)
        result = sizer.compute(entry_price=100.0, stop_loss=95.0, confidence=1.0)

        assert result.risk_pct <= 0.02 + 0.001  # Allow tiny float rounding
