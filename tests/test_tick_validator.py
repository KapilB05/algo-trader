"""Tests for the Tick Validator — zero price, stale, spike filtering."""

from __future__ import annotations

import time

from src.data.protobuf_decoder import DecodedTick
from src.data.tick_validator import TickValidator


class TestTickValidator:
    """Test the tick validation pipeline."""

    def setup_method(self):
        self.validator = TickValidator(
            max_stale_seconds=5.0,
            max_spike_pct=0.05,
            spike_window_seconds=1.0,
        )

    def test_valid_tick_passes(self):
        """A normal tick should pass all checks."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=100.0,
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick) is True
        assert self.validator.metrics.total_passed == 1

    def test_zero_price_rejected(self):
        """Zero-price ticks must be rejected."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=0.0,
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick) is False
        assert self.validator.metrics.zero_price_dropped == 1

    def test_negative_price_rejected(self):
        """Negative prices must be rejected."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=-5.0,
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick) is False
        assert self.validator.metrics.zero_price_dropped == 1

    def test_stale_tick_rejected(self):
        """Ticks older than max_stale_seconds must be rejected."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=100.0,
            ltq=10,
            ltt=time.time() - 10.0,  # 10 seconds old
        )
        assert self.validator.validate(tick) is False
        assert self.validator.metrics.stale_dropped == 1

    def test_spike_detection(self):
        """Price spikes > 5% within 1 second must be rejected."""
        # First tick — establishes baseline
        tick1 = DecodedTick(
            instrument_token="TEST|002",
            ltp=100.0,
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick1) is True

        # Second tick — 10% spike within same second
        tick2 = DecodedTick(
            instrument_token="TEST|002",
            ltp=110.0,  # 10% jump
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick2) is False
        assert self.validator.metrics.spike_dropped == 1

    def test_normal_price_movement_passes(self):
        """Normal price movements within threshold should pass."""
        tick1 = DecodedTick(
            instrument_token="TEST|003",
            ltp=100.0,
            ltq=10,
            ltt=time.time(),
        )
        self.validator.validate(tick1)

        # 2% move — within 5% threshold
        tick2 = DecodedTick(
            instrument_token="TEST|003",
            ltp=102.0,
            ltq=10,
            ltt=time.time(),
        )
        assert self.validator.validate(tick2) is True

    def test_metrics_reset(self):
        """Resetting metrics returns snapshot and clears counters."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=0.0,
            ltq=10,
            ltt=time.time(),
        )
        self.validator.validate(tick)

        snapshot = self.validator.reset_metrics()
        assert snapshot.zero_price_dropped == 1
        assert self.validator.metrics.zero_price_dropped == 0

    def test_no_ltt_skips_stale_check(self):
        """Ticks without a timestamp should skip the stale check."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=100.0,
            ltq=10,
            ltt=0,  # No timestamp
        )
        assert self.validator.validate(tick) is True
