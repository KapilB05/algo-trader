"""Tests for the Candle Aggregator — Polars-based tick→candle."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from src.data.candle_aggregator import Candle, CandleAggregator
from src.data.protobuf_decoder import DecodedTick


class TestCandleAggregator:
    """Test Polars-based candle aggregation."""

    def setup_method(self):
        self.aggregator = CandleAggregator(history_size=100)
        self.received_candles: list[Candle] = []

        async def capture(candle: Candle):
            self.received_candles.append(candle)

        self.aggregator.register_callback(capture)

    @pytest.mark.asyncio
    async def test_tick_buffering(self):
        """Ticks should be buffered without producing a candle."""
        tick = DecodedTick(
            instrument_token="TEST|001",
            ltp=100.0,
            ltq=10,
            ltt=time.time(),
            volume=1000,
        )
        await self.aggregator.on_tick(tick)
        assert "TEST|001" in self.aggregator.active_instruments
        assert len(self.received_candles) == 0  # No candle until minute crosses

    @pytest.mark.asyncio
    async def test_candle_on_minute_cross(self):
        """Candle should be emitted when time crosses to a new minute."""
        now = time.time()
        current_minute = int(now // 60)

        # Send ticks in "previous" minute
        for i in range(5):
            tick = DecodedTick(
                instrument_token="TEST|002",
                ltp=100.0 + i,
                ltq=10,
                ltt=now,
                volume=1000 * (i + 1),
            )
            await self.aggregator.on_tick(tick)

        # Manually set the current minute to trigger candle close
        self.aggregator._current_minute["TEST|002"] = current_minute - 1
        self.aggregator._tick_buffers["TEST|002"] = [
            {"timestamp": now - 30, "price": 100.0, "volume": 1000, "oi": 0, "ltq": 10},
            {"timestamp": now - 20, "price": 102.0, "volume": 2000, "oi": 0, "ltq": 15},
            {"timestamp": now - 10, "price": 99.0, "volume": 3000, "oi": 0, "ltq": 20},
            {"timestamp": now - 5, "price": 101.0, "volume": 4000, "oi": 100, "ltq": 12},
        ]

        # New tick in current minute triggers close of previous
        new_tick = DecodedTick(
            instrument_token="TEST|002",
            ltp=101.5,
            ltq=10,
            ltt=now,
            volume=5000,
        )
        await self.aggregator.on_tick(new_tick)

        # Should have one candle
        assert len(self.received_candles) == 1
        candle = self.received_candles[0]
        assert candle.instrument_token == "TEST|002"
        assert candle.open == 100.0
        assert candle.high == 102.0
        assert candle.low == 99.0
        assert candle.close == 101.0
        assert candle.tick_count == 4

    def test_empty_history(self):
        """Empty instrument should return empty history."""
        history = self.aggregator.get_history("NONEXISTENT")
        assert len(history) == 0

    def test_history_df(self):
        """get_history_df should return a Polars DataFrame."""
        df = self.aggregator.get_history_df("NONEXISTENT")
        assert df.height == 0

    def test_candle_to_dict(self):
        """Candle serialization."""
        candle = Candle(
            instrument_token="TEST|001",
            timestamp=datetime(2026, 8, 1, 10, 0, tzinfo=timezone.utc),
            open=100.0, high=105.0, low=99.0, close=103.0,
            volume=50000, oi=1000, tick_count=150, vwap=101.5,
        )
        d = candle.to_dict()
        assert d["open"] == 100.0
        assert d["high"] == 105.0
        assert d["vwap"] == 101.5
        assert "timestamp" in d
