"""
Test Fixtures — Shared fixtures for all test modules.

Provides mock broker clients, sample tick data, test settings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.audit.audit_logger import AuditLogger
from src.data.candle_aggregator import Candle, CandleAggregator
from src.data.protobuf_decoder import OHLC, DecodedTick


@pytest.fixture
def audit_logger(tmp_path: Path) -> AuditLogger:
    """Create an audit logger writing to a temp directory."""
    return AuditLogger(log_dir=str(tmp_path / "audit"))


@pytest.fixture
def sample_tick() -> DecodedTick:
    """A valid sample tick for testing."""
    return DecodedTick(
        instrument_token="NSE_EQ|INE002A01018",
        ltp=2890.50,
        ltq=10,
        ltt=datetime.now(timezone.utc).timestamp(),
        volume=1500000,
        oi=45200,
        total_buy_qty=52000,
        total_sell_qty=48000,
        ohlc=OHLC(open=2880.0, high=2895.0, low=2875.0, close=2890.5),
    )


@pytest.fixture
def sample_ticks() -> list[DecodedTick]:
    """A sequence of valid ticks for candle aggregation tests."""
    base_time = datetime.now(timezone.utc).timestamp()
    return [
        DecodedTick(
            instrument_token="TEST|001",
            ltp=100.0 + i * 0.5,
            ltq=10 + i,
            ltt=base_time + i,
            volume=1000 * (i + 1),
            oi=5000,
        )
        for i in range(20)
    ]


@pytest.fixture
def candle_aggregator() -> CandleAggregator:
    """Fresh candle aggregator instance."""
    return CandleAggregator(history_size=100)


@pytest.fixture
def sample_candles() -> list[Candle]:
    """A sequence of completed candles for indicator tests."""
    base = datetime(2026, 8, 1, 9, 15, tzinfo=timezone.utc)
    candles = []
    price = 24800.0
    for i in range(50):
        import random
        random.seed(i)
        change = random.uniform(-30, 35)
        o = price
        c = price + change
        h = max(o, c) + random.uniform(0, 15)
        l = min(o, c) - random.uniform(0, 15)
        v = random.randint(50000, 200000)

        candles.append(Candle(
            instrument_token="NSE_INDEX|Nifty 50",
            timestamp=base.replace(minute=15 + i),
            open=o, high=h, low=l, close=c,
            volume=v, oi=0, tick_count=100,
        ))
        price = c

    return candles


@pytest.fixture
def mock_order_api() -> MagicMock:
    """Mock OrderApi."""
    api = MagicMock()
    api.place_order = MagicMock(return_value=MagicMock(
        data=MagicMock(to_dict=lambda: {"order_id": "ORD-001", "status": "PENDING"})
    ))
    api.cancel_order = MagicMock(return_value=MagicMock(
        data=MagicMock(to_dict=lambda: {"order_id": "ORD-001", "status": "CANCELLED"})
    ))
    api.get_order_book = MagicMock(return_value=MagicMock(data=[]))
    return api
