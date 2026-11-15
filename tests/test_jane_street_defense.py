"""
Unit Tests for Jane Street Anti-Manipulation Defense Shield (Layer 5).
"""

from datetime import datetime, time
import pandas as pd
import pytest
from src.strategy.manipulation_detector import JaneStreetDefenseEngine, jane_street_defense


def test_expiry_cutoff_at_1430():
    """Verify mandatory 14:30 IST position cutoff on expiry days."""
    engine = JaneStreetDefenseEngine()

    # Pre-cutoff (14:15 IST)
    dt_pre = datetime(2026, 8, 28, 14, 15)
    assert engine.is_expiry_cutoff_active(is_expiry_day=True, current_time=dt_pre) is False

    # Post-cutoff (14:35 IST)
    dt_post = datetime(2026, 8, 28, 14, 35)
    assert engine.is_expiry_cutoff_active(is_expiry_day=True, current_time=dt_post) is True

    # Non-expiry day is not affected
    assert engine.is_expiry_cutoff_active(is_expiry_day=False, current_time=dt_post) is False


def test_anti_spoofing_2nd_candle_validation():
    """Verify anti-spoofing requires confirming 2nd candle."""
    engine = JaneStreetDefenseEngine()

    # Two confirming bullish candles
    df_valid = pd.DataFrame({
        "open": [50000.0, 50050.0],
        "close": [50050.0, 50120.0],
    })
    assert engine.validate_anti_spoofing(df_valid, direction="LONG") is True

    # False breakout / rejection candle (bearish 2nd candle)
    df_spoof = pd.DataFrame({
        "open": [50000.0, 50120.0],
        "close": [50100.0, 50020.0],
    })
    assert engine.validate_anti_spoofing(df_spoof, direction="LONG") is False


def test_max_oi_trap_target_trimming():
    """Verify target price is trimmed 0.5% before max OI strike pin."""
    engine = JaneStreetDefenseEngine()

    # Long target (58200) exceeds max call strike (58000)
    adjusted_tp = engine.adjust_for_oi_trap(
        entry_price=57500.0,
        target_price=58200.0,
        direction="LONG",
        max_call_strike=58000.0,
        max_put_strike=57000.0,
    )
    # Trimmed to 58000 * 0.995 = 57710.0
    assert adjusted_tp == 57710.0
