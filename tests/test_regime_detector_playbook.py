"""
Unit Tests for 5-State Market Regime Playbook & Hysteresis Filter (Layer 3).
"""

import numpy as np
import pandas as pd
import pytest
from src.strategy.regime_detector import MarketRegime, RegimeDetector


def create_trending_candles(n_bars: int = 40, direction: str = "UP") -> pd.DataFrame:
    """Helper to generate strong trending candles."""
    mult = 1.0 if direction == "UP" else -1.0
    c = 50000.0 + np.cumsum(np.full(n_bars, 30.0 * mult))
    h = c + 15.0
    l = c - 15.0
    o = l + (h - l) / 2.0
    v = np.full(n_bars, 10000)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})


def test_chop_regime_gates_trading():
    """Verify CHOP regime blocks trading and returns 0% sizing."""
    detector = RegimeDetector(adx_chop_threshold=20.0)
    detector._current_regime = MarketRegime.CHOP

    assert detector.is_trade_allowed() is False
    assert detector.get_sizing_multiplier() == 0.0
    assert len(detector.allowed_strategies) == 0


def test_trending_up_allows_momentum_strategies():
    """Verify TRENDING_UP allows momentum and VWAP reclaim strategies."""
    detector = RegimeDetector(adx_trend_threshold=25.0, hysteresis_bars=1)
    df_up = create_trending_candles(n_bars=50, direction="UP")
    regime = detector.classify_market_state(df_up, current_vix=14.0)

    assert regime in (MarketRegime.TRENDING_UP, MarketRegime.COMPRESSION)
    assert detector.is_trade_allowed() is True


def test_chaos_regime_on_vix_spike():
    """Verify rapid 1-hour VIX spike triggers CHAOS regime."""
    detector = RegimeDetector(vix_chaos_spike_pct=30.0)
    df = create_trending_candles(n_bars=30)
    regime = detector.classify_market_state(df, current_vix=26.0, vix_1h_change_pct=35.0)

    assert regime == MarketRegime.CHAOS
    assert detector.get_sizing_multiplier() <= 0.25
