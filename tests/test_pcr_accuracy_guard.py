"""
Unit tests for Put-Call Ratio (PCR) Institutional Guard & Accuracy Booster
(tests/test_pcr_accuracy_guard.py).
"""

from typing import cast
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.live_paper_trader import LivePaperTrader


@pytest.fixture
def mock_trader() -> LivePaperTrader:
    """Construct LivePaperTrader instance with ML model bypassed to focus on PCR guard."""
    trader = LivePaperTrader()
    # Mock ML scorer to always approve with 75% confidence
    trader.ml_scorer = MagicMock()
    trader.ml_scorer.is_available = True
    trader.ml_scorer.predict_confidence.return_value = 0.75
    trader.min_ml_confidence = 0.65
    return trader


@pytest.fixture
def bullish_setup_df() -> pd.DataFrame:
    """Synthesize 100 5m bars producing a valid bullish 3/3 consensus (ORB + Supertrend + Flow)."""
    dates = pd.date_range("2026-09-30 09:15", periods=30, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.linspace(56000, 56500, 30),
        "high": np.linspace(56050, 56550, 30),
        "low": np.linspace(55980, 56480, 30),
        "close": np.linspace(56020, 56520, 30),
        "volume": [5000.0] * 29 + [15000.0],  # Volume surge on latest bar
        "vol_ma20": [4000.0] * 30,
        "orh": [56100.0] * 30,
        "orl": [55950.0] * 30,
        "supertrend_dir": [1.0] * 30,
        "ema9": np.linspace(56010, 56510, 30),
        "ema21": np.linspace(55990, 56450, 30),
        "vwap": np.linspace(56000, 56400, 30),
        "atr14": [80.0] * 30,
        "adx14": [28.0] * 30,
        "bbw_rank": [0.6] * 30,
    })
    return df


@pytest.fixture
def bearish_setup_df() -> pd.DataFrame:
    """Synthesize 100 5m bars producing a valid bearish 3/3 consensus."""
    dates = pd.date_range("2026-09-30 09:15", periods=30, freq="5min")
    df = pd.DataFrame({
        "timestamp": dates,
        "open": np.linspace(56500, 56000, 30),
        "high": np.linspace(56550, 56050, 30),
        "low": np.linspace(56480, 55980, 30),
        "close": np.linspace(56450, 55920, 30),
        "volume": [5000.0] * 29 + [15000.0],
        "vol_ma20": [4000.0] * 30,
        "orh": [56600.0] * 30,
        "orl": [56400.0] * 30,
        "supertrend_dir": [-1.0] * 30,
        "ema9": np.linspace(56450, 55950, 30),
        "ema21": np.linspace(56480, 56050, 30),
        "vwap": np.linspace(56500, 56100, 30),
        "atr14": [80.0] * 30,
        "adx14": [28.0] * 30,
        "bbw_rank": [0.6] * 30,
    })
    return df


def test_pcr_defaults_and_properties(mock_trader: LivePaperTrader) -> None:
    """Verify default PCR attributes and public properties exist."""
    assert mock_trader.current_pcr == 1.0
    assert mock_trader.current_pcr_signal == "NEUTRAL"


def test_pcr_ml_feature_extraction(
    mock_trader: LivePaperTrader, bullish_setup_df: pd.DataFrame
) -> None:
    """Verify extract_ml_features returns PCR metrics."""
    feats = mock_trader.extract_ml_features(bullish_setup_df, vix=13.5, pcr=1.35)
    assert "pcr_value" in feats
    assert feats["pcr_value"] == 1.35
    assert feats["pcr_regime_bullish"] == 1.0
    assert feats["pcr_regime_bearish"] == 0.0

    feats_bear = mock_trader.extract_ml_features(bullish_setup_df, vix=13.5, pcr=0.65)
    assert feats_bear["pcr_value"] == 0.65
    assert feats_bear["pcr_regime_bullish"] == 0.0
    assert feats_bear["pcr_regime_bearish"] == 1.0


def test_pcr_guard_blocks_long_into_heavy_call_resistance(
    mock_trader: LivePaperTrader, bullish_setup_df: pd.DataFrame
) -> None:
    """PCR < 0.80 indicates heavy institutional Call writing -> LONG must be rejected."""
    direction, mult, strats = mock_trader.evaluate_signals(
        bullish_setup_df,
        regime_name="trending_up",
        vix_val=13.0,
        pcr_val=0.72,  # Heavy Call Writing resistance overhead
    )
    assert direction == "NONE", "Long trade must be blocked when PCR < 0.80"
    assert mult == 0.0
    assert strats == []


def test_pcr_guard_blocks_short_into_heavy_put_support(
    mock_trader: LivePaperTrader, bearish_setup_df: pd.DataFrame
) -> None:
    """PCR > 1.25 indicates heavy institutional Put writing floor -> SHORT must be rejected."""
    direction, mult, strats = mock_trader.evaluate_signals(
        bearish_setup_df,
        regime_name="trending_down",
        vix_val=13.0,
        pcr_val=1.38,  # Heavy Put Writing floor below
    )
    assert direction == "NONE", "Short trade must be blocked when PCR > 1.25"
    assert mult == 0.0
    assert strats == []


def test_pcr_tailwinds_boosts_sizing_multiplier(
    mock_trader: LivePaperTrader, bullish_setup_df: pd.DataFrame
) -> None:
    """PCR >= 1.20 provides strong institutional Put support tailwind -> scales sizing by 1.15x."""
    mock_trader.min_ml_confidence = 0.50
    cast(MagicMock, mock_trader.ml_scorer).predict_confidence.return_value = 0.55

    # Test neutral PCR (1.0)
    dir_neutral, mult_neutral, _ = mock_trader.evaluate_signals(
        bullish_setup_df, regime_name="trending_up", vix_val=13.0, pcr_val=1.00
    )
    assert dir_neutral == "LONG"

    # Test high PCR (1.25)
    dir_boost, mult_boost, _ = mock_trader.evaluate_signals(
        bullish_setup_df, regime_name="trending_up", vix_val=13.0, pcr_val=1.25
    )
    assert dir_boost == "LONG"
    assert mult_boost > mult_neutral, "High PCR should scale up the position size"


def test_ai_telemetry_includes_pcr_guard_layer(mock_trader: LivePaperTrader) -> None:
    """Telemetry dictionary must expose Layer 6 PCR Guard."""
    mock_trader._cached_pcr = 1.24
    mock_trader._cached_pcr_signal = "BULLISH"
    mock_trader._cached_pcr_15m_delta = 0.05

    telemetry = mock_trader.get_ai_telemetry()
    assert telemetry["status"] == "SUCCESS"
    layers = telemetry["ai_layers"]
    assert "layer_6_pcr_guard" in layers

    pcr_layer = layers["layer_6_pcr_guard"]
    assert pcr_layer["current_pcr"] == 1.24
    assert pcr_layer["pcr_signal"] == "BULLISH"
    assert pcr_layer["pcr_15m_delta"] == 0.05
    assert "long_hurdle" in pcr_layer
    assert "short_hurdle" in pcr_layer
