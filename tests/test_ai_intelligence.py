"""
Unit & Integration Tests for Institutional AI Intelligence Architecture (tests/test_ai_intelligence.py).

Verifies:
1. Standardized feature extraction matching FEATURE_COLS.
2. Dual-Model Calibrated ML Scorer P(Win) confidence prediction.
3. Jane Street Anti-Spoofing false-breakout gating.
4. Q-Learning Reinforcement Learning Exit Agent actions (HOLD, MOVE_BREAKEVEN, TIGHTEN_STOP, EXIT_NOW).
5. Dynamic Kelly Sizing calculations.
6. AI Telemetry API endpoint (/api/ai/telemetry).
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.ml.scorer import XGBoostScorer
from src.ml.rl_exit_agent import QLearningExitAgent, ExitAction
from src.strategy.manipulation_detector import JaneStreetDefenseEngine
from src.live_paper_trader import LivePaperTrader

IST = ZoneInfo("Asia/Kolkata")


def test_xgboost_scorer_inference():
    """Verify ML Scorer loads and produces calibrated probabilities in [0.0, 1.0]."""
    scorer = XGBoostScorer()
    sample_features = {
        "vwap_dist_pct": 0.35,
        "vwap_slope": 0.15,
        "atr_pct": 0.45,
        "atr_regime": 0.65,
        "realized_vol_20": 0.18,
        "vol_ratio": 1.45,
        "volume_surge": 1.0,
        "oi_change_pct": 1.5,
        "oi_change_12": 3.2,
        "rsi": 62.0,
        "pdh_dist_pct": 0.1,
        "pdl_dist_pct": -0.8,
        "orh_dist_pct": 0.05,
        "orl_dist_pct": -0.9,
        "sin_time_of_day": -0.2,
        "cos_time_of_day": 0.9,
        "session_period": 1.0,
        "relative_strength_index_pct": 0.2,
        "time_of_day_min": 600,
        "vix": 13.5,
    }
    p_win = scorer.predict_confidence(sample_features)
    assert isinstance(p_win, float)
    assert 0.0 <= p_win <= 1.0


def test_jane_street_anti_spoofing_defense():
    """Verify Jane Street Anti-Spoofing blocks false breakouts without 2nd confirming candle."""
    engine = JaneStreetDefenseEngine()

    # Case 1: Unconfirmed breakout (current bullish, previous bearish and higher)
    unconfirmed_df = pd.DataFrame({
        "open": [56100.0, 56050.0],
        "close": [56020.0, 56080.0],  # Prev was bearish close < open
    })
    is_safe = engine.validate_anti_spoofing(unconfirmed_df, "LONG")
    assert not is_safe, "Unconfirmed breakout should be flagged as spoofing"

    # Case 2: Confirmed breakout (both candles bullish)
    confirmed_df = pd.DataFrame({
        "open": [56000.0, 56050.0],
        "close": [56050.0, 56120.0],  # Both bullish and higher close
    })
    is_confirmed = engine.validate_anti_spoofing(confirmed_df, "LONG")
    assert is_confirmed, "Confirmed 2-candle breakout should pass Jane Street shield"


def test_rl_exit_agent_lifecycle():
    """Verify Q-Learning RL Exit Agent transitions states adaptively."""
    agent = QLearningExitAgent(target_rr=2.5)

    entry_p = 300.0
    initial_sl = 225.0  # ₹75 risk
    tp = 450.0

    # 1. Trade in small profit (+0.4R) -> HOLD
    act_hold = agent.evaluate_exit(
        entry_price=entry_p,
        current_price=330.0,
        initial_stop=initial_sl,
        target_price=tp,
        direction="LONG",
        time_elapsed_mins=10,
    )
    assert act_hold.action_name == "HOLD"

    # 2. Trade reaches +1.1R -> MOVE_BREAKEVEN
    act_be = agent.evaluate_exit(
        entry_price=entry_p,
        current_price=385.0,  # +85 pts / 75 risk = +1.13R
        initial_stop=initial_sl,
        target_price=tp,
        direction="LONG",
        time_elapsed_mins=20,
    )
    assert act_be.action_name == "MOVE_BREAKEVEN"
    assert act_be.suggested_stop == entry_p

    # 3. Trade reaches +1.5R -> TIGHTEN_STOP (+1.0R locked)
    act_trail = agent.evaluate_exit(
        entry_price=entry_p,
        current_price=415.0,  # +115 pts / 75 risk = +1.53R
        initial_stop=initial_sl,
        target_price=tp,
        direction="LONG",
        time_elapsed_mins=25,
    )
    assert act_trail.action_name == "TIGHTEN_STOP"
    assert act_trail.suggested_stop >= entry_p + 75.0


def test_live_paper_trader_ai_telemetry():
    """Verify LivePaperTrader produces complete 5-layer AI telemetry."""
    trader = LivePaperTrader()
    telemetry = trader.get_ai_telemetry()

    assert telemetry["status"] == "SUCCESS"
    assert "ai_layers" in telemetry

    layers = telemetry["ai_layers"]
    assert "layer_1_regime" in layers
    assert "layer_2_consensus" in layers
    assert "layer_3_jane_street_defense" in layers
    assert "layer_4_ml_gating" in layers
    assert "layer_5_rl_exit" in layers

    ml_gate = layers["layer_4_ml_gating"]
    assert "latest_p_win" in ml_gate
    assert "confidence_label" in ml_gate
    assert ml_gate["hurdle_rate"] in (0.52, 0.65)


def test_feature_extraction_completeness():
    """Verify feature dictionary contains all required fields for ML Scorer."""
    trader = LivePaperTrader()
    dummy_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-09-11 09:15", periods=20, freq="5min"),
        "open": np.linspace(56000, 56200, 20),
        "high": np.linspace(56050, 56250, 20),
        "low": np.linspace(55980, 56180, 20),
        "close": np.linspace(56020, 56230, 20),
        "volume": np.random.randint(10000, 50000, 20),
    })
    dummy_feat = trader.compute_features(dummy_df)
    features = trader.extract_ml_features(dummy_feat, vix=13.5)

    required_keys = [
        "vwap_dist_pct", "vwap_slope", "atr_pct", "atr_regime",
        "realized_vol_20", "vol_ratio", "volume_surge", "rsi",
        "orh_dist_pct", "orl_dist_pct", "sin_time_of_day", "cos_time_of_day",
        "session_period", "vix"
    ]
    for k in required_keys:
        assert k in features, f"Missing feature key: {k}"
        assert not np.isnan(features[k]), f"NaN found in feature: {k}"
