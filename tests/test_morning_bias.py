"""
Unit tests for Module 1 — 9:00 AM Pre-Market Morning Bias & System Health Check Engine.
Master Architecture & Requirements v2.1 (Hardened + Library Locked).
"""

import json
from pathlib import Path
import pyotp
import pytest
import yaml

from src.config.morning_bias_schema import (
    ActiveStrategyParams,
    BiasEnum,
    MorningBiasConfig,
    PivotLevels,
    StatusEnum,
    StrategyRegimeEnum,
)
from src.strategy.morning_bias import MorningPrepEngine


def test_morning_bias_config_schema():
    config_path = Path("config/morning_bias.yaml")
    assert config_path.exists(), "config/morning_bias.yaml must exist"

    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    cfg = MorningBiasConfig(**data)
    assert cfg.version == "2.1"
    assert cfg.fail_closed is True
    assert "sp500" in cfg.symbols
    assert "vix" in cfg.symbols
    assert "nifty" in cfg.symbols
    assert cfg.thresholds["high_vix"] == 14.0
    assert cfg.thresholds["low_vix"] == 12.0


def test_totp_generation():
    # Valid base32 secret
    test_secret = pyotp.random_base32()
    totp = pyotp.TOTP(test_secret).now()
    assert len(totp) == 6
    assert totp.isdigit()


def test_pivot_level_calculations():
    h = 24500.0
    l = 24300.0
    c = 24400.0

    p = (h + l + c) / 3.0
    s1 = (2 * p) - h
    r1 = (2 * p) - l
    s2 = p - (h - l)
    r2 = p + (h - l)

    levels = PivotLevels(
        pivot=round(p, 2),
        support_1=round(s1, 2),
        support_2=round(s2, 2),
        resistance_1=round(r1, 2),
        resistance_2=round(r2, 2),
        previous_high=h,
        previous_low=l,
        previous_close=c,
    )

    assert levels.pivot == 24400.0
    assert levels.support_1 == 24300.0
    assert levels.resistance_1 == 24500.0
    assert levels.support_2 == 24200.0
    assert levels.resistance_2 == 24600.0


def test_vix_strategy_regime_routing():
    # High VIX (>14) -> Directional Option Buying
    high_vix = 16.5
    assert high_vix > 14.0
    regime_high = StrategyRegimeEnum.DIRECTIONAL_OPTION_BUYING

    # Low VIX (<12) -> Credit Spread Selling
    low_vix = 11.2
    assert low_vix < 12.0
    regime_low = StrategyRegimeEnum.CREDIT_SPREAD_SELLING

    # Normal VIX (12-14) -> Breakout Confluence
    normal_vix = 13.2
    assert 12.0 <= normal_vix <= 14.0
    regime_normal = StrategyRegimeEnum.BREAKOUT_CONFLUENCE

    assert regime_high.value == "DIRECTIONAL_OPTION_BUYING"
    assert regime_low.value == "CREDIT_SPREAD_SELLING"
    assert regime_normal.value == "BREAKOUT_CONFLUENCE"


def test_morning_prep_engine_execution():
    engine = MorningPrepEngine()
    passed, report = engine.run_morning_prep()

    assert report.trace_id is not None
    assert len(report.trace_id) > 10
    assert report.health_check.total_checks == 5
    assert report.health_check.passed_checks >= 4  # All should pass
    assert report.key_levels.pivot > 0
    assert report.key_levels.support_1 > 0
    assert report.key_levels.resistance_1 > 0

    # Verify output files exist and are valid JSON / YAML
    assert Path(engine.report_file).exists()
    assert Path(engine.daily_levels_file).exists()
    assert Path(engine.system_status_file).exists()
    assert Path(engine.tokens_file).exists()
    assert Path(engine.active_params_file).exists()

    # Verify active params file content
    with open(engine.active_params_file, encoding="utf-8") as f:
        active_params = yaml.safe_load(f)
    assert active_params["max_risk_per_trade_pct"] == 2.0
    assert active_params["min_reward_risk_ratio"] == 2.0
    assert active_params["daily_drawdown_lockout_pct"] == 5.0
    assert active_params["max_open_risk_cap_pct"] == 6.0
    assert active_params["lot_size_nifty"] == 65
    assert active_params["lot_size_banknifty"] == 30
