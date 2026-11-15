"""
Unit Tests for Black-Scholes Greeks Engine (Layer 1).
"""

import math
import pytest
from src.data.greeks_engine import GreeksEngine, greeks_engine


def test_greeks_call_delta_bounds():
    """Test Call Delta is strictly between 0 and 1."""
    # Deep ITM Call
    g_itm = greeks_engine.calculate_greeks(spot=24500, strike=24000, time_to_expiry_years=0.02, market_price=520, option_type="CE")
    assert 0.50 < g_itm.delta <= 1.0

    # Deep OTM Call
    g_otm = greeks_engine.calculate_greeks(spot=24000, strike=24500, time_to_expiry_years=0.02, market_price=10, option_type="CE")
    assert 0.0 <= g_otm.delta < 0.50


def test_greeks_put_delta_bounds():
    """Test Put Delta is strictly between -1 and 0."""
    # Deep ITM Put
    g_itm = greeks_engine.calculate_greeks(spot=24000, strike=24500, time_to_expiry_years=0.02, market_price=520, option_type="PE")
    assert -1.0 <= g_itm.delta < -0.50

    # Deep OTM Put
    g_otm = greeks_engine.calculate_greeks(spot=24500, strike=24000, time_to_expiry_years=0.02, market_price=10, option_type="PE")
    assert -0.50 <= g_otm.delta <= 0.0


def test_greeks_gamma_positive():
    """Test Gamma is strictly positive for both Calls and Puts."""
    g_call = greeks_engine.calculate_greeks(spot=24000, strike=24000, time_to_expiry_years=0.05, market_price=250, option_type="CE")
    g_put = greeks_engine.calculate_greeks(spot=24000, strike=24000, time_to_expiry_years=0.05, market_price=250, option_type="PE")
    assert g_call.gamma > 0.0
    assert g_put.gamma > 0.0


def test_greeks_theta_negative():
    """Test Theta (daily time decay) is negative for option buyers."""
    g_call = greeks_engine.calculate_greeks(spot=24000, strike=24000, time_to_expiry_years=0.05, market_price=250, option_type="CE")
    assert g_call.theta < 0.0


def test_implied_volatility_solver():
    """Test IV solver converges to reasonable volatility."""
    iv = greeks_engine.calculate_implied_volatility(
        market_price=250.0,
        spot=24000.0,
        strike=24000.0,
        time_to_expiry_years=0.05,
        option_type="CE",
    )
    assert 0.05 <= iv <= 0.80  # Between 5% and 80%
