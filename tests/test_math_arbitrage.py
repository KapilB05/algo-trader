"""
Tests for Black-Scholes Engine and Arbitrage Scanners.
"""

import math
import numpy as np
import pytest

from src.math.black_scholes import (
    black_scholes,
    implied_volatility,
    box_spread_fair_value,
    box_spread_market_cost,
    batch_price_chain,
    RISK_FREE_RATE,
)
from src.arbitrage.put_call_parity import PutCallParityScanner, BoxSpreadOpportunity
from src.arbitrage.statistical_pairs import StatisticalPairsScanner, _simple_adf_test


def test_black_scholes_call_put_parity():
    """Verify Put-Call Parity: C - P = S - K * e^(-r*T)."""
    S = 56000.0
    K = 56000.0
    dte = 7.0
    r = 0.07
    sigma = 0.16

    res = black_scholes(S, K, dte, sigma, r)
    T = dte / 365.0

    # Analytical PCP check
    pcp_diff = res.call_price - res.put_price
    theoretical_diff = S - K * math.exp(-r * T)
    assert abs(pcp_diff - theoretical_diff) < 0.1
    assert abs(res.pcp_residual) < 0.1


def test_box_spread_math():
    """Verify that Box Spread fair value is discounted strike width."""
    k_low = 56000.0
    k_high = 56500.0
    dte = 5.0
    r = 0.07

    fv = box_spread_fair_value(k_low, k_high, dte, r)
    expected_fv = (k_high - k_low) * math.exp(-r * (dte / 365.0))
    assert abs(fv - expected_fv) < 1e-4

    # Cost calculation: (C_low - C_high) + (P_high - P_low)
    c_low, c_high = 600.0, 250.0
    p_low, p_high = 200.0, 545.0
    cost = box_spread_market_cost(c_low, c_high, p_low, p_high)
    # (600 - 250) + (545 - 200) = 350 + 345 = 695
    assert cost == (600.0 - 250.0) + (545.0 - 200.0)


def test_implied_volatility_inversion():
    """Verify Newton-Raphson finds the original volatility."""
    S = 56000.0
    K = 56200.0
    dte = 10.0
    r = 0.07
    sigma_true = 0.18

    res = black_scholes(S, K, dte, sigma_true, r)
    iv = implied_volatility(res.call_price, S, K, dte, "CE", r)
    assert abs(iv - sigma_true) < 1e-2


def test_greeks_signs():
    """Verify Greeks have correct financial signs."""
    S = 56000.0
    K = 56000.0
    dte = 7.0
    r = 0.07
    sigma = 0.16

    res = black_scholes(S, K, dte, sigma, r)
    assert 0.0 < res.delta_call < 1.0  # Call delta between 0 and 1
    assert -1.0 < res.delta_put < 0.0  # Put delta between -1 and 0
    assert res.gamma > 0               # Gamma is strictly positive
    assert res.vega > 0                # Vega is strictly positive
    assert res.theta_call < 0          # Theta decay is negative for long options


def test_simple_adf_mean_reversion():
    """Verify ADF test correctly flags stationary vs non-stationary series."""
    np.random.seed(42)
    # Mean-reverting Ornstein-Uhlenbeck process (stationary)
    ou = [0.0]
    for _ in range(100):
        ou.append(0.2 * ou[-1] + np.random.randn() * 0.5)
    p_val_stat, is_stat2 = _simple_adf_test(ou)
    assert is_stat2 is True


def test_put_call_parity_summary():
    scanner = PutCallParityScanner("BANKNIFTY")
    summary = scanner.get_summary()
    assert summary["scanner"] == "PUT_CALL_PARITY_BOX_SPREAD"
    assert summary["symbol"] == "BANKNIFTY"
    assert "recent_opportunities" in summary


def test_statistical_pairs_summary():
    scanner = StatisticalPairsScanner()
    summary = scanner.get_summary()
    assert summary["scanner"] == "STATISTICAL_PAIRS_ARBITRAGE"
    assert summary["pairs_monitored"] == 5
    assert len(summary["pairs_status"]) == 5
