"""
Black-Scholes Mathematical Engine — Vectorized Pricing & Greeks.

Provides ultra-fast, dependency-free Black-Scholes calculations using pure NumPy.
Used by all downstream arbitrage scanners to compute fair value, detect mispricings,
and build Box Spread / Synthetic Parity arbitrage opportunities.

Key formulas:
  d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
  d2 = d1 - σ*√T
  Call = S*N(d1) - K*e^(-r*T)*N(d2)
  Put  = K*e^(-r*T)*N(-d2) - S*N(-d1)
  Put-Call Parity: C - P = S - K*e^(-r*T)   (must hold always)
"""

from __future__ import annotations

from math import erf, exp, log, pi, sqrt  # type: ignore[attr-defined]
from dataclasses import dataclass
from typing import Tuple

import numpy as np

# Risk-free rate: India 10-year G-Sec yield approximation
RISK_FREE_RATE = 0.07   # 7% p.a.

# Minimum time to expiry (0.5 calendar days in years) to prevent division by zero
MIN_T = 0.5 / 365.0


# ── Scalar CDF for Black-Scholes ──────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (fast C-level call)."""
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


# ── Vectorized CDF / PDF for batch operations ─────────────────────────────────

def norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF."""
    return (1.0 + np.vectorize(erf)(x / sqrt(2.0))) / 2.0


def norm_pdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal PDF."""
    return np.exp(-0.5 * x ** 2) / sqrt(2.0 * pi)


# ── Core Dataclass ────────────────────────────────────────────────────────────

@dataclass
class BSResult:
    """Full Black-Scholes result for a single option."""
    call_price: float    # Theoretical call premium (₹)
    put_price: float     # Theoretical put premium (₹)
    delta_call: float    # Call delta ∈ [0, 1]
    delta_put: float     # Put delta ∈ [-1, 0]
    gamma: float         # Gamma (same for call & put)
    vega: float          # Vega per 1% IV move (₹)
    theta_call: float    # Call theta per calendar day (₹)
    theta_put: float     # Put theta per calendar day (₹)
    iv: float            # Implied volatility (decimal, e.g. 0.14 = 14%)
    d1: float
    d2: float
    # Put-Call Parity residual: C - P - (S - K*e^{-rT})
    # Should be ~0. Large deviations = arbitrage opportunity.
    pcp_residual: float


# ── Main Pricing Engine ───────────────────────────────────────────────────────

def black_scholes(
    spot: float,
    strike: float,
    dte_days: float,
    iv: float,
    r: float = RISK_FREE_RATE,
) -> BSResult:
    """
    Compute full Black-Scholes analytics for a European option.

    Args:
        spot:      Current underlying price (e.g. 56,000 for BANKNIFTY)
        strike:    Option strike price (e.g. 56,000)
        dte_days:  Days to expiry (e.g. 3.5 for 3.5 calendar days)
        iv:        Implied volatility as decimal (e.g. 0.14 for 14%)
        r:         Annualized risk-free rate (default 7%)

    Returns:
        BSResult dataclass with call/put prices, all Greeks, and PCP residual.
    """
    T = max(MIN_T, dte_days / 365.0)
    sigma = max(0.01, iv)
    sqrt_T = sqrt(T)

    d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = _norm_cdf(d1)
    N_d2 = _norm_cdf(d2)
    N_nd1 = _norm_cdf(-d1)
    N_nd2 = _norm_cdf(-d2)
    pdf_d1 = _norm_pdf(d1)

    disc = exp(-r * T)

    call = spot * N_d1 - strike * disc * N_d2
    put  = strike * disc * N_nd2 - spot * N_nd1

    delta_call = N_d1
    delta_put  = N_d1 - 1.0
    gamma  = pdf_d1 / (spot * sigma * sqrt_T)
    vega   = spot * pdf_d1 * sqrt_T / 100.0      # per 1% IV change in ₹
    theta_call = (
        -(spot * pdf_d1 * sigma) / (2.0 * sqrt_T)
        - r * strike * disc * N_d2
    ) / 365.0
    theta_put = (
        -(spot * pdf_d1 * sigma) / (2.0 * sqrt_T)
        + r * strike * disc * N_nd2
    ) / 365.0

    # Put-Call Parity: C - P = S - K*e^(-rT)
    pcp_residual = call - put - (spot - strike * disc)

    return BSResult(
        call_price=round(max(0.0, call), 4),
        put_price=round(max(0.0, put), 4),
        delta_call=round(delta_call, 4),
        delta_put=round(delta_put, 4),
        gamma=round(gamma, 6),
        vega=round(vega, 4),
        theta_call=round(theta_call, 4),
        theta_put=round(theta_put, 4),
        iv=sigma,
        d1=round(d1, 4),
        d2=round(d2, 4),
        pcp_residual=round(pcp_residual, 4),
    )


# ── IV Solver (Newton-Raphson) ────────────────────────────────────────────────

def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    dte_days: float,
    option_type: str = "CE",
    r: float = RISK_FREE_RATE,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Solve for Implied Volatility using Newton-Raphson iteration.

    Args:
        market_price: Observed market premium (₹)
        spot:         Current underlying spot price
        strike:       Option strike
        dte_days:     Days to expiry
        option_type:  'CE' or 'PE'
        r:            Risk-free rate
        max_iter:     Maximum Newton iterations
        tol:          Convergence tolerance

    Returns:
        IV as decimal (e.g. 0.145 = 14.5%). Returns 0.15 as fallback if
        Newton-Raphson fails to converge.
    """
    T = max(MIN_T, dte_days / 365.0)
    if isinstance(option_type, bool):
        is_call = option_type
    else:
        is_call = option_type.upper() in ("CE", "CALL", "TRUE")

    # Initial guess: Brenner-Subrahmanyam approximation
    sigma = sqrt(2.0 * pi / T) * market_price / spot
    sigma = max(0.01, min(5.0, sigma))

    for _ in range(max_iter):
        sqrt_T = sqrt(T)
        d1 = (log(spot / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        disc = exp(-r * T)
        pdf_d1 = _norm_pdf(d1)

        if is_call:
            price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
        else:
            price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)

        vega_raw = spot * pdf_d1 * sqrt_T
        if vega_raw < 1e-8:
            break

        diff = price - market_price
        if abs(diff) < tol:
            break

        sigma -= diff / vega_raw
        sigma = max(0.001, min(5.0, sigma))

    return round(sigma, 6)


# ── Box Spread Fair Value ────────────────────────────────────────────────────

def box_spread_fair_value(strike_low: float, strike_high: float, dte_days: float, r: float = RISK_FREE_RATE) -> float:
    """
    Calculate the risk-free fair value of a Box Spread.

    A Box Spread = Bull Call Spread + Bear Put Spread on the same two strikes.
    At expiry, the payout is ALWAYS exactly (K_high - K_low), regardless of market level.

    Fair value today (PV of guaranteed payout):
        Box_FV = (K_high - K_low) * e^(-r * T)

    If market_cost < Box_FV → GUARANTEED risk-free profit = Box_FV - market_cost.

    Args:
        strike_low:  Lower strike (e.g. 56000)
        strike_high: Higher strike (e.g. 56500)
        dte_days:    Days to expiry
        r:           Risk-free rate

    Returns:
        Present value of the guaranteed payout (₹ per unit).
    """
    T = max(MIN_T, dte_days / 365.0)
    return (strike_high - strike_low) * exp(-r * T)


# ── Box Spread Market Cost ─────────────────────────────────────────────────

def box_spread_market_cost(
    call_low_premium: float,   # Buy call at K_low
    call_high_premium: float,  # Sell call at K_high
    put_low_premium: float,    # Sell put at K_low
    put_high_premium: float,   # Buy put at K_high
) -> float:
    """
    Compute actual market cost of entering a Box Spread (4 legs).

    Box = (Buy C_low - Sell C_high) + (Buy P_high - Sell P_low)
        = Bull Call Spread + Bear Put Spread

    If this < box_spread_fair_value(), there is a RISK-FREE arbitrage.
    """
    bull_call = call_low_premium - call_high_premium    # Cost to buy bull call spread
    bear_put  = put_high_premium - put_low_premium      # Cost to buy bear put spread
    return bull_call + bear_put


# ── Box Spread All-In Cost Model ────────────────────────────────────────────

# Flat brokerage per order leg (matches BacktestEngine.brokerage_per_order).
BOX_BROKERAGE_PER_ORDER = 20.0
# Simplified turnover levy (STT + exchange + stamp + GST), same schedule as the
# backtest engine's stt_and_turnover_pct. Deliberately conservative vs itemized NSE.
BOX_TURNOVER_LEVY_PCT = 0.0003
# Spread-crossing fallback per leg when the chain carries no bid/ask quotes
# (Angel One `getMarketData` returns LTP only): buys lift, sells hit, so each
# leg pays ~0.1% of its LTP. Documented assumption — replace with real half-
# spread the day bid/ask fields become available.
BOX_SLIPPAGE_PCT_OF_LTP = 0.001


@dataclass
class BoxCostBreakdown:
    """All-in frictional cost of entering a 4-leg box (₹, for `lots` lots)."""
    brokerage: float      # 4 legs × ₹20 × lots
    turnover_levy: float  # levy % on total 4-leg premium turnover
    slippage: float       # spread-crossing estimate (LTP-% fallback)
    total: float


def box_all_in_cost(
    leg_ltps: Tuple[float, float, float, float],
    lot_size: int,
    lots: int = 1,
    slippage_pct: float = BOX_SLIPPAGE_PCT_OF_LTP,
) -> BoxCostBreakdown:
    """
    Frictional cost of putting on a box: buy C_low, sell C_high, buy P_high,
    sell P_low (4 order legs).

    Args:
        leg_ltps:   (call_low, call_high, put_low, put_high) LTPs in ₹.
        lot_size:   Exchange lot size (e.g. 30 for BANKNIFTY).
        lots:       Number of lots.
        slippage_pct: Per-leg spread-crossing fallback (no bid/ask on feed).

    Returns:
        BoxCostBreakdown with rupee costs for the full position.
    """
    units = float(lot_size * max(1, lots))
    turnover = sum(max(0.0, float(p)) for p in leg_ltps) * units
    brokerage = 4.0 * BOX_BROKERAGE_PER_ORDER * max(1, lots)
    levy = turnover * BOX_TURNOVER_LEVY_PCT
    slippage = turnover * slippage_pct
    return BoxCostBreakdown(
        brokerage=round(brokerage, 2),
        turnover_levy=round(levy, 2),
        slippage=round(slippage, 2),
        total=round(brokerage + levy + slippage, 2),
    )


def box_net_edge(
    market_cost: float,
    fair_value: float,
    lot_size: int,
    lots: int = 1,
    leg_ltps: Tuple[float, float, float, float] | None = None,
) -> Tuple[float, BoxCostBreakdown]:
    """
    Post-cost box edge in ₹ for the full position.

    gross = (fair_value - market_cost) per unit × units; net = gross − costs.
    A box is only a real arbitrage when net > 0 — the old gross-only gate
    booked theoretical edges that fees consumed (e.g. ₹5.12/unit gross on a
    500-wide box ≈ ₹154/lot vs ≈ ₹145/lot all-in costs: noise, not profit).

    Returns:
        (net_edge_inr, BoxCostBreakdown). `leg_ltps` defaults to a turnover
        estimate from market_cost when leg detail is unavailable.
    """
    units = float(lot_size * max(1, lots))
    gross = (fair_value - market_cost) * units
    if leg_ltps is None:
        # Turnover ≈ 2× box cost is a rough but conservative proxy
        # (4 legs each ~half the box width on average).
        proxy = (abs(market_cost) * 2.0 / 4.0,)
        leg_ltps = (proxy[0], proxy[0], proxy[0], proxy[0])
    costs = box_all_in_cost(leg_ltps, lot_size, lots)
    return round(gross - costs.total, 2), costs


# ── Vectorized Batch Pricer ─────────────────────────────────────────────────

def batch_price_chain(
    spot: float,
    strikes: np.ndarray,
    dte_days: float,
    iv_calls: np.ndarray,
    iv_puts: np.ndarray,
    r: float = RISK_FREE_RATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized batch pricing of an entire option chain.

    Args:
        spot:       Current spot price
        strikes:    Array of strike prices
        dte_days:   Days to expiry
        iv_calls:   Array of call IVs (decimal, e.g. 0.14)
        iv_puts:    Array of put IVs (decimal)
        r:          Risk-free rate

    Returns:
        (call_prices, put_prices) as numpy arrays.
    """
    T = max(MIN_T, dte_days / 365.0)
    sqrt_T = sqrt(T)
    disc = exp(-r * T)

    # Vectorize d1/d2 for calls
    sigma_c = np.maximum(0.01, iv_calls)
    d1_c = (np.log(spot / strikes) + (r + 0.5 * sigma_c ** 2) * T) / (sigma_c * sqrt_T)
    d2_c = d1_c - sigma_c * sqrt_T
    calls = spot * norm_cdf_vec(d1_c) - strikes * disc * norm_cdf_vec(d2_c)

    # Vectorize d1/d2 for puts
    sigma_p = np.maximum(0.01, iv_puts)
    d1_p = (np.log(spot / strikes) + (r + 0.5 * sigma_p ** 2) * T) / (sigma_p * sqrt_T)
    d2_p = d1_p - sigma_p * sqrt_T
    puts = strikes * disc * norm_cdf_vec(-d2_p) - spot * norm_cdf_vec(-d1_p)

    return np.maximum(0.0, calls), np.maximum(0.0, puts)
