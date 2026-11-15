"""
Unit tests for Black-Scholes Option Fair Value and IV-Crush Shield.
"""

from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

from src.live_paper_trader import LivePaperTrader
from src.math.black_scholes import black_scholes, RISK_FREE_RATE

IST = ZoneInfo("Asia/Kolkata")


def test_shield_fair_price_approved():
    """Verify that a fairly priced option passes the shield."""
    trader = LivePaperTrader()
    spot = 56000.0
    strike = 56000.0
    dte = 3.0
    vix = 13.0  # 13%

    # Calculate analytical fair value
    bs = black_scholes(spot, strike, dte, vix / 100.0, RISK_FREE_RATE)
    fair_call = bs.call_price

    # Market price is fair (e.g. within 5% of theoretical)
    market_price = fair_call * 1.05

    is_safe, fv, mkt_iv, ratio, reason = trader._evaluate_option_fair_value_shield(
        spot_price=spot,
        strike=strike,
        dte_days=dte,
        is_call=True,
        market_price=market_price,
        vix_val=vix,
    )

    assert is_safe is True
    assert abs(fv - fair_call) < 1.0
    assert ratio <= 1.30
    assert reason == "FAIR_VALUE_CONFIRMED"


def test_shield_overpriced_premium_rejected():
    """Verify that an overpriced option (> 30% over Black-Scholes) is blocked."""
    trader = LivePaperTrader()
    spot = 56000.0
    strike = 56000.0
    dte = 3.0
    vix = 13.0

    bs = black_scholes(spot, strike, dte, vix / 100.0, RISK_FREE_RATE)
    fair_call = bs.call_price

    # Market price is heavily inflated (e.g. 50% above fair value)
    pumped_price = fair_call * 1.50

    is_safe, fv, mkt_iv, ratio, reason = trader._evaluate_option_fair_value_shield(
        spot_price=spot,
        strike=strike,
        dte_days=dte,
        is_call=True,
        market_price=pumped_price,
        vix_val=vix,
    )

    assert is_safe is False
    assert ratio > 1.30
    assert "Overpriced" in reason or "IV Overheated" in reason


def test_shield_blocks_paper_entry(monkeypatch):
    """Verify that execute_paper_entry refuses to open an order if the option is overpriced."""
    trader = LivePaperTrader()
    assert trader.open_position is None

    # Mock _evaluate_option_fair_value_shield to return unsafe (overpriced)
    monkeypatch.setattr(
        trader,
        "_evaluate_option_fair_value_shield",
        lambda *args, **kwargs: (False, 400.0, 0.35, 1.65, "Overpriced Premium (1.65x Fair Value)"),
    )

    now = datetime(2026, 9, 15, 10, 0, tzinfo=IST)
    trader.execute_paper_entry(
        signal_direction="LONG",
        spot_price=56000.0,
        atr_val=150.0,
        multiplier=1.0,
        strats=["ORB", "SUPERTREND", "FLOW"],
        now=now,
    )

    # Position must NOT be opened
    assert trader.open_position is None
