"""
Unit tests for Research-Backed Option Engines (tests/test_research_strategies.py).
Validates:
1. Wu et al. (2022) Kelly Position Sizer with Fractional Quarter-Kelly & 2% Hard Cap.
2. Mohanti & Priyan (2014) Liquidity & Bid-Ask Spread Guard.
3. Bangur (2019, 2020) Defined-Risk Spread Engine (Bull Call 2.75x, Bear Put, Angled Iron Condor).
4. FastAPI Endpoints (/api/research/spreads & /api/research/execute-spread).
"""

import pytest
from fastapi.testclient import TestClient

from src.risk.kelly_sizing import KellyPositionSizer
from src.risk.liquidity_filter import LiquidityFilter
from src.strategy.spread_engine import ResearchSpreadEngine
from src.dashboard_v2.server import app

client = TestClient(app)


# ── 1. Kelly Position Sizer Tests (Wu et al. 2022) ───────────────────────────

def test_kelly_sizing_banknifty_lot_size():
    sizer = KellyPositionSizer(default_capital=1000000.0, fractional_kelly_mult=0.25, max_trade_risk_pct=0.02)
    res = sizer.compute_size(
        capital=1000000.0,
        win_rate=0.768,
        reward_risk_ratio=2.0,
        option_premium=300.0,
        stop_loss_pct=0.25,
        symbol="BANKNIFTY",
    )
    assert res.lot_size == 30
    assert res.clamped_risk_pct == 2.0  # 2% hard ceiling
    assert res.risk_rupees == 20000.0
    assert res.recommended_lots >= 1
    assert res.recommended_qty == res.recommended_lots * 30
    assert res.expected_growth_rate > 0


def test_kelly_sizing_nifty_lot_size():
    sizer = KellyPositionSizer(default_capital=1000000.0, fractional_kelly_mult=0.25, max_trade_risk_pct=0.02)
    res = sizer.compute_size(
        capital=1000000.0,
        win_rate=0.768,
        reward_risk_ratio=2.0,
        option_premium=150.0,
        stop_loss_pct=0.25,
        symbol="NIFTY",
    )
    assert res.lot_size == 65
    assert res.clamped_risk_pct == 2.0
    assert res.risk_rupees == 20000.0
    assert res.recommended_lots >= 1
    assert res.recommended_qty == res.recommended_lots * 65


def test_kelly_zero_or_negative_edge():
    sizer = KellyPositionSizer(default_capital=500000.0)
    # Win rate 20% with 1:1 R:R has negative edge
    res = sizer.compute_size(capital=500000.0, win_rate=0.20, reward_risk_ratio=1.0, option_premium=200.0)
    assert res.full_kelly_fraction == 0.0
    assert res.clamped_risk_pct == 0.0
    assert res.recommended_lots == 1  # Base lot minimum safe allocation


# ── 2. Liquidity & Spread Guard Tests (Mohanti & Priyan 2014) ─────────────────

def test_liquidity_filter_approval():
    filter_engine = LiquidityFilter()
    res = filter_engine.validate_option(
        symbol="NIFTY",
        strike=25400,
        option_type="CE",
        ltp=150.0,
        open_interest=150000,
        volume=12000,
        bid=149.5,
        ask=150.5,
    )
    assert res.approved is True
    assert res.rejection_code is None
    assert res.spread_points == 1.0


def test_liquidity_filter_zero_price_reject():
    filter_engine = LiquidityFilter()
    res = filter_engine.validate_option(
        symbol="BANKNIFTY",
        strike=60000,
        option_type="CE",
        ltp=0.30,
        open_interest=5000,
        volume=10,
    )
    assert res.approved is False
    assert res.rejection_code == "ZERO_PRICE_REJECT"


def test_liquidity_filter_low_oi_reject():
    filter_engine = LiquidityFilter(min_oi_nifty=100000)
    res = filter_engine.validate_option(
        symbol="NIFTY",
        strike=25900,
        option_type="CE",
        ltp=45.0,
        open_interest=25000,  # Below 100k
        volume=6000,
    )
    assert res.approved is False
    assert res.rejection_code == "LOW_OI_REJECT"


def test_liquidity_filter_wide_spread_reject():
    filter_engine = LiquidityFilter(max_spread_pct=1.50)
    res = filter_engine.validate_option(
        symbol="BANKNIFTY",
        strike=56000,
        option_type="CE",
        ltp=100.0,
        open_interest=80000,
        volume=10000,
        bid=95.0,
        ask=102.0,  # 7 pt spread = 7% > 1.5%
    )
    assert res.approved is False
    assert res.rejection_code == "WIDE_SPREAD_REJECT"


# ── 3. Spread Strategy Engine Tests (Bangur 2019/2020) ────────────────────────

def test_bull_call_spread_nifty():
    engine = ResearchSpreadEngine()
    spread = engine.build_bull_call_spread(underlying="NIFTY", spot_price=25420.0, atm_call_ltp=200.0, lots=1)
    
    assert spread.direction == "BULLISH"
    assert spread.edge_multiplier == 2.75
    assert spread.theta_reduction_pct == 60.0
    assert len(spread.legs) == 2
    assert spread.legs[0].action == "BUY"
    assert spread.legs[0].strike == 25400.0
    assert spread.legs[0].quantity == 65
    assert spread.legs[1].action == "SELL"
    assert spread.legs[1].strike == 25450.0  # 50-pt step for Nifty
    assert spread.legs[1].quantity == 65
    assert spread.net_premium > 0
    assert spread.max_loss_rupees == round(spread.net_premium * 65, 2)
    assert spread.max_profit_rupees > 0
    assert spread.breakeven_price == 25400.0 + spread.net_premium


def test_bear_put_spread_banknifty():
    engine = ResearchSpreadEngine()
    spread = engine.build_bear_put_spread(underlying="BANKNIFTY", spot_price=56295.0, atm_put_ltp=650.0, lots=2)
    
    assert spread.direction == "BEARISH"
    assert len(spread.legs) == 2
    assert spread.legs[0].action == "BUY"
    assert spread.legs[0].strike == 56300.0
    assert spread.legs[0].quantity == 60  # 2 lots * 30
    assert spread.legs[1].action == "SELL"
    assert spread.legs[1].strike == 56200.0  # 100-pt step for Bank Nifty
    assert spread.legs[1].quantity == 60
    assert spread.net_premium > 0
    assert spread.max_loss_rupees == round(spread.net_premium * 60, 2)


def test_angled_iron_condor():
    engine = ResearchSpreadEngine()
    condor = engine.build_angled_iron_condor(underlying="NIFTY", spot_price=25400.0, atm_call_ltp=150.0, atm_put_ltp=140.0, lots=1)
    
    assert condor.direction == "NEUTRAL"
    assert condor.premium_type == "NET_CREDIT"
    assert condor.theta_reduction_pct == 100.0
    assert len(condor.legs) == 4
    assert sum(1 for l in condor.legs if l.action == "SELL") == 2
    assert sum(1 for l in condor.legs if l.action == "BUY") == 2
    assert condor.max_loss_rupees > 0
    assert condor.max_profit_rupees > 0


# ── 4. API Endpoints Integration Tests ────────────────────────────────────────

def test_api_research_spreads_endpoint():
    res = client.get("/api/research/spreads?symbol=NIFTY&lots=1")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["underlying"] == "NIFTY"
    assert "kelly_sizing" in data
    assert data["kelly_sizing"]["recommended_lots"] >= 1
    assert len(data["spreads"]) == 3
    assert data["spreads"][0]["strategy_name"] == "Bull Call Spread (Bangur 2.75x)"


def test_api_execute_research_spread():
    res = client.post("/api/research/execute-spread", json={
        "strategy_type": "BULL_CALL",
        "symbol": "BANKNIFTY",
        "lots": 1,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["strategy_name"] == "Bull Call Spread (Bangur 2.75x)"
    assert len(data["executed_legs"]) == 2
    assert data["exec_mode"] in ("PAPER", "LIVE")
