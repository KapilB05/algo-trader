"""
Comprehensive Unit Test Suite for Master Prompt Backend Hardening + Features
(tests/test_master_prompt_backend.py)
"""

import pytest
from fastapi.testclient import TestClient
from src.dashboard_v2.server import app
from src.execution.order_state_machine import OrderTracker, TrackedOrder
from src.execution.trailing_stop import TrailingStopManager
from src.risk.risk_manager import RiskManager
from src.wfo.wfo_engine import WFOEngine, WFOParameters

client = TestClient(app)


# ── 1. Risk Manager Tests ───────────────────────────────────────────────────

def test_risk_manager_trade_risk_limit() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0, max_trade_risk_pct=0.02)
    
    # Valid order (1.5% risk)
    res_ok = risk_mgr.validate_order(
        symbol="BANKNIFTY FUT", side="BUY", quantity=15,
        entry_price=52000.0, stop_loss=51000.0,  # 1000 * 15 = 15000 = 1.5%
        open_positions=[], available_margin=1000000.0,
    )
    assert res_ok.approved is True

    # Violating order (3.0% risk > 2.0% limit)
    res_fail = risk_mgr.validate_order(
        symbol="BANKNIFTY FUT", side="BUY", quantity=30,
        entry_price=52000.0, stop_loss=51000.0,  # 1000 * 30 = 30000 = 3.0%
        open_positions=[], available_margin=1000000.0,
    )
    assert res_fail.approved is False
    assert "Single trade risk" in res_fail.reason


def test_risk_manager_total_open_risk_limit() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0, max_total_risk_pct=0.06)
    
    existing_positions = [
        {"entry_price": 52000.0, "stop_loss": 50000.0, "quantity": 25}  # 2000 * 25 = 50,000 = 5.0%
    ]

    # New order (2.0% risk) -> Total = 7.0% > 6.0% limit
    res_fail = risk_mgr.validate_order(
        symbol="NIFTY 50", side="BUY", quantity=50,
        entry_price=24000.0, stop_loss=23600.0,  # 400 * 50 = 20,000 = 2.0%
        open_positions=existing_positions, available_margin=1000000.0,
    )
    assert res_fail.approved is False
    assert "Total open risk" in res_fail.reason


def test_risk_manager_circuit_breaker_lockout() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0, daily_drawdown_limit_pct=0.05)
    
    # Update P&L with 6% drawdown
    res = risk_mgr.validate_order(
        symbol="NIFTY 50", side="BUY", quantity=25,
        entry_price=24000.0, stop_loss=23800.0,
        open_positions=[], available_margin=1000000.0,
        current_daily_pnl=-60000.0,  # -6.0% P&L
    )
    assert res.approved is False
    assert "Circuit breaker locked out" in res.reason


def test_risk_manager_option_margin_check() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0, option_margin_per_lot=120000.0)

    # Option sell with insufficient margin (₹50,000 available < ₹1,20,000 required)
    res = risk_mgr.validate_order(
        symbol="BANKNIFTY 52500 CE", side="SELL", quantity=15,
        entry_price=350.0, stop_loss=400.0,
        open_positions=[], available_margin=50000.0,
    )
    assert res.approved is False
    assert "Margin insufficient for Option Selling" in res.reason


# ── 2. Order State Machine & Persistence Tests ─────────────────────────────

def test_order_tracker_sqlite_persistence() -> None:
    from src.audit.audit_logger import AuditLogger
    audit = AuditLogger()
    tracker = OrderTracker(audit)

    order = tracker.create_order("NIFTY 50", "BUY", 25, 24366.0, "MARKET")
    tracker.transition(order.client_order_id, "submit")
    tracker.transition(order.client_order_id, "acknowledge", broker_order_id="ANGEL_12345")
    tracker.transition(order.client_order_id, "fill", filled_qty=25, avg_price=24366.0)

    assert order.state == "filled"
    assert order.broker_order_id == "ANGEL_12345"

    # Recover tracker state from SQLite
    tracker2 = OrderTracker(audit)
    recovered = tracker2.get_by_client_id(order.client_order_id)
    assert recovered is not None
    assert recovered.state == "filled"
    assert recovered.broker_order_id == "ANGEL_12345"


# ── 3. Trailing Stop-Loss Tests ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_trailing_stop_percentage_mode() -> None:
    ts = TrailingStopManager()
    state = ts.attach(
        instrument_token="BANKNIFTY FUT", direction="LONG",
        entry_price=50000.0, initial_stop=49000.0,
        trail_mode="percentage", trail_value=2.0,  # 2% trail
    )

    # Price moves up to 52000 -> New Stop = 52000 * 0.98 = 50960
    await ts.on_price_update("BANKNIFTY FUT", 52000.0)
    assert state.current_stop == 50960.0

    # Price drops to 50900 -> Stop triggered
    res = await ts.on_price_update("BANKNIFTY FUT", 50900.0)
    assert res == "TRIGGERED"
    assert state.triggered is True


# ── 4. API Endpoints & Mode Safety Tests ────────────────────────────────────

def test_order_endpoint_risk_and_paper_mode() -> None:
    # 1. Successful paper order
    res_ok = client.post("/api/order", json={
        "symbol": "BANKNIFTY FUT", "side": "BUY", "quantity": 15,
        "order_type": "MARKET", "product_type": "MIS", "price": 52450.0,
    })
    assert res_ok.status_code == 200
    data = res_ok.json()
    assert data["status"] == "SUCCESS"
    assert data["mode"] == "PAPER"
    assert "trace_id" in data


def test_option_chain_endpoint() -> None:
    res = client.get("/api/option-chain?symbol=NIFTY")
    assert res.status_code == 200
    data = res.json()
    assert "spot_price" in data
    assert "pcr" in data
    assert "max_pain" in data
    assert "strikes" in data


def test_wfo_pipeline_and_reload_endpoints() -> None:
    res_run = client.post("/api/wfo/run")
    assert res_run.status_code == 200
    report = res_run.json()
    assert report["status"] in ("SUCCESS", "NO_PASSING_WINDOW")
    if report["status"] == "SUCCESS":
        assert "calibrated_parameters" in report

    res_reload = client.post("/api/wfo/reload")
    assert res_reload.status_code == 200
    data = res_reload.json()
    assert data["status"] == "SUCCESS"
    assert "version" in data
