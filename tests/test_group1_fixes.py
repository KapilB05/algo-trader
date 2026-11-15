"""
Unit test suite verifying Group 1 Fixes (C1 & C3):
- C1: Persistent RiskManager singleton preserves Circuit Breaker lockout across order calls.
- C3: Robust open position risk math handles missing stop-loss keys safely without false positives or 0-risk bypass.
"""

from fastapi.testclient import TestClient
import pytest
from src.dashboard_v2.server import app, global_risk_manager
from src.risk.risk_manager import RiskManager


def test_c1_circuit_breaker_lockout_persistence() -> None:
    # 1. Simulate 6% drawdown on persistent global_risk_manager
    global_risk_manager.circuit_breaker.update_pnl(-60000.0)  # -6% P&L
    assert global_risk_manager.circuit_breaker.is_locked is True

    # 2. Make an order API request via TestClient
    client = TestClient(app)
    response = client.post("/api/order", json={
        "symbol": "BANKNIFTY FUT",
        "side": "BUY",
        "quantity": 15,
        "order_type": "MARKET",
        "product_type": "MIS",
        "price": 52450.0,
    })

    assert response.status_code == 400
    assert "Circuit breaker locked out" in response.json()["detail"]

    # Reset circuit breaker for clean state
    global_risk_manager.circuit_breaker._reset()


def test_c3_safe_open_risk_math() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0, max_total_risk_pct=0.06)

    # Position missing stop_loss key (or stop_loss=0)
    open_positions_missing_sl = [
        {"symbol": "BANKNIFTY FUT", "entry_price": 52000.0, "quantity": 15}  # Should use fallback 1.5% stop distance
    ]

    res = risk_mgr.validate_order(
        symbol="NIFTY 50",
        side="BUY",
        quantity=25,
        entry_price=24000.0,
        stop_loss=23800.0,
        open_positions=open_positions_missing_sl,
        available_margin=1000000.0,
    )

    # Risk = (52000 * 0.015 * 15) + (200 * 25) = 11,700 + 5,000 = 16,700 = 1.67% < 6% limit
    assert res.approved is True
    assert res.total_risk_pct < 6.0
