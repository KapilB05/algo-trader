"""
Unit test suite for Part 1: Paper / Live Mode Safety (tests/test_paper_live_safety.py).
"""

from fastapi.testclient import TestClient
import pytest

from src.dashboard_v2.server import app, global_state

client = TestClient(app)


def test_paper_mode_simulated_execution() -> None:
    # 1. Set mode to PAPER
    res_mode = client.post("/api/mode", json={"mode": "PAPER"})
    assert res_mode.status_code == 200
    assert res_mode.json()["mode"] == "PAPER"

    # Reset killswitch if active
    client.post("/api/killswitch", json={"halt": False, "reason": ""})

    # 2. Dispatch simulated order
    res_order = client.post("/api/order", json={
        "symbol": "BANKNIFTY FUT",
        "side": "BUY",
        "quantity": 15,
        "order_type": "MARKET",
        "product_type": "MIS",
        "price": 52450.0,
    })
    assert res_order.status_code == 200
    data = res_order.json()

    assert data["status"] == "SUCCESS"
    assert data["mode"] == "PAPER"
    assert "order_id" in data
    assert data["order_id"].startswith("ORD_")
    assert "fill_price" in data
    # In BUY order, fill_price includes positive slippage
    assert data["fill_price"] >= 52450.0


def test_killswitch_blocks_orders_in_all_modes() -> None:
    # Activate kill switch
    client.post("/api/killswitch", json={"halt": True, "reason": "Emergency Test Lockout"})

    # Attempt order in PAPER mode
    client.post("/api/mode", json={"mode": "PAPER"})
    res_order = client.post("/api/order", json={
        "symbol": "NIFTY 24400 CE",
        "side": "BUY",
        "quantity": 25,
        "order_type": "MARKET",
        "product_type": "MIS",
        "price": 120.0,
    })
    assert res_order.status_code == 400
    assert "Kill switch is active" in res_order.json()["detail"]

    # Deactivate kill switch
    client.post("/api/killswitch", json={"halt": False, "reason": ""})


def test_mode_switch_validation() -> None:
    # Valid modes
    for valid_m in ["PAPER", "LIVE", "AUTO", "SEMI", "MANUAL"]:
        res = client.post("/api/mode", json={"mode": valid_m})
        assert res.status_code == 200
        assert res.json()["mode"] == valid_m

    # Invalid mode
    res_inv = client.post("/api/mode", json={"mode": "UNKNOWN_DANGEROUS_MODE"})
    assert res_inv.status_code == 400

    # Return to safe default
    client.post("/api/mode", json={"mode": "PAPER"})


def test_live_mode_risk_manager_pretrade_gate() -> None:
    # Set to LIVE mode
    client.post("/api/mode", json={"mode": "LIVE"})
    client.post("/api/killswitch", json={"halt": False, "reason": ""})

    # Order with huge quantity that violates single trade risk (e.g. 5000 lots)
    res = client.post("/api/order", json={
        "symbol": "BANKNIFTY FUT",
        "side": "BUY",
        "quantity": 10000,
        "order_type": "MARKET",
        "product_type": "MIS",
        "price": 52450.0,
    })
    assert res.status_code == 400
    assert "REJECTED" in res.json()["detail"] or "Risk" in res.json()["detail"]

    # Return to safe default
    client.post("/api/mode", json={"mode": "PAPER"})
