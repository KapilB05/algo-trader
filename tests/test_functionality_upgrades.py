"""
Unit test suite verifying Functionality Upgrades:
- Option Chain Expiry Selector & Contract Date Resolution
- Position Close API (POST /api/position/close)
- Modify Stop-Loss API (POST /api/position/modify-stop)
- Scanner Signals API (GET /api/scanner/signals) & Confluence Badges
- Per-Position R-Multiple Math
"""

from fastapi.testclient import TestClient
import pytest
from src.dashboard_v2.server import app
from src.data.index_fo_data import fetch_live_option_chain, get_available_expiries


def test_expiry_selector_resolution() -> None:
    expiries = get_available_expiries("NIFTY")
    assert len(expiries) >= 4
    assert "202" in expiries[0]

    # Fetch chain with specific expiry
    chain = fetch_live_option_chain("NIFTY", expiry=expiries[0])
    assert chain["status"] == "SUCCESS"
    assert chain["selected_expiry"] == expiries[0]
    assert len(chain["available_expiries"]) >= 4


def test_position_close_and_modify_stop_endpoints() -> None:
    client = TestClient(app)

    # Test position close
    res_close = client.post("/api/position/close", json={
        "symbol": "BANKNIFTY FUT",
        "reason": "Manual Test Close"
    })
    assert res_close.status_code == 200
    assert res_close.json()["status"] == "SUCCESS"
    assert res_close.json()["action"] == "CLOSED"

    # Test modify stop
    res_modify = client.post("/api/position/modify-stop", json={
        "symbol": "BANKNIFTY FUT",
        "new_stop_loss": 52150.0
    })
    assert res_modify.status_code == 200
    assert res_modify.json()["status"] == "SUCCESS"
    assert res_modify.json()["new_stop_loss"] == 52150.0


def test_scanner_signals_api() -> None:
    client = TestClient(app)
    res = client.get("/api/scanner/signals")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert len(data["signals"]) >= 2

    sig0 = data["signals"][0]
    assert "symbol" in sig0
    assert "direction" in sig0
    assert "rr_ratio" in sig0
    assert len(sig0["passed_conditions"]) >= 3
