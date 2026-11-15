"""
Unit tests verifying Dashboard Fixes & Endpoints (tests/test_dashboard_fixes.py).
"""

from fastapi.testclient import TestClient

from src.dashboard_v2.server import app
from src.strategy.morning_bias import MorningPrepEngine

client = TestClient(app)


def test_morning_prep_pivot_points() -> None:
    engine = MorningPrepEngine()
    passed, report = engine.run_morning_prep()

    assert report.key_levels.pivot > 0
    assert report.key_levels.support_1 > 0
    assert report.key_levels.support_2 > 0
    assert report.key_levels.resistance_1 > 0
    assert report.key_levels.resistance_2 > 0


def test_status_endpoint_pivot_levels() -> None:
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()

    assert data["system_status"] in ("READY", "NOT_READY")
    assert "key_levels" in data
    assert data["key_levels"]["pivot"] > 0
    assert "gift_nifty_gap" in data


def test_order_dispatch_endpoint() -> None:
    response = client.post("/api/order", json={
        "symbol": "BANKNIFTY FUT",
        "side": "BUY",
        "quantity": 15,
        "order_type": "MARKET",
        "product_type": "MIS",
        "price": 0.0,
    })
    assert response.status_code == 200
    data = response.json()

    assert data["status"] in ("SUCCESS", "PLACED", "EXECUTED")
    assert "order_id" in data


def test_multi_alert_scanner_endpoint() -> None:
    response = client.get("/api/scanner")
    assert response.status_code == 200
    signals = response.json()

    assert isinstance(signals, list)
    assert len(signals) >= 4

    first = signals[0]
    assert "symbol" in first
    assert "side" in first
    assert "entry" in first
    assert "sl" in first
    assert "target" in first
    assert "rr_ratio" in first
    assert "confidence" in first
