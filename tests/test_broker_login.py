"""
Unit tests for Angel One Broker Authentication & Account Sync endpoints.

All broker calls are mocked — no live API requests, no token file writes.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.dashboard_v2.server import app


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_auth():
    mock_token = MagicMock()
    mock_token.created_at = "2026-08-18T07:30:00Z"
    with patch("src.auth.angel_auth.api_manager.authenticate_custom") as mock_auth_call, \
         patch("src.auth.angel_auth.api_manager.generate_totp") as mock_totp, \
         patch("src.auth.angel_auth.api_manager.smartApi") as mock_api:
        mock_auth_call.return_value = (True, "Login Successful! Session active.", mock_token)
        mock_totp.return_value = "123456"
        mock_api.rmsLimit.return_value = {"status": True, "data": {"net": 150000.0, "availablecash": 150000.0}}
        yield mock_auth_call, mock_token


def test_broker_status_endpoint(client):
    res = client.get("/api/broker/status")
    assert res.status_code == 200
    data = res.json()
    assert "authenticated" in data
    assert "client_id" in data
    assert "funds" in data
    assert "mode" in data


def test_broker_login_manual_missing_totp(client):
    res = client.post(
        "/api/broker/login",
        json={"mode": "manual", "client_id": "TEST1234", "pin": "1234", "totp": ""},
    )
    assert res.status_code == 400
    assert "6-digit TOTP code is required" in res.json()["detail"]


def test_broker_login_manual_success(client, mock_auth):
    mock_auth_call, _ = mock_auth
    res = client.post(
        "/api/broker/login",
        json={"mode": "manual", "client_id": "A987654", "pin": "4321", "totp": "654321"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["client_id"] == "A987654"
    assert data["funds"] == 150000.0
    mock_auth_call.assert_called_once()
    _, kwargs = mock_auth_call.call_args
    assert kwargs["client_id"] == "A987654"
    assert kwargs["totp_code"] == "654321"


def test_broker_login_auto_totp_with_entered_creds(client, mock_auth):
    """Auto-Secret tab: entered client_id/pin, TOTP generated from configured secret."""
    mock_auth_call, _ = mock_auth
    res = client.post(
        "/api/broker/login",
        json={"mode": "auto", "auto_totp": True, "client_id": "B567890", "pin": "9999", "totp": ""},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    assert data["client_id"] == "B567890"
    mock_auth_call.assert_called_once()
    _, kwargs = mock_auth_call.call_args
    assert kwargs["client_id"] == "B567890"
    assert kwargs["pin"] == "9999"
    assert kwargs["totp_code"] == "123456"


def test_broker_login_auto_env_creds(client, mock_auth):
    """1-Click gateway: no entered creds — uses .env credentials with generated TOTP."""
    mock_auth_call, _ = mock_auth
    res = client.post("/api/broker/login", json={"mode": "auto"})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "SUCCESS"
    mock_auth_call.assert_called_once()


def test_broker_login_failure_returns_401(client):
    with patch("src.auth.angel_auth.api_manager.authenticate_custom") as mock_auth_call, \
         patch("src.auth.angel_auth.api_manager.generate_totp") as mock_totp, \
         patch("src.auth.angel_auth.api_manager.smartApi") as mock_api:
        mock_auth_call.return_value = (False, "Invalid TOTP code (Code: AB1001)", None)
        mock_totp.return_value = "000000"
        mock_api.rmsLimit.return_value = {"status": False, "data": {}}

        res = client.post("/api/broker/login", json={"mode": "auto"})
        assert res.status_code == 401
        assert "AB1001" in res.json()["detail"]


def test_broker_logout_endpoint(client):
    with patch("src.auth.angel_auth.api_manager.logout") as mock_logout:
        mock_logout.return_value = True
        res = client.post("/api/broker/logout")
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"


def test_broker_positions_endpoint(client):
    res = client.get("/api/broker/positions")
    assert res.status_code == 200
    data = res.json()
    assert "positions" in data
    assert "status" in data


def test_broker_funds_endpoint(client):
    res = client.get("/api/broker/funds")
    assert res.status_code == 200
    data = res.json()
    assert "available_margin" in data
    assert "status" in data