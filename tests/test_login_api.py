"""
Unit tests for Angel One SmartAPI Gateway and Authentication API endpoints.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.dashboard_v2.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_auth_status_endpoint(client):
    res = client.get("/api/auth/status")
    assert res.status_code == 200
    data = res.json()
    assert "authenticated" in data
    assert "client_id" in data
    assert "is_configured" in data
    assert "has_secret" in data
    assert "funds" in data


def test_auth_login_auto_mode(client):
    with patch("src.auth.angel_auth.api_manager.generate_totp") as mock_totp, \
         patch("src.auth.angel_auth.api_manager.authenticate_custom") as mock_auth, \
         patch("src.auth.angel_auth.api_manager.smartApi") as mock_api:

        mock_totp.return_value = "123456"
        mock_token = MagicMock()
        mock_token.created_at = "2026-08-18T08:00:00Z"
        mock_auth.return_value = (True, "Login Successful! Session active.", mock_token)
        mock_api.rmsLimit.return_value = {"status": True, "data": {"net": 250000.0, "availablecash": 250000.0}}

        res = client.post("/api/auth/login", json={"mode": "auto"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert "Successfully authenticated" in data["message"]
        assert data["funds"] == 250000.0


def test_auth_login_manual_validation_and_success(client):
    # 1. Missing TOTP
    res_err = client.post("/api/auth/login", json={"mode": "manual", "client_id": "S1680402", "pin": "1234", "totp": ""})
    assert res_err.status_code == 400
    assert "6-digit TOTP code is required" in res_err.json()["detail"]

    # 2. Valid Manual TOTP login
    with patch("src.auth.angel_auth.api_manager.authenticate_custom") as mock_auth, \
         patch("src.auth.angel_auth.api_manager.smartApi") as mock_api:

        mock_token = MagicMock()
        mock_token.created_at = "2026-08-18T08:00:00Z"
        mock_auth.return_value = (True, "Login Successful!", mock_token)
        mock_api.rmsLimit.return_value = {"status": True, "data": {"net": 180000.0, "availablecash": 180000.0}}

        res = client.post("/api/auth/login", json={"mode": "manual", "client_id": "S1680402", "pin": "1234", "totp": "654321"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["client_id"] == "S1680402"
        assert data["funds"] == 180000.0


def test_setup_credentials_endpoint(client, tmp_path):
    with patch("src.dashboard_v2.server.update_env_file") as mock_env_update, \
         patch("src.auth.angel_auth.api_manager.generate_totp") as mock_totp, \
         patch("src.auth.angel_auth.api_manager.authenticate_custom") as mock_auth, \
         patch("src.auth.angel_auth.api_manager.smartApi") as mock_api:

        mock_totp.return_value = "888999"
        mock_token = MagicMock()
        mock_auth.return_value = (True, "Login Successful!", mock_token)
        mock_api.rmsLimit.return_value = {"status": True, "data": {"net": 500000.0}}

        payload = {
            "api_key": "TEST_API_KEY",
            "client_id": "TEST_USER_99",
            "pin": "9999",
            "totp_secret": "JBSWY3DPEHPK3PXP",
        }
        res = client.post("/api/auth/credentials", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "SUCCESS"
        assert data["client_id"] == "TEST_USER_99"
        assert mock_env_update.called


def test_auth_logout_endpoint(client):
    with patch("src.auth.angel_auth.api_manager.logout") as mock_logout:
        mock_logout.return_value = True
        res = client.post("/api/auth/logout")
        assert res.status_code == 200
        assert res.json()["status"] == "SUCCESS"
