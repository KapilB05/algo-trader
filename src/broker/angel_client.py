"""
Angel One SmartAPI Client Factory — Lazily initializes SmartConnect SDK instances.

Wraps SmartApi.SmartConnect to provide authenticated API access.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.auth.token_manager import TokenManager

logger = logging.getLogger(__name__)


class AngelClientFactory:
    """
    Factory for Angel One SmartConnect client objects.

    Lazily initializes and authenticates `SmartConnect` using API key & session.
    """

    def __init__(self, token_manager: TokenManager) -> None:
        self._token_mgr = token_manager
        self._smart_api_instance: Optional[object] = None
        self._mock_fallback_used: bool = False

    @property
    def is_mock(self) -> bool:
        """True if the SDK/auth failed and a simulated client is in use."""
        return self._mock_fallback_used

    @property
    def client(self) -> object:
        """Get or initialize authenticated SmartConnect client."""
        if self._smart_api_instance is None:
            auth = self._token_mgr.angel_auth
            token_data = auth.load_token()

            try:
                from SmartApi import SmartConnect

                smart_api = SmartConnect(api_key=auth.api_key)
                if token_data and not token_data.is_expired:
                    smart_api.setAccessToken(token_data.jwt_token)
                else:
                    new_token = auth.login()
                    if new_token:
                        smart_api.setAccessToken(new_token.jwt_token)

                self._smart_api_instance = smart_api
                logger.info("Angel One SmartAPI SDK client initialized.")
            except ImportError:
                logger.warning("SmartApi not installed — running in fallback/mock mode.")
                self._mock_fallback_used = True
                self._smart_api_instance = self._build_mock_client()
            except Exception as exc:
                logger.error("Failed to initialize Angel One SDK: %s", exc)
                self._mock_fallback_used = True
                self._smart_api_instance = self._build_mock_client()

        return self._smart_api_instance

    def _build_mock_client(self) -> object:
        """Fallback mock Angel One client."""
        class MockAngel:
            def placeOrder(self, params):
                return {"status": True, "message": "SUCCESS", "data": {"orderid": "ANGEL-ORD-001"}}
            def cancelOrder(self, orderid, variety="NORMAL"):
                return {"status": True, "message": "SUCCESS", "data": {"orderid": orderid}}
            def modifyOrder(self, params):
                return {"status": True, "message": "SUCCESS", "data": {"orderid": params.get("orderid")}}
            def orderBook(self):
                return {"status": True, "data": []}
            def position(self):
                return {"status": True, "data": []}
            def holdings(self):
                return {"status": True, "data": []}
            def rmsLimit(self):
                return {"status": True, "data": {"net": "100000.0"}}
        return MockAngel()
