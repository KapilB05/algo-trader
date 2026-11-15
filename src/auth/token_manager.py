"""
Token Manager — handles token lifecycle for the trading engine.

Responsibilities:
- Load persisted token from data/angel_token.json.
- Provide get_valid_token() for all downstream modules.
- Check token authentication state.
- Exclusively optimized for Angel One SmartAPI.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.auth.angel_auth import AngelAuth

logger = logging.getLogger(__name__)


class TokenManager:
    """
    Central token authority for the trading engine.

    Exclusively optimized for Angel One SmartAPI.
    All modules that need an access token call `get_valid_token()`.
    """

    def __init__(self, broker: str = "angelone") -> None:
        self.broker = broker
        self._angel_auth: Optional[AngelAuth] = None
        self._cached_token: Optional[str] = None

    @property
    def angel_auth(self) -> AngelAuth:
        if self._angel_auth is None:
            self._angel_auth = AngelAuth()
        return self._angel_auth

    def load(self) -> Optional[str]:
        """Attempt to load a valid access token from disk or authenticate fresh."""
        token = self.angel_auth.load_token()
        if token and not token.is_expired:
            logger.info("Loaded valid Angel One access token.")
            self._cached_token = token.jwt_token
            return token.jwt_token

        # Try fresh authentication via TOTP
        if self.angel_auth.authenticate():
            token = self.angel_auth._token_data
            if token:
                self._cached_token = token.jwt_token
                return token.jwt_token
        return None

    def get_valid_token(self) -> str:
        """
        Return a valid access token string.

        Raises RuntimeError if no valid token is available.
        """
        if self._cached_token:
            return self._cached_token

        token_str = self.load()
        if token_str is not None:
            self._cached_token = token_str
            return token_str

        raise RuntimeError(
            f"No valid access token available for broker: {self.broker}. "
            "Please check your credentials in .env."
        )

    def set_token(self, token_str: str) -> None:
        """Manually set an access token string."""
        self._cached_token = token_str
        logger.info("Token manually set.")

    @property
    def is_authenticated(self) -> bool:
        """Check if a valid token is currently available."""
        if self._cached_token:
            return True
        return self.load() is not None
