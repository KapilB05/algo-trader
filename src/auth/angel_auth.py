"""
Angel One SmartAPI Authentication Manager — Automated Login & Session Management.

Initializes the Angel One SmartAPI client using environment variables and generates live TOTP.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from typing import Any
import threading
import time

import pyotp
from dotenv import load_dotenv

logger = logging.getLogger("AngelAuth")

TOKEN_FILE = Path("data/angel_token.json")


class SmartApiRateLimiter:
    """Thread-safe rate limiter enforcing max 2.0 requests/sec for Angel One free tier."""

    def __init__(self, max_per_sec: float = 2.0):
        self.interval = 1.0 / max_per_sec
        self.last_call = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self.last_call = time.time()


smartapi_rate_limiter = SmartApiRateLimiter(max_per_sec=2.0)


@dataclass
class AngelTokenData:
    """Container for Angel One SmartAPI session tokens."""

    jwt_token: str
    refresh_token: str
    feed_token: str
    api_key: str
    client_id: str
    created_at: str

    @property
    def is_expired(self) -> bool:
        """Check if session is older than 8 hours or from a previous calendar day in IST."""
        try:
            from zoneinfo import ZoneInfo

            IST = ZoneInfo("Asia/Kolkata")
            created = datetime.fromisoformat(self.created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
            now = datetime.now(UTC)

            # Expire if older than 8 hours
            if (now - created).total_seconds() >= 28800:
                return True

            # Expire if date rolled over in IST (Angel One resets sessions at 05:30 IST)
            created_ist = created.astimezone(IST)
            now_ist = now.astimezone(IST)
            if created_ist.date() != now_ist.date():
                return True

            return False
        except Exception:
            return True

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


class AngelManager:
    """Initializes the Angel One SmartAPI client using environment variables."""

    def __init__(self) -> None:
        load_dotenv()

        self.api_key = os.getenv("ANGEL_API_KEY", "")
        self.client_id = os.getenv("ANGEL_CLIENT_ID", "")
        self.pin = os.getenv("ANGEL_PIN", "")
        self.totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

        self.smartApi: Any | None = None
        self._token_data: AngelTokenData | None = None

        if not all([self.api_key, self.client_id, self.pin, self.totp_secret]):
            logger.warning("Missing Angel One credentials in .env file (mock mode enabled).")
            return

        try:
            from SmartApi import SmartConnect

            # Initialize the SmartConnect client and load existing session or authenticate fresh
            self.smartApi = SmartConnect(api_key=self.api_key)
            if not self.load_token():
                logger.info(
                    "No valid cached Angel One session found. Attempting automated login..."
                )
                self.authenticate()
            else:
                logger.info("Loaded valid cached Angel One session token.")
        except ImportError:
            logger.warning("SmartApi package not installed — running in mock mode.")
        except Exception as e:
            logger.error(f"🚨 Failed to initialize SmartConnect: {e}")

    def generate_totp(self) -> str:
        """Generate current 6-digit TOTP code using pyotp."""
        if not self.totp_secret:
            raise ValueError("ANGEL_TOTP_SECRET is not configured in .env")
        totp = pyotp.TOTP(self.totp_secret)
        return totp.now()

    def authenticate(self) -> bool:
        """Generates the TOTP and logs in automatically."""
        if not self.smartApi:
            logger.warning("SmartConnect not initialized.")
            return False

        try:
            # Generate the live TOTP code
            totp = self.generate_totp()

            # Execute login
            data = self.smartApi.generateSession(self.client_id, self.pin, totp)

            if isinstance(data, dict) and data.get("status"):
                logger.info(f"✅ Login Successful! Auth Token generated for: {self.client_id}")

                session_data = data.get("data", {})
                jwt_token = session_data.get("jwtToken", "")
                clean_jwt = jwt_token.replace("Bearer ", "").strip() if jwt_token else ""
                refresh_token = session_data.get("refreshToken", "")
                feed_token = session_data.get("feedToken", "")

                if self.smartApi:
                    self.smartApi.setAccessToken(clean_jwt)
                    if feed_token:
                        self.smartApi.setFeedToken(feed_token)
                    if refresh_token:
                        self.smartApi.setRefreshToken(refresh_token)

                self._token_data = AngelTokenData(
                    jwt_token=clean_jwt,
                    refresh_token=refresh_token,
                    feed_token=feed_token,
                    api_key=self.api_key,
                    client_id=self.client_id,
                    created_at=datetime.now(UTC).isoformat(),
                )
                self.persist_token(self._token_data)
                return True
            else:
                msg = data.get("message") if isinstance(data, dict) else str(data)
                logger.error(f"❌ Login Failed: {msg}")
                return False

        except Exception as e:
            logger.error(f"❌ Authentication threw an exception: {e}")
            return False

    def authenticate_custom(
        self, client_id: str, pin: str, totp_code: str, api_key: str | None = None
    ) -> tuple[bool, str, AngelTokenData | None]:
        """Authenticate dynamically for any user using custom credentials."""
        target_api_key = (api_key or self.api_key or "").strip()
        cid = client_id.strip().upper()
        p = pin.strip()
        totp = totp_code.strip()

        if not target_api_key:
            return False, "Missing SmartAPI API Key. Please provide one.", None
        if not cid:
            return False, "Missing Client ID.", None
        if not p:
            return False, "Missing 4-Digit Trading PIN.", None
        if not totp:
            return False, "Missing TOTP Code.", None

        try:
            from SmartApi import SmartConnect

            # Initialize SmartConnect with target API key
            client = SmartConnect(api_key=target_api_key)
            data = client.generateSession(cid, p, totp)

            if isinstance(data, dict) and data.get("status"):
                logger.info(f"✅ Custom Login Successful for Client: {cid}")
                session_data = data.get("data", {})
                jwt_token = session_data.get("jwtToken", "")
                clean_jwt = jwt_token.replace("Bearer ", "").strip() if jwt_token else ""
                refresh_token = session_data.get("refreshToken", "")
                feed_token = session_data.get("feedToken", "")

                client.setAccessToken(clean_jwt)
                if feed_token:
                    client.setFeedToken(feed_token)
                if refresh_token:
                    client.setRefreshToken(refresh_token)

                self.smartApi = client
                self.client_id = cid
                self.pin = p
                self.api_key = target_api_key

                token_data = AngelTokenData(
                    jwt_token=clean_jwt,
                    refresh_token=refresh_token,
                    feed_token=feed_token,
                    api_key=target_api_key,
                    client_id=cid,
                    created_at=datetime.now(UTC).isoformat(),
                )
                self._token_data = token_data
                self.persist_token(token_data)
                return True, "Login Successful! Session active.", token_data
            else:
                msg = data.get("message") if isinstance(data, dict) else str(data)
                err_code = data.get("errorCode", "") if isinstance(data, dict) else ""
                err_detail = f"{msg} (Code: {err_code})" if err_code else str(msg)
                logger.error(f"❌ Custom Login Failed for {cid}: {err_detail}")
                return False, err_detail, None
        except Exception as exc:
            logger.error(f"❌ Custom authentication exception: {exc}")
            return False, str(exc), None

    def logout(self) -> bool:
        """Clear active session tokens and reset client state."""
        try:
            self._token_data = None
            if self.smartApi:
                try:
                    self.smartApi.setAccessToken("")
                    self.smartApi.setFeedToken("")
                    self.smartApi.setRefreshToken("")
                except Exception as exc:
                    logger.debug("SmartApi state clear exception ignored during logout: %s", exc)

            if TOKEN_FILE.exists():
                TOKEN_FILE.unlink(missing_ok=True)
            tokens_json = Path("data/tokens.json")
            if tokens_json.exists():
                tokens_json.unlink(missing_ok=True)

            logger.info("Angel One session logged out and tokens cleared.")
            return True
        except Exception as exc:
            logger.error("Error during logout: %s", exc)
            return False

    def login(self) -> AngelTokenData | None:
        """Authenticate and return AngelTokenData or None."""
        if self.authenticate():
            return self._token_data
        return None

    def load_token(self) -> AngelTokenData | None:
        """Load persisted session tokens from data/angel_token.json."""
        if not TOKEN_FILE.exists():
            return None

        try:
            raw = json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
            if raw.get("jwt_token"):
                raw["jwt_token"] = raw["jwt_token"].replace("Bearer ", "").strip()
            token = AngelTokenData(**raw)

            if token.is_expired:
                logger.info("Persisted Angel One session token expired. Auto-logging in via TOTP...")
                if self.authenticate():
                    return self._token_data
                return None

            self._token_data = token
            if self.smartApi and token.jwt_token:
                self.smartApi.setAccessToken(token.jwt_token)
                if token.feed_token:
                    self.smartApi.setFeedToken(token.feed_token)
                if token.refresh_token:
                    self.smartApi.setRefreshToken(token.refresh_token)
            return token
        except Exception as exc:
            logger.error("Failed to load Angel One token: %s", exc)
            return None

    def persist_token(self, token: AngelTokenData) -> None:
        """Save token data to data/angel_token.json and data/tokens.json."""
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOKEN_FILE.write_text(json.dumps(token.to_dict(), indent=2), encoding="utf-8")
        tokens_json = Path("data/tokens.json")
        tokens_json.write_text(
            json.dumps(
                {
                    "jwt_token": token.jwt_token,
                    "feed_token": token.feed_token,
                    "updated_at": token.created_at,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Angel One token persisted to %s and %s", TOKEN_FILE, tokens_json)

    def get_authenticated_client(self) -> Any:
        """Return the authenticated SmartConnect client, authenticating if necessary."""
        if not self.smartApi or not self._token_data or self._token_data.is_expired:
            self.load_token()
        if not self.smartApi or not self._token_data:
            self.authenticate()
        return self.smartApi


# Alias for backward compatibility
AngelAuth = AngelManager

# Create a singleton instance to be imported by other modules
api_manager = AngelManager()
angel_client = api_manager.smartApi

if __name__ == "__main__":
    # Run this file directly to test the automated login
    api_manager.authenticate()
