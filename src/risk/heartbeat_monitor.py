"""
Heartbeat Monitor — 5-second Broker server ping with cancel-on-disconnect.

Pings every 5 seconds. On 3 consecutive failures:
- Cancels all open/pending orders.
- Sends Telegram alert.
- Dashboard shows DISCONNECTED banner.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Coroutine, Optional

import httpx

logger = logging.getLogger(__name__)

OnDisconnectCallback = Callable[[], Coroutine[Any, Any, None]]


class HeartbeatMonitor:
    """
    Monitors connectivity to Broker servers.

    Sends lightweight HTTP pings at configured intervals.
    Triggers safety actions on consecutive failures.
    """

    PING_URL = "https://apiconnect.angelone.in"

    def __init__(
        self,
        access_token: str,
        interval_seconds: float = 5.0,
        max_failures: int = 3,
    ) -> None:
        self._token = access_token
        self._interval = interval_seconds
        self._max_failures = max_failures
        self._consecutive_failures = 0
        self._is_connected = True
        self._last_success: float = 0.0
        self._total_pings: int = 0
        self._total_failures: int = 0
        self._running = False

        self._on_disconnect: Optional[OnDisconnectCallback] = None
        self._on_reconnect: Optional[OnDisconnectCallback] = None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "connected": self._is_connected,
            "consecutive_failures": self._consecutive_failures,
            "last_success": self._last_success,
            "total_pings": self._total_pings,
            "total_failures": self._total_failures,
            "uptime_pct": (
                ((self._total_pings - self._total_failures) / self._total_pings * 100)
                if self._total_pings > 0
                else 100.0
            ),
        }

    def set_on_disconnect(self, callback: OnDisconnectCallback) -> None:
        """Set callback for disconnect event (cancel orders, alert)."""
        self._on_disconnect = callback

    def set_on_reconnect(self, callback: OnDisconnectCallback) -> None:
        """Set callback for reconnection event."""
        self._on_reconnect = callback

    async def start(self) -> None:
        """Start the heartbeat monitoring loop."""
        self._running = True
        logger.info(
            "Heartbeat monitor started (interval=%ds, max_failures=%d)",
            self._interval, self._max_failures,
        )

        while self._running:
            await self._ping()
            await asyncio.sleep(self._interval)

    async def stop(self) -> None:
        """Stop the heartbeat monitor."""
        self._running = False
        logger.info("Heartbeat monitor stopped.")

    async def _ping(self) -> None:
        """Send a single heartbeat ping."""
        self._total_pings += 1

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    self.PING_URL,
                    headers={"Authorization": f"Bearer {self._token}"},
                )
                # Any response with status < 500 confirms the broker gateway is reachable and responding
                if response.status_code < 500:
                    await self._on_success()
                else:
                    await self._on_failure(f"HTTP {response.status_code}")

        except httpx.TimeoutException:
            await self._on_failure("Timeout")
        except httpx.ConnectError:
            await self._on_failure("Connection refused")
        except Exception as exc:
            await self._on_failure(str(exc))

    async def _on_success(self) -> None:
        """Handle successful ping."""
        was_disconnected = not self._is_connected
        self._consecutive_failures = 0
        self._is_connected = True
        self._last_success = time.time()

        if was_disconnected:
            logger.info("✅ Connection restored to Broker API.")
            if self._on_reconnect:
                try:
                    await self._on_reconnect()
                except Exception as exc:
                    logger.error("Reconnect callback failed: %s", exc)

    async def _on_failure(self, reason: str) -> None:
        """Handle failed ping."""
        self._consecutive_failures += 1
        self._total_failures += 1

        logger.warning(
            "Heartbeat failure %d/%d: %s",
            self._consecutive_failures, self._max_failures, reason,
        )

        if self._consecutive_failures >= self._max_failures and self._is_connected:
            self._is_connected = False
            logger.critical(
                "🚨 CONNECTIVITY LOST! %d consecutive failures. "
                "Cancelling all orders...",
                self._consecutive_failures,
            )

            if self._on_disconnect:
                try:
                    await self._on_disconnect()
                except Exception as exc:
                    logger.error("Disconnect callback failed: %s", exc)
