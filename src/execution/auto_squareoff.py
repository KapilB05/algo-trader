"""
Auto Square-Off — 3:15 PM IST intraday position closure.

Scheduled task that closes all intraday positions before market close.
Sends market orders for all open intraday positions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from datetime import time as dt_time
from typing import Optional
from zoneinfo import ZoneInfo

from src.audit.audit_logger import AuditLogger

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")


class AutoSquareOff:
    """
    Automated intraday position square-off at configured time.

    Default: 3:15 PM IST. Sends market exit orders for all open
    intraday positions. Logs details to audit trail.
    """

    def __init__(
        self,
        squareoff_time: str = "15:15",
        warn_minutes_before: int = 5,
        audit: Optional[AuditLogger] = None,
    ) -> None:
        parts = squareoff_time.split(":")
        self._squareoff_time = dt_time(int(parts[0]), int(parts[1]))
        self._warn_minutes = warn_minutes_before
        self._audit = audit
        self._triggered_today = False
        self._warning_sent = False

        # Callbacks
        self._squareoff_callback = None
        self._warning_callback = None

    def set_squareoff_callback(self, callback) -> None:
        """Set callback to execute square-off orders."""
        self._squareoff_callback = callback

    def set_warning_callback(self, callback) -> None:
        """Set callback for pre-squareoff warning."""
        self._warning_callback = callback

    async def check(self) -> Optional[str]:
        """
        Check if it's time to square off. Call this periodically.

        Returns:
            "warned" if warning was sent
            "squaredoff" if positions were closed
            None if no action needed
        """
        now = datetime.now(IST)
        # Weekday check: Monday=0, Sunday=6. Market is closed on Saturday (5) and Sunday (6)
        if now.weekday() >= 5:
            return None

        current_time = now.time()

        # Reset daily flag
        if current_time < dt_time(9, 0):
            self._triggered_today = False
            self._warning_sent = False
            return None

        if self._triggered_today:
            return None

        # ── Warning phase ─────────────────────────────────────────────
        warn_time = (
            datetime.combine(now.date(), self._squareoff_time)
            - timedelta(minutes=self._warn_minutes)
        ).time()

        if current_time >= warn_time and not self._warning_sent:
            self._warning_sent = True
            logger.warning(
                "⚠️ Auto square-off in %d minutes at %s IST!",
                self._warn_minutes, self._squareoff_time.strftime("%H:%M"),
            )
            if self._warning_callback:
                try:
                    await self._warning_callback()
                except Exception as exc:
                    logger.error("Warning callback failed: %s", exc)
            return "warned"

        # ── Square-off phase ──────────────────────────────────────────
        if current_time >= self._squareoff_time:
            self._triggered_today = True
            logger.warning("🔒 AUTO SQUARE-OFF TRIGGERED at %s IST", now.strftime("%H:%M:%S"))

            if self._audit:
                self._audit.log_event(
                    event_type="AUTO_SQUAREOFF",
                    trace_id="",
                    payload={"time": now.isoformat(), "squareoff_time": str(self._squareoff_time)},
                    source="auto_squareoff",
                )

            if self._squareoff_callback:
                try:
                    await self._squareoff_callback()
                except Exception as exc:
                    logger.error("Square-off callback failed: %s", exc)

            return "squaredoff"

        return None

    @property
    def is_triggered_today(self) -> bool:
        return self._triggered_today

    @property
    def squareoff_time(self) -> dt_time:
        return self._squareoff_time

    def time_to_squareoff(self) -> Optional[timedelta]:
        """Get time remaining until square-off."""
        now = datetime.now(IST)
        squareoff_dt = datetime.combine(now.date(), self._squareoff_time, tzinfo=IST)
        if now >= squareoff_dt:
            return None
        return squareoff_dt - now
