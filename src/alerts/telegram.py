"""
Telegram Alerts — Push notifications for signals, fills, and system health.

Uses python-telegram-bot to send formatted messages:
- Entry signals, exit fills, trailing stop updates
- System health warnings (heartbeat loss, circuit breaker, kill switch)
- Morning bias summary
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


class AlertType(str, Enum):
    ENTRY_SIGNAL = "entry_signal"
    EXIT_FILL = "exit_fill"
    TRAILING_STOP = "trailing_stop"
    CIRCUIT_BREAKER = "circuit_breaker"
    KILL_SWITCH = "kill_switch"
    HEARTBEAT_LOSS = "heartbeat_loss"
    HEARTBEAT_RESTORED = "heartbeat_restored"
    MORNING_BIAS = "morning_bias"
    SYSTEM_INFO = "system_info"
    SQUAREOFF = "squareoff"
    CONFLUENCE_HIT = "confluence_hit"


# ── Emoji map ─────────────────────────────────────────────────────────
ALERT_EMOJI = {
    AlertType.ENTRY_SIGNAL: "📈",
    AlertType.EXIT_FILL: "📉",
    AlertType.TRAILING_STOP: "🔄",
    AlertType.CIRCUIT_BREAKER: "🚨",
    AlertType.KILL_SWITCH: "🛑",
    AlertType.HEARTBEAT_LOSS: "💔",
    AlertType.HEARTBEAT_RESTORED: "💚",
    AlertType.MORNING_BIAS: "🌅",
    AlertType.SYSTEM_INFO: "ℹ️",
    AlertType.SQUAREOFF: "🔒",
    AlertType.CONFLUENCE_HIT: "🎯",
}


class TelegramAlerts:
    """
    Sends formatted Telegram notifications for trading events.

    Enable/disable specific alert types via config/settings.yaml.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = False,
        send_entry_signals: bool = True,
        send_exit_fills: bool = True,
        send_system_health: bool = True,
        send_morning_bias: bool = True,
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = enabled and bool(self._token) and bool(self._chat_id)
        self._send_entry = send_entry_signals
        self._send_exit = send_exit_fills
        self._send_health = send_system_health
        self._send_bias = send_morning_bias
        self._bot: Any = None

        if self._enabled:
            try:
                from telegram import Bot
                self._bot = Bot(token=self._token)
                logger.info("Telegram alerts initialized.")
            except ImportError:
                logger.warning("python-telegram-bot not installed. Alerts disabled.")
                self._enabled = False
            except Exception as exc:
                logger.warning("Telegram init failed: %s", exc)
                self._enabled = False
        else:
            logger.info("Telegram alerts disabled.")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    async def send(self, alert_type: AlertType, message: str) -> bool:
        """
        Send a Telegram alert if the type is enabled.

        Returns True if sent successfully.
        """
        if not self._enabled or not self._bot:
            return False

        # Check if this alert type is enabled
        if alert_type == AlertType.ENTRY_SIGNAL and not self._send_entry:
            return False
        if alert_type == AlertType.EXIT_FILL and not self._send_exit:
            return False
        if alert_type in (
            AlertType.CIRCUIT_BREAKER, AlertType.KILL_SWITCH,
            AlertType.HEARTBEAT_LOSS, AlertType.HEARTBEAT_RESTORED,
        ) and not self._send_health:
            return False
        if alert_type == AlertType.MORNING_BIAS and not self._send_bias:
            return False

        emoji = ALERT_EMOJI.get(alert_type, "📌")
        timestamp = datetime.now(IST).strftime("%H:%M:%S")
        formatted = f"{emoji} *{alert_type.value.upper().replace('_', ' ')}*\n{message}\n\n⏰ {timestamp} IST"

        try:
            await self._bot.send_message(
                chat_id=self._chat_id,
                text=formatted,
                parse_mode="Markdown",
            )
            logger.debug("Telegram alert sent: %s", alert_type.value)
            return True
        except Exception as exc:
            logger.error("Telegram send failed: %s", exc)
            return False

    # ── Convenience Methods ───────────────────────────────────────────

    async def signal_alert(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        target: float,
        confidence: float,
    ) -> bool:
        """Send an entry signal alert."""
        rr = abs(target - entry) / abs(entry - stop_loss) if abs(entry - stop_loss) > 0 else 0
        msg = (
            f"*{symbol}* — {direction}\n"
            f"Entry: ₹{entry:,.2f}\n"
            f"Stop Loss: ₹{stop_loss:,.2f}\n"
            f"Target: ₹{target:,.2f}\n"
            f"R:R: {rr:.1f} | Confidence: {confidence:.0%}"
        )
        return await self.send(AlertType.ENTRY_SIGNAL, msg)

    async def exit_alert(
        self,
        symbol: str,
        direction: str,
        entry: float,
        exit_price: float,
        pnl: float,
        reason: str = "Target/Stop",
    ) -> bool:
        """Send an exit fill alert."""
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        msg = (
            f"{pnl_emoji} *{symbol}* — {direction} CLOSED\n"
            f"Entry: ₹{entry:,.2f} → Exit: ₹{exit_price:,.2f}\n"
            f"P&L: ₹{pnl:+,.2f}\n"
            f"Reason: {reason}"
        )
        return await self.send(AlertType.EXIT_FILL, msg)

    async def circuit_breaker_alert(self, drawdown_pct: float, pnl: float) -> bool:
        """Send circuit breaker trigger alert."""
        msg = (
            f"Daily drawdown limit breached!\n"
            f"Drawdown: {drawdown_pct:.1%}\n"
            f"Day P&L: ₹{pnl:+,.0f}\n"
            f"*All new entries blocked for today.*"
        )
        return await self.send(AlertType.CIRCUIT_BREAKER, msg)

    async def killswitch_alert(self, reason: str) -> bool:
        """Send financial kill switch alert."""
        msg = f"Trading halted — alpha degradation detected.\n{reason}"
        return await self.send(AlertType.KILL_SWITCH, msg)

    async def heartbeat_alert(self, connected: bool) -> bool:
        """Send heartbeat status change alert."""
        if connected:
            return await self.send(AlertType.HEARTBEAT_RESTORED, "Connection to Broker API restored.")
        else:
            return await self.send(
                AlertType.HEARTBEAT_LOSS,
                "Connection to Broker API LOST!\nAll pending orders cancelled.",
            )

    async def morning_bias_alert(
        self,
        bias: str,
        sp500_change: float,
        vix: float,
        pivot: float,
    ) -> bool:
        """Send morning bias summary."""
        bias_emoji = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}.get(bias, "⚪")
        msg = (
            f"{bias_emoji} Daily Bias: *{bias}*\n"
            f"S&P 500: {sp500_change:+.1f}%\n"
            f"India VIX: {vix:.1f}\n"
            f"Nifty Pivot: {pivot:,.0f}"
        )
        return await self.send(AlertType.MORNING_BIAS, msg)

    async def confluence_alert(
        self, symbol: str, direction: str, score: float, checks_passed: int
    ) -> bool:
        """Send confluence hit alert."""
        msg = (
            f"*{symbol}* — {direction}\n"
            f"Score: {score:.0%} | Checks: {checks_passed}/4\n"
            f"Review on dashboard Scanner page."
        )
        return await self.send(AlertType.CONFLUENCE_HIT, msg)
