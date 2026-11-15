"""
Background Scanner — Silent worker that processes 1-minute candles
and flags zero-FOMO trade setups when confluence criteria hit.

Runs as an asyncio background task.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

from src.alerts.telegram import TelegramAlerts
from src.data.candle_aggregator import Candle, CandleAggregator
from src.strategy.confluence_engine import ConfluenceEngine, ConfluenceResult, Direction

logger = logging.getLogger(__name__)
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class ScannerAlert:
    """A scanner alert for a qualifying setup."""

    timestamp: datetime
    instrument_token: str
    symbol: str
    direction: Direction
    confidence: float
    confluence: ConfluenceResult
    current_price: float
    acknowledged: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "instrument_token": self.instrument_token,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "confidence": self.confidence,
            "checks_passed": self.confluence.checks_passed,
            "total_checks": self.confluence.total_checks,
            "score": self.confluence.score,
            "current_price": self.current_price,
            "acknowledged": self.acknowledged,
        }


class BackgroundScanner:
    """
    Continuously scans the watchlist for confluence setups.

    Processes completed 1-minute candles and evaluates the
    confluence checklist for each instrument. Qualifying setups
    are flagged on the dashboard and pushed via Telegram.
    """

    def __init__(
        self,
        confluence_engine: ConfluenceEngine,
        candle_aggregator: CandleAggregator,
        telegram: Optional[TelegramAlerts] = None,
        min_confidence: float = 0.5,
        max_alerts: int = 100,
    ) -> None:
        self._confluence = confluence_engine
        self._aggregator = candle_aggregator
        self._telegram = telegram
        self._min_confidence = min_confidence
        self._max_alerts = max_alerts

        # Alert storage (most recent first)
        self._alerts: list[ScannerAlert] = []

        # Track last scanned candle per instrument to avoid duplicates
        self._last_scanned: dict[str, datetime] = {}

        self._running = False
        self._scan_count = 0

    @property
    def alerts(self) -> list[ScannerAlert]:
        """Get all scanner alerts (most recent first)."""
        return list(reversed(self._alerts))

    @property
    def unacknowledged_alerts(self) -> list[ScannerAlert]:
        """Get alerts not yet acknowledged by the user."""
        return [a for a in self.alerts if not a.acknowledged]

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "total_scans": self._scan_count,
            "total_alerts": len(self._alerts),
            "unacknowledged": len(self.unacknowledged_alerts),
        }

    async def on_candle(self, candle: Candle) -> None:
        """
        Process a completed candle — callback from CandleAggregator.

        Evaluates confluence for the instrument and generates alerts.
        """
        token = candle.instrument_token

        # Skip if we already scanned this candle
        if token in self._last_scanned and self._last_scanned[token] >= candle.timestamp:
            return

        self._last_scanned[token] = candle.timestamp
        self._scan_count += 1

        # Evaluate confluence
        result = self._confluence.evaluate(token, candle.close)

        # Generate alert if confluence passes or score exceeds threshold
        if result.passed or result.score >= self._min_confidence:
            alert = ScannerAlert(
                timestamp=datetime.now(IST),
                instrument_token=token,
                symbol=token.split("|")[-1] if "|" in token else token,
                direction=result.direction,
                confidence=result.score,
                confluence=result,
                current_price=candle.close,
            )

            self._alerts.append(alert)

            # Cap alert history
            if len(self._alerts) > self._max_alerts:
                self._alerts = self._alerts[-self._max_alerts:]

            log_level = logging.INFO if result.passed else logging.DEBUG
            logger.log(
                log_level,
                "🎯 Scanner alert: %s %s | Score: %.2f | %d/%d checks",
                alert.symbol,
                alert.direction.value,
                alert.confidence,
                result.checks_passed,
                result.total_checks,
            )

            # Push Telegram alert for full confluence hits
            if result.passed and self._telegram:
                await self._telegram.confluence_alert(
                    symbol=alert.symbol,
                    direction=alert.direction.value,
                    score=alert.confidence,
                    checks_passed=result.checks_passed,
                )

    def acknowledge_alert(self, index: int) -> None:
        """Mark an alert as acknowledged."""
        alerts = self.alerts
        if 0 <= index < len(alerts):
            alerts[index].acknowledged = True

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns count cleared."""
        count = len(self._alerts)
        self._alerts.clear()
        return count

    async def start(self) -> None:
        """Register as a candle callback and start scanning."""
        self._running = True
        self._aggregator.register_callback(self.on_candle)
        logger.info("Background scanner started.")

    async def stop(self) -> None:
        """Stop the scanner and unregister the candle callback."""
        self._running = False
        self._aggregator.unregister_callback(self.on_candle)
        logger.info("Background scanner stopped.")
