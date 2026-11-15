"""
Execution Modes — Auto / Semi-Auto / Manual mode router.

Routes signals based on execution_mode setting:
- auto:   Signal → validate R:R → compute size → place order immediately
- semi:   Signal → validate → push to dashboard for user click
- manual: No algorithmic signals; user dispatches from terminal
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.strategy.confluence_engine import ConfluenceResult, Direction

logger = logging.getLogger(__name__)


class ExecutionMode(str, Enum):
    PAPER = "paper"
    AUTO = "auto"
    SEMI = "semi"
    MANUAL = "manual"


@dataclass
class TradeSignal:
    """A trade signal ready for execution."""

    instrument_token: str
    direction: Direction
    entry_price: float
    stop_loss: float
    target: float
    quantity: int
    confidence: float
    confluence: ConfluenceResult
    option_setup: Optional[dict] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def risk_reward_ratio(self) -> float:
        if self.entry_price == self.stop_loss:
            return 0.0
        risk = abs(self.entry_price - self.stop_loss)
        reward = abs(self.target - self.entry_price)
        return reward / risk if risk > 0 else 0.0


class ExecutionRouter:
    """
    Routes trade signals based on the active execution mode.

    In PAPER mode, signals are executed in simulation with zero broker risk.
    In AUTO mode, signals are dispatched immediately to broker.
    In SEMI mode, signals are queued for user confirmation.
    In MANUAL mode, algorithmic signals are suppressed.
    """

    def __init__(self, mode: ExecutionMode = ExecutionMode.PAPER) -> None:
        self._mode = mode
        # Queue for semi-auto mode: signals awaiting user approval
        self._pending_signals: list[TradeSignal] = []
        # Callback for auto-dispatch
        self._dispatch_callback: Optional[Any] = None

    @property
    def mode(self) -> ExecutionMode:
        return self._mode

    def set_mode(self, mode: ExecutionMode) -> None:
        self._mode = mode
        logger.info("Execution mode set to: %s", mode.value)

    def set_dispatch_callback(self, callback: Any) -> None:
        """Set the callback for auto-dispatching orders."""
        self._dispatch_callback = callback

    async def route_signal(self, signal: TradeSignal) -> dict[str, Any]:
        """
        Route a trade signal based on current execution mode.

        Returns a status dict with the routing outcome.
        """
        if self._mode == ExecutionMode.MANUAL:
            logger.debug("Manual mode — signal suppressed for %s", signal.instrument_token)
            return {"action": "suppressed", "mode": "manual"}

        # Validate minimum R:R ratio
        if signal.risk_reward_ratio < 1.5:
            logger.warning(
                "Signal rejected: R:R ratio %.1f < 1.5 for %s",
                signal.risk_reward_ratio, signal.instrument_token,
            )
            return {
                "action": "rejected",
                "reason": f"R:R ratio {signal.risk_reward_ratio:.1f} < 1.5",
            }

        if self._mode == ExecutionMode.PAPER:
            logger.info(
                "PAPER execution: Simulated order for %s %s @ %.2f | Qty: %d",
                signal.direction.value, signal.instrument_token, signal.entry_price, signal.quantity,
            )
            return {"action": "paper_executed", "mode": "paper"}

        if self._mode == ExecutionMode.AUTO:
            return await self._auto_dispatch(signal)
        if self._mode == ExecutionMode.SEMI:
            return self._semi_queue(signal)

        return {"action": "unknown_mode", "mode": str(self._mode)}

    async def _auto_dispatch(self, signal: TradeSignal) -> dict[str, Any]:
        """Auto mode: dispatch immediately."""
        logger.info(
            "AUTO dispatch: %s %s @ %.2f | SL: %.2f | TGT: %.2f | Qty: %d",
            signal.direction.value, signal.instrument_token,
            signal.entry_price, signal.stop_loss, signal.target, signal.quantity,
        )

        if self._dispatch_callback:
            try:
                await self._dispatch_callback(signal)
                return {"action": "dispatched", "mode": "auto"}
            except Exception as exc:
                logger.error("Auto dispatch failed: %s", exc)
                return {"action": "error", "error": str(exc)}

        return {"action": "no_callback"}

    def _semi_queue(self, signal: TradeSignal) -> dict[str, Any]:
        """Semi mode: queue for user approval on dashboard."""
        self._pending_signals.append(signal)
        logger.info(
            "SEMI queue: %s %s @ %.2f | R:R: %.1f | Confidence: %.2f",
            signal.direction.value, signal.instrument_token,
            signal.entry_price, signal.risk_reward_ratio, signal.confidence,
        )
        return {
            "action": "queued",
            "mode": "semi",
            "pending_count": len(self._pending_signals),
        }

    def get_pending_signals(self) -> list[TradeSignal]:
        """Get signals awaiting user approval (semi mode)."""
        return list(self._pending_signals)

    async def approve_signal(self, index: int) -> dict[str, Any]:
        """Approve and dispatch a pending signal by index."""
        if 0 <= index < len(self._pending_signals):
            signal = self._pending_signals.pop(index)
            return await self._auto_dispatch(signal)
        return {"action": "invalid_index"}

    def reject_signal(self, index: int) -> None:
        """Reject a pending signal."""
        if 0 <= index < len(self._pending_signals):
            removed = self._pending_signals.pop(index)
            logger.info("Signal rejected by user: %s", removed.instrument_token)

    def clear_pending(self) -> int:
        """Clear all pending signals. Returns count cleared."""
        count = len(self._pending_signals)
        self._pending_signals.clear()
        return count
