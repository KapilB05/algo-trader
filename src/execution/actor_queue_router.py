"""
Actor Model & Event-Driven Queue Router — Algo-Trader Execution Layer (Layer 7).

Inspired by Sentinel PRIME & AI-Trader:
- Lock-free, non-blocking message passing via collections.deque
- Token Bucket Rate Limiter (enforces broker API rate caps during high volatility)
- Event dispatching for ticks, signals, risk checks, and orders
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger("ActorQueueRouter")


@dataclass
class EventMessage:
    """Event container for actor pipeline."""

    event_type: str         # TICK, SIGNAL, RISK_CHECK, ORDER_DISPATCH, FILL, HALT
    payload: Dict[str, Any]
    timestamp_mono: float = field(default_factory=time.monotonic)
    event_id: str = field(default_factory=lambda: str(uuid4())[:8])


class TokenBucketRateLimiter:
    """
    Token Bucket Rate Limiter to prevent broker API throttling / bans.
    Default: 10 orders/sec burst capacity with 5 tokens/sec replenishment.
    """

    def __init__(self, capacity: float = 10.0, refill_rate: float = 5.0) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_update = time.monotonic()

    def acquire(self, tokens_needed: float = 1.0) -> bool:
        """Attempt to acquire tokens."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + (elapsed * self.refill_rate))
        self.last_update = now

        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False


class ActorQueueRouter:
    """
    Event-driven decoupled queue dispatcher for sub-millisecond signal-to-order routing.
    """

    def __init__(self, max_queue_size: int = 10000) -> None:
        self._inbox: deque[EventMessage] = deque(maxlen=max_queue_size)
        self._rate_limiter = TokenBucketRateLimiter(capacity=10.0, refill_rate=5.0)
        self._subscribers: Dict[str, List[Callable[[EventMessage], Any]]] = {}
        self._running: bool = False

    def subscribe(self, event_type: str, handler: Callable[[EventMessage], Any]) -> None:
        """Register an event handler."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def publish(self, event_type: str, payload: Dict[str, Any]) -> EventMessage:
        """Enqueue an event (O(1) operation)."""
        msg = EventMessage(event_type=event_type, payload=payload)
        self._inbox.append(msg)

        # Dispatch immediately if sync handlers exist
        handlers = self._subscribers.get(event_type, [])
        for h in handlers:
            try:
                h(msg)
            except Exception as e:
                logger.error("Error in handler for %s: %s", event_type, e)

        return msg

    def check_rate_limit(self) -> bool:
        """Check if an outgoing order is permitted under broker API limits."""
        return self._rate_limiter.acquire(1.0)


# Global singleton router
event_router = ActorQueueRouter()
