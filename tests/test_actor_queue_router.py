"""
Unit Tests for Actor Model Queue Router & Token Bucket Rate Limiter (Layer 7).
"""

import pytest
from src.execution.actor_queue_router import ActorQueueRouter, TokenBucketRateLimiter


def test_token_bucket_rate_limiter():
    """Verify Token Bucket permits bursts up to capacity and limits overflow."""
    limiter = TokenBucketRateLimiter(capacity=3.0, refill_rate=1.0)

    # 3 rapid tokens permitted
    assert limiter.acquire(1.0) is True
    assert limiter.acquire(1.0) is True
    assert limiter.acquire(1.0) is True

    # 4th immediate token denied (empty bucket)
    assert limiter.acquire(1.0) is False


def test_actor_queue_publish_subscribe():
    """Verify event router dispatches published events to subscribers."""
    router = ActorQueueRouter()
    received_events = []

    def on_signal(event):
        received_events.append(event.payload.get("symbol"))

    router.subscribe("SIGNAL", on_signal)
    router.publish("SIGNAL", {"symbol": "BANKNIFTY", "action": "BUY"})

    assert len(received_events) == 1
    assert received_events[0] == "BANKNIFTY"
