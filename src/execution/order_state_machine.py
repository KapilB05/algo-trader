"""
Order State Machine — Idempotent order lifecycle tracking.

Uses python-statemachine to model:
  New → Pending → Acknowledged → Filled | PartiallyFilled | Cancelled | Rejected

Each order instance has a unique client_order_id (UUID4).
State transitions emit audit log events with Trace IDs.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from statemachine import State, StateMachine

from src.audit.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class OrderStateMachine(StateMachine):
    """
    Explicit order lifecycle state machine.

    Prevents illegal state transitions (e.g., Filled → New)
    and ensures order tracking integrity.
    """

    # ── States ────────────────────────────────────────────────────────
    new = State(initial=True)
    pending = State()
    acknowledged = State()
    partially_filled = State()
    filled = State(final=True)
    cancelled = State(final=True)
    rejected = State(final=True)

    # ── Transitions ───────────────────────────────────────────────────
    submit = new.to(pending)
    acknowledge = pending.to(acknowledged)
    partial_fill = acknowledged.to(partially_filled) | partially_filled.to(partially_filled)
    fill = acknowledged.to(filled) | partially_filled.to(filled)
    cancel = (
        new.to(cancelled)
        | pending.to(cancelled)
        | acknowledged.to(cancelled)
        | partially_filled.to(cancelled)
    )
    reject = pending.to(rejected) | new.to(rejected)


@dataclass
class TrackedOrder:
    """
    An order tracked through its complete lifecycle.

    Wraps the state machine with metadata and audit logging.
    """

    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    broker_order_id: str = ""
    instrument_token: str = ""
    side: str = ""  # BUY or SELL
    quantity: int = 0
    filled_quantity: int = 0
    price: float = 0.0
    average_fill_price: float = 0.0
    order_type: str = ""
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sm = OrderStateMachine()

    @property
    def state(self) -> str:
        if hasattr(self._sm, "current_state_value"):
            return str(self._sm.current_state_value).lower()
        return "new"

    @property
    def is_terminal(self) -> bool:
        return self.state in ("filled", "cancelled", "rejected")

    @property
    def is_active(self) -> bool:
        return self.state not in ("filled", "cancelled", "rejected")

    def submit(self) -> None:
        self._sm.submit()

    def acknowledge(self, broker_order_id: str = "") -> None:
        if broker_order_id:
            self.broker_order_id = broker_order_id
        self._sm.acknowledge()

    def partial_fill(self, filled_qty: int, avg_price: float) -> None:
        self.filled_quantity = filled_qty
        self.average_fill_price = avg_price
        self._sm.partial_fill()

    def fill(self, filled_qty: int, avg_price: float) -> None:
        self.filled_quantity = filled_qty
        self.average_fill_price = avg_price
        self._sm.fill()

    def cancel(self) -> None:
        self._sm.cancel()

    def reject(self, reason: str = "") -> None:
        self.metadata["reject_reason"] = reason
        self._sm.reject()

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "instrument_token": self.instrument_token,
            "side": self.side,
            "quantity": self.quantity,
            "filled_quantity": self.filled_quantity,
            "price": self.price,
            "average_fill_price": self.average_fill_price,
            "state": self.state,
            "trace_id": self.trace_id,
        }


import asyncio
import json
import sqlite3
import threading
from pathlib import Path

DB_PATH = Path("data/trader.db")


class OrderTracker:
    """
    Central registry for all tracked orders.

    Ensures idempotency via client_order_id, provides lookup by both client
    and broker order IDs, and persists order state to SQLite (data/trader.db).
    Guarded by thread-safe mutex locks to prevent database lock race conditions.
    """

    def __init__(self, audit: AuditLogger, db_path: Path = DB_PATH) -> None:
        self._orders: dict[str, TrackedOrder] = {}  # client_order_id → order
        self._broker_map: dict[str, str] = {}  # broker_id → client_order_id
        self._audit = audit
        self._db_path = db_path
        self._lock = threading.Lock()
        self._init_db()
        self.load_persisted_orders()

    def _init_db(self) -> None:
        """Ensure SQLite order_states table exists."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS order_states (
                        client_order_id TEXT PRIMARY KEY,
                        broker_order_id TEXT,
                        instrument_token TEXT,
                        side TEXT,
                        quantity INTEGER,
                        filled_quantity INTEGER,
                        price REAL,
                        average_fill_price REAL,
                        order_type TEXT,
                        state TEXT,
                        trace_id TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize order_states table: %s", exc)

    def persist_order(self, order: TrackedOrder) -> None:
        """Persist or update an order state in SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO order_states (
                        client_order_id, broker_order_id, instrument_token, side,
                        quantity, filled_quantity, price, average_fill_price,
                        order_type, state, trace_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(client_order_id) DO UPDATE SET
                        broker_order_id=excluded.broker_order_id,
                        filled_quantity=excluded.filled_quantity,
                        average_fill_price=excluded.average_fill_price,
                        state=excluded.state,
                        updated_at=CURRENT_TIMESTAMP
                """, (
                    order.client_order_id, order.broker_order_id, order.instrument_token,
                    order.side, order.quantity, order.filled_quantity, order.price,
                    order.average_fill_price, order.order_type, order.state, order.trace_id
                ))
                conn.commit()
        except Exception as exc:
            logger.error("Failed to persist order %s to SQLite: %s", order.client_order_id, exc)

    def load_persisted_orders(self) -> None:
        """Recover active and historic orders from SQLite on restart."""
        try:
            if not self._db_path.exists():
                return
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM order_states")
                rows = cur.fetchall()

                for r in rows:
                    rec = dict(r)
                    order = TrackedOrder(
                        client_order_id=rec["client_order_id"],
                        broker_order_id=rec.get("broker_order_id", ""),
                        instrument_token=rec.get("instrument_token", ""),
                        side=rec.get("side", ""),
                        quantity=rec.get("quantity", 0),
                        filled_quantity=rec.get("filled_quantity", 0),
                        price=rec.get("price", 0.0),
                        average_fill_price=rec.get("average_fill_price", 0.0),
                        order_type=rec.get("order_type", "MARKET"),
                        trace_id=rec.get("trace_id", ""),
                    )
                    
                    # Advance state machine to persisted state
                    target_state = rec.get("state", "new")
                    if target_state == "pending":
                        order.submit()
                    elif target_state == "acknowledged":
                        order.submit()
                        order.acknowledge(rec.get("broker_order_id", ""))
                    elif target_state == "partially_filled":
                        order.submit()
                        order.acknowledge(rec.get("broker_order_id", ""))
                        order.partial_fill(rec.get("filled_quantity", 0), rec.get("average_fill_price", 0.0))
                    elif target_state == "filled":
                        order.submit()
                        order.acknowledge(rec.get("broker_order_id", ""))
                        order.fill(rec.get("filled_quantity", order.quantity), rec.get("average_fill_price", order.price))
                    elif target_state == "cancelled":
                        order.cancel()
                    elif target_state == "rejected":
                        order.reject("Recovered from DB")

                    self._orders[order.client_order_id] = order
                    if order.broker_order_id:
                        self._broker_map[order.broker_order_id] = order.client_order_id

                if rows:
                    logger.info("Loaded %d persisted orders from SQLite.", len(rows))
        except Exception as exc:
            logger.error("Failed to load persisted orders: %s", exc)

    def create_order(
        self,
        instrument_token: str,
        side: str,
        quantity: int,
        price: float = 0.0,
        order_type: str = "MARKET",
        client_order_id: str = "",
    ) -> TrackedOrder:
        """Create and register a new tracked order with idempotency check under lock."""
        with self._lock:
            cid = client_order_id or str(uuid.uuid4())
            if cid in self._orders:
                logger.info("Idempotent hit for client_order_id: %s", cid)
                return self._orders[cid]

            order = TrackedOrder(
                client_order_id=cid,
                instrument_token=instrument_token,
                side=side,
                quantity=quantity,
                price=price,
                order_type=order_type,
            )
            self._orders[order.client_order_id] = order
            self.persist_order(order)

            self._audit.log_event(
                event_type="ORDER_CREATED",
                trace_id=order.trace_id,
                payload=order.to_dict(),
                source="order_tracker",
            )

            return order

    def transition(self, client_order_id: str, event: str, **kwargs: Any) -> None:
        """Apply a state transition to a tracked order and persist update under lock."""
        with self._lock:
            order = self._orders.get(client_order_id)
            if order is None:
                logger.warning("Unknown order: %s", client_order_id)
                return

            old_state = order.state

            try:
                if event == "submit":
                    order.submit()
                elif event == "acknowledge":
                    order.acknowledge(kwargs.get("broker_order_id", ""))
                    if order.broker_order_id:
                        self._broker_map[order.broker_order_id] = client_order_id
                elif event == "partial_fill":
                    order.partial_fill(
                        kwargs.get("filled_qty", 0),
                        kwargs.get("avg_price", 0.0),
                    )
                elif event == "fill":
                    order.fill(
                        kwargs.get("filled_qty", order.quantity),
                        kwargs.get("avg_price", order.price),
                    )
                elif event == "cancel":
                    order.cancel()
                elif event == "reject":
                    order.reject(kwargs.get("reason", ""))
                else:
                    logger.warning("Unknown event: %s", event)
                    return

                self.persist_order(order)

                self._audit.log_event(
                    event_type="ORDER_STATE_CHANGE",
                    trace_id=order.trace_id,
                    payload={
                        "client_order_id": client_order_id,
                        "from": old_state,
                        "to": order.state,
                        "event": event,
                        **kwargs,
                    },
                    source="order_tracker",
                )

            except Exception as exc:
                logger.error(
                    "Invalid transition %s for order %s (state=%s): %s",
                    event, client_order_id, old_state, exc,
                )

    def get_by_client_id(self, client_order_id: str) -> Optional[TrackedOrder]:
        return self._orders.get(client_order_id)

    def get_by_broker_id(self, broker_order_id: str) -> Optional[TrackedOrder]:
        client_id = self._broker_map.get(broker_order_id)
        if client_id:
            return self._orders.get(client_id)
        return None

    def get_active_orders(self) -> list[TrackedOrder]:
        return [o for o in self._orders.values() if o.is_active]

    def get_all_orders(self) -> list[TrackedOrder]:
        return list(self._orders.values())
