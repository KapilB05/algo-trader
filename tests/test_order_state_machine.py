"""Tests for the Order State Machine — valid and invalid transitions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.audit.audit_logger import AuditLogger
from src.execution.order_state_machine import (
    OrderStateMachine,
    OrderTracker,
    TrackedOrder,
)


class TestOrderStateMachine:
    """Test the python-statemachine order lifecycle."""

    def test_initial_state(self):
        """Orders start in the 'new' state."""
        sm = OrderStateMachine()
        assert sm.current_state.id == "new"

    def test_valid_lifecycle(self):
        """Test the happy path: new → pending → ack → filled."""
        sm = OrderStateMachine()
        sm.submit()
        assert sm.current_state.id == "pending"
        sm.acknowledge()
        assert sm.current_state.id == "acknowledged"
        sm.fill()
        assert sm.current_state.id == "filled"

    def test_cancel_from_pending(self):
        """Orders can be cancelled from pending state."""
        sm = OrderStateMachine()
        sm.submit()
        sm.cancel()
        assert sm.current_state.id == "cancelled"

    def test_cancel_from_acknowledged(self):
        """Orders can be cancelled after acknowledgement."""
        sm = OrderStateMachine()
        sm.submit()
        sm.acknowledge()
        sm.cancel()
        assert sm.current_state.id == "cancelled"

    def test_partial_fill(self):
        """Orders can be partially filled."""
        sm = OrderStateMachine()
        sm.submit()
        sm.acknowledge()
        sm.partial_fill()
        assert sm.current_state.id == "partially_filled"

    def test_partial_to_full_fill(self):
        """Partially filled orders can be fully filled."""
        sm = OrderStateMachine()
        sm.submit()
        sm.acknowledge()
        sm.partial_fill()
        sm.fill()
        assert sm.current_state.id == "filled"

    def test_reject_from_pending(self):
        """Orders can be rejected from pending state."""
        sm = OrderStateMachine()
        sm.submit()
        sm.reject()
        assert sm.current_state.id == "rejected"

    def test_invalid_transition_filled_to_new(self):
        """Cannot transition from filled back to new."""
        sm = OrderStateMachine()
        sm.submit()
        sm.acknowledge()
        sm.fill()
        # filled is a final state — no transitions should be valid
        with pytest.raises(Exception):
            sm.submit()

    def test_invalid_transition_cancelled_to_filled(self):
        """Cannot fill a cancelled order."""
        sm = OrderStateMachine()
        sm.submit()
        sm.cancel()
        with pytest.raises(Exception):
            sm.fill()


class TestTrackedOrder:
    """Test the TrackedOrder wrapper."""

    def test_unique_ids(self):
        """Each order gets a unique client_order_id and trace_id."""
        o1 = TrackedOrder(instrument_token="TEST|001", side="BUY", quantity=10)
        o2 = TrackedOrder(instrument_token="TEST|001", side="BUY", quantity=10)
        assert o1.client_order_id != o2.client_order_id
        assert o1.trace_id != o2.trace_id

    def test_state_tracking(self):
        """TrackedOrder tracks state transitions."""
        order = TrackedOrder(instrument_token="TEST|001", side="BUY", quantity=10)
        assert order.state == "new"
        assert order.is_active

        order.submit()
        assert order.state == "pending"

        order.acknowledge("BROKER-123")
        assert order.state == "acknowledged"
        assert order.broker_order_id == "BROKER-123"

        order.fill(10, 100.50)
        assert order.state == "filled"
        assert order.is_terminal
        assert not order.is_active
        assert order.filled_quantity == 10
        assert order.average_fill_price == 100.50

    def test_to_dict(self):
        """Serialization to dict."""
        order = TrackedOrder(instrument_token="TEST|001", side="BUY", quantity=10)
        d = order.to_dict()
        assert d["instrument_token"] == "TEST|001"
        assert d["side"] == "BUY"
        assert d["quantity"] == 10
        assert d["state"] == "new"


class TestOrderTracker:
    """Test the central order registry."""

    def setup_method(self):
        import sqlite3
        from pathlib import Path
        self.audit = MagicMock(spec=AuditLogger)
        test_db = Path("data/test_tmp_trader.db")
        if test_db.exists():
            try:
                with sqlite3.connect(test_db) as conn:
                    conn.execute("DELETE FROM order_states")
                    conn.commit()
            except Exception:
                pass
        self.tracker = OrderTracker(self.audit, db_path=test_db)
        self.tracker._orders.clear()
        self.tracker._broker_map.clear()

    def test_create_and_lookup(self):
        """Create an order and look it up."""
        order = self.tracker.create_order("TEST|001", "BUY", 10, 100.0)
        found = self.tracker.get_by_client_id(order.client_order_id)
        assert found is not None
        assert found.instrument_token == "TEST|001"

    def test_transition_tracking(self):
        """Apply transitions via the tracker."""
        order = self.tracker.create_order("TEST|001", "BUY", 10)
        self.tracker.transition(order.client_order_id, "submit")
        assert order.state == "pending"
        self.tracker.transition(
            order.client_order_id, "acknowledge", broker_order_id="BRK-001"
        )
        assert order.state == "acknowledged"
        assert order.broker_order_id == "BRK-001"

    def test_broker_id_lookup(self):
        """Look up orders by broker ID after acknowledgement."""
        order = self.tracker.create_order("TEST|001", "BUY", 10)
        self.tracker.transition(order.client_order_id, "submit")
        self.tracker.transition(
            order.client_order_id, "acknowledge", broker_order_id="BRK-002"
        )
        found = self.tracker.get_by_broker_id("BRK-002")
        assert found is not None
        assert found.client_order_id == order.client_order_id

    def test_active_orders(self):
        """Get only active (non-terminal) orders."""
        o1 = self.tracker.create_order("TEST|001", "BUY", 10)
        o2 = self.tracker.create_order("TEST|002", "SELL", 20)

        self.tracker.transition(o1.client_order_id, "submit")
        self.tracker.transition(o2.client_order_id, "submit")
        self.tracker.transition(o2.client_order_id, "cancel")

        active = self.tracker.get_active_orders()
        assert len(active) == 1
        assert active[0].client_order_id == o1.client_order_id
