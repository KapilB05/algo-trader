"""
Unit test suite verifying Group 3 Fixes (H1 & H2):
- H1: Futures short margin validation rule (requires ₹1,10,000 margin per lot).
- H2: OrderTracker thread-safe mutex lock preventing state transition race conditions.
"""

import threading
from pathlib import Path
import pytest
from src.audit.audit_logger import AuditLogger
from src.execution.order_state_machine import OrderTracker
from src.risk.risk_manager import RiskManager


def test_h1_futures_short_margin_check() -> None:
    risk_mgr = RiskManager(total_capital=1000000.0)

    # Insufficient margin for short futures order (₹50,000 < ₹1,10,000 required per lot)
    res_fail = risk_mgr.validate_order(
        symbol="BANKNIFTY FUT", side="SELL", quantity=15,
        entry_price=52000.0, stop_loss=52500.0,
        open_positions=[], available_margin=50000.0,
    )
    assert res_fail.approved is False
    assert "Margin insufficient for Futures Shorting" in res_fail.reason

    # Sufficient margin for short futures order (₹2,00,000 >= ₹1,10,000)
    res_ok = risk_mgr.validate_order(
        symbol="BANKNIFTY FUT", side="SELL", quantity=15,
        entry_price=52000.0, stop_loss=52500.0,
        open_positions=[], available_margin=200000.0,
    )
    assert res_ok.approved is True


def test_h2_order_tracker_thread_safety() -> None:
    test_db = Path("data/test_group3_tmp.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass

    audit = AuditLogger()
    tracker = OrderTracker(audit, db_path=test_db)
    order = tracker.create_order("NIFTY 50", "BUY", 25, 24000.0)
    tracker.transition(order.client_order_id, "submit")
    tracker.transition(order.client_order_id, "acknowledge", broker_order_id="BRK_999")

    # Concurrent transition execution across multiple threads
    errors = []

    def worker_fill():
        try:
            tracker.transition(order.client_order_id, "fill", filled_qty=25, avg_price=24000.0)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker_fill) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert order.state == "filled"

    # Safe cleanup
    try:
        if test_db.exists():
            test_db.unlink()
    except Exception:
        pass
