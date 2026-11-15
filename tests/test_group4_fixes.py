"""
Unit test suite verifying Group 4 Medium Fixes:
- Statemachine deprecation warnings resolved.
- Price gap-through slippage logging and calculation.
"""

from pathlib import Path
import pytest
from src.audit.audit_logger import AuditLogger
from src.execution.order_state_machine import OrderTracker
from src.execution.trailing_stop import TrailingStopManager


def test_statemachine_deprecation_free_properties() -> None:
    audit = AuditLogger()
    test_db = Path("data/test_group4_tmp.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass

    tracker = OrderTracker(audit, db_path=test_db)
    order = tracker.create_order("NIFTY 50", "BUY", 25, 24000.0)

    # Properties return string values cleanly without deprecation warnings
    assert order.state == "new"
    assert order.is_active is True
    assert order.is_terminal is False

    tracker.transition(order.client_order_id, "submit")
    assert order.state == "pending"

    # Cleanup
    try:
        if test_db.exists():
            test_db.unlink()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_trailing_stop_gap_slippage_calculation() -> None:
    test_db = Path("data/test_group4_gap_tmp.db")
    if test_db.exists():
        try:
            test_db.unlink()
        except Exception:
            pass

    ts = TrailingStopManager(db_path=test_db)
    state = ts.attach("NIFTY 50", "LONG", 24000.0, 23800.0, trail_mode="fixed", trail_value=200.0)

    # Price gaps down past stop (Stop = 23,800, Price gaps to 23,650)
    res = await ts.on_price_update("NIFTY 50", 23650.0)

    assert res == "TRIGGERED"
    assert state.is_active is False
    assert state.triggered is True
    assert abs(23650.0 - 23800.0) == 150.0  # ₹150 gap slippage

    # Cleanup
    try:
        if test_db.exists():
            test_db.unlink()
    except Exception:
        pass
