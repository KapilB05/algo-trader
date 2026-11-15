"""
Unit test suite verifying Group 2 Fix (C2):
- C2: TrailingStopManager executes exit callback FIRST before setting is_active=False.
If callback fails, state remains active for retry on subsequent price ticks.
"""

from pathlib import Path
import pytest
from src.execution.trailing_stop import TrailingStopManager


@pytest.mark.asyncio
async def test_c2_trailing_stop_failed_exit_callback_retry() -> None:
    test_db = Path("data/test_group2_tmp.db")
    if test_db.exists():
        test_db.unlink()

    ts = TrailingStopManager(db_path=test_db)

    # Mock an exit callback that raises an exception on first attempt
    attempts = {"count": 0}

    async def mock_failing_exit_callback(symbol: str, side: str, price: float) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("Broker connection timeout")
        # Second attempt succeeds

    ts.set_exit_callback(mock_failing_exit_callback)

    # Attach stop
    state = ts.attach("BANKNIFTY FUT", "LONG", 50000.0, 49000.0, trail_mode="fixed", trail_value=500.0)

    # Price hits stop @ 48900 (Attempt 1 fails)
    res1 = await ts.on_price_update("BANKNIFTY FUT", 48900.0)
    assert res1 == "RETRY_REQUIRED"
    assert state.is_active is True  # REMAINS ACTIVE FOR RETRY!
    assert state.triggered is False

    # Next tick @ 48850 (Attempt 2 succeeds)
    res2 = await ts.on_price_update("BANKNIFTY FUT", 48850.0)
    assert res2 == "TRIGGERED"
    assert state.is_active is False
    assert state.triggered is True
    assert attempts["count"] == 2

    # Clean up test DB safely
    try:
        if test_db.exists():
            test_db.unlink()
    except Exception:
        pass
