import pytest
import pandas as pd
from pathlib import Path
from src.utils.paper_portfolio import get_open_positions, get_trade_history, calculate_summary

def test_paper_portfolio_get_open_positions_aliases():
    """Verify get_open_positions generates full dual-aliases without undefined fields."""
    sample_df = pd.DataFrame([{
        "trade_id": "TEST_101",
        "symbol": "BANKNIFTY 56000 CE",
        "direction": "BUY",
        "entry_time": "2026-09-14 09:30:00",
        "entry_price": 250.0,
        "quantity": 15,
        "exit_price": None,
        "strategies_triggered": "ORB+SUPERTREND",
        "regime": "BULLISH",
    }])

    prices = {"BANKNIFTY 56000 CE": 270.0}
    positions = get_open_positions(sample_df, prices)
    assert len(positions) >= 1
    pos = positions[0]

    # Verify direction and side are both present and match
    assert pos["direction"] == "BUY"
    assert pos["side"] == "BUY"
    # Verify quantity and qty
    assert pos["quantity"] == 15
    assert pos["qty"] == 15
    # Verify price aliases
    assert pos["entry_price"] == 250.0
    assert pos["avg_price"] == 250.0
    assert pos["price"] == 250.0
    assert pos["current_price"] == 270.0
    assert pos["ltp"] == 270.0
    # Verify PnL aliases
    assert pos["unrealized_pnl"] == 300.0  # (270 - 250) * 15
    assert pos["pnl"] == 300.0
    assert pos["pnl_rupees"] == 300.0
    assert pos["net_pnl"] == 300.0
    # Verify Return % aliases
    assert pos["return_pct"] == 8.0
    assert pos["pnl_pct"] == 8.0
    # Verify Stop Loss aliases
    assert "stop_loss" in pos and pos["stop_loss"] > 0
    assert "stop_price" in pos and pos["stop_price"] > 0
    # Verify ID
    assert pos["trade_id"] == "TEST_101"
    assert pos["id"] == "TEST_101"


def test_paper_portfolio_get_trade_history_aliases():
    """Verify get_trade_history returns both side and direction, pnl and net_pnl, etc."""
    sample_df = pd.DataFrame([{
        "trade_id": "TEST_202",
        "symbol": "BANKNIFTY 56600 CE",
        "direction": "BUY",
        "entry_time": "2026-09-14 09:15:00",
        "entry_price": 312.9,
        "exit_time": "2026-09-14 15:15:01",
        "exit_price": 306.6,
        "quantity": 90,
        "gross_pnl": -567.0,
        "brokerage_stt": 59.05,
        "net_pnl": -626.05,
        "return_pct": -2.01,
        "exit_reason": "EOD_1515_SQUAREOFF",
        "strategies_triggered": "ORB+SUPERTREND",
        "regime": "NORMAL",
    }])

    history = get_trade_history(sample_df, limit=10)
    assert len(history) >= 1
    t = history[0]

    # Exactly the fields that were undefined for the user:
    assert t["side"] == "BUY"
    assert t["direction"] == "BUY"
    assert t["quantity"] == 90
    assert t["qty"] == 90
    assert t["net_pnl"] == -626.05
    assert t["pnl"] == -626.05
    assert t["return_pct"] == -2.01
    assert t["pnl_pct"] == -2.01
    assert t["entry_price"] == 312.9
    assert t["avg_price"] == 312.9
    assert t["exit_price"] == 306.6
    assert t["trade_id"] == "TEST_202"
    assert t["id"] == "TEST_202"
    assert t["strategy"] == "ORB+SUPERTREND"
    assert t["exit_reason"] == "EOD_1515_SQUAREOFF"
