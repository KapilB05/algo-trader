"""
Database — SQLite ORM for trades, P&L, metrics, and audit events.

Uses aiosqlite for async-compatible database operations.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data/trading.db")

# ── Schema ────────────────────────────────────────────────────────────
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,
    instrument_token TEXT NOT NULL,
    symbol TEXT,
    direction TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity INTEGER NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT,
    stop_loss REAL,
    target REAL,
    pnl REAL DEFAULT 0,
    slippage_bps REAL DEFAULT 0,
    confidence REAL DEFAULT 0,
    regime TEXT,
    strategy TEXT,
    status TEXT DEFAULT 'OPEN',
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS daily_pnl (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    total_pnl REAL DEFAULT 0,
    realized_pnl REAL DEFAULT 0,
    unrealized_pnl REAL DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count INTEGER DEFAULT 0,
    loss_count INTEGER DEFAULT 0,
    max_drawdown REAL DEFAULT 0,
    nifty_close REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS audit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    trace_id TEXT,
    event_type TEXT NOT NULL,
    source TEXT,
    payload TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument_token);
CREATE INDEX IF NOT EXISTS idx_daily_pnl_date ON daily_pnl(date);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_trace ON audit_events(trace_id);
"""


class Database:
    """
    Async SQLite database for the trading system.

    Manages trades, daily P&L, metrics, and audit events.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._path = db_path or DEFAULT_DB_PATH
        self._conn: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Connect to the database and ensure schema exists."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self._path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.commit()
        logger.info("Database connected: %s", self._path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            await self._conn.close()
            logger.info("Database closed.")

    # ── Trades ────────────────────────────────────────────────────────

    async def insert_trade(self, trade: dict[str, Any]) -> None:
        """Insert a new trade record."""
        assert self._conn
        await self._conn.execute(
            """INSERT INTO trades
            (trade_id, instrument_token, symbol, direction, entry_price,
             quantity, entry_time, stop_loss, target, confidence, regime, strategy, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                trade["trade_id"], trade["instrument_token"],
                trade.get("symbol", ""), trade["direction"],
                trade["entry_price"], trade["quantity"],
                trade["entry_time"], trade.get("stop_loss", 0),
                trade.get("target", 0), trade.get("confidence", 0),
                trade.get("regime", ""), trade.get("strategy", ""),
                trade.get("status", "OPEN"),
            ),
        )
        await self._conn.commit()

    async def update_trade_exit(
        self, trade_id: str, exit_price: float, exit_time: str,
        pnl: float, slippage_bps: float = 0, status: str = "CLOSED",
    ) -> None:
        """Update a trade with exit details."""
        assert self._conn
        await self._conn.execute(
            """UPDATE trades SET exit_price=?, exit_time=?, pnl=?,
            slippage_bps=?, status=? WHERE trade_id=?""",
            (exit_price, exit_time, pnl, slippage_bps, status, trade_id),
        )
        await self._conn.commit()

    async def get_open_trades(self) -> list[dict[str, Any]]:
        """Fetch all open trades."""
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM trades WHERE status='OPEN'"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_trade_history(
        self, days: int = 30, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Fetch recent trade history."""
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM trades WHERE entry_time >= date('now', ?) ORDER BY entry_time DESC LIMIT ?",
            (f"-{days} days", limit),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ── Daily P&L ────────────────────────────────────────────────────

    async def upsert_daily_pnl(self, record: dict[str, Any]) -> None:
        """Insert or update daily P&L record."""
        assert self._conn
        await self._conn.execute(
            """INSERT INTO daily_pnl (date, total_pnl, realized_pnl,
            unrealized_pnl, trade_count, win_count, loss_count, max_drawdown, nifty_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
            total_pnl=?, realized_pnl=?, unrealized_pnl=?,
            trade_count=?, win_count=?, loss_count=?, max_drawdown=?, nifty_close=?""",
            (
                record["date"], record["total_pnl"], record.get("realized_pnl", 0),
                record.get("unrealized_pnl", 0), record.get("trade_count", 0),
                record.get("win_count", 0), record.get("loss_count", 0),
                record.get("max_drawdown", 0), record.get("nifty_close", 0),
                record["total_pnl"], record.get("realized_pnl", 0),
                record.get("unrealized_pnl", 0), record.get("trade_count", 0),
                record.get("win_count", 0), record.get("loss_count", 0),
                record.get("max_drawdown", 0), record.get("nifty_close", 0),
            ),
        )
        await self._conn.commit()

    async def get_daily_pnl_history(self, days: int = 90) -> list[dict[str, Any]]:
        """Fetch daily P&L history."""
        assert self._conn
        cursor = await self._conn.execute(
            "SELECT * FROM daily_pnl ORDER BY date DESC LIMIT ?",
            (days,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # ── Metrics ──────────────────────────────────────────────────────

    async def insert_metric(
        self, metric_name: str, value: float, metadata: str = "{}"
    ) -> None:
        """Insert a metric data point."""
        assert self._conn
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO metrics (date, metric_name, metric_value, metadata) VALUES (?, ?, ?, ?)",
            (now, metric_name, value, metadata),
        )
        await self._conn.commit()

    # ── Audit ────────────────────────────────────────────────────────

    async def insert_audit_event(
        self, timestamp: str, trace_id: str, event_type: str,
        source: str, payload: str,
    ) -> None:
        """Insert an audit event into the database."""
        assert self._conn
        await self._conn.execute(
            """INSERT INTO audit_events (timestamp, trace_id, event_type, source, payload)
            VALUES (?, ?, ?, ?, ?)""",
            (timestamp, trace_id, event_type, source, payload),
        )
        await self._conn.commit()
