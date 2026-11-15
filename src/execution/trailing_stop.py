import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger("TrailingStopManager")
ExitCallback = Callable[[str, str, float], Coroutine[Any, Any, None]]
DB_PATH = Path("data/trader.db")


@dataclass
class TrailingStopState:
    """State for a single trailing stop."""

    instrument_token: str
    direction: str  # LONG or SHORT
    entry_price: float
    initial_stop: float
    current_stop: float
    highest_price: float  # For LONG
    lowest_price: float   # For SHORT
    trail_mode: str       # atr | fixed | percentage
    trail_value: float    # ATR multiplier, fixed points, or percentage
    is_active: bool = True
    triggered: bool = False
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class TrailingStopManager:
    """
    Manages trailing stop-losses for all active positions.
    Supports ATR, fixed point, and percentage trailing modes with SQLite persistence.
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._stops: dict[str, TrailingStopState] = {}
        self._exit_callback: Optional[ExitCallback] = None
        self._db_path = db_path
        self._init_db()
        self.restore_stops()

    def _init_db(self) -> None:
        """Ensure trailing_stops table exists."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trailing_stops (
                        instrument_token TEXT PRIMARY KEY,
                        direction TEXT,
                        entry_price REAL,
                        initial_stop REAL,
                        current_stop REAL,
                        highest_price REAL,
                        lowest_price REAL,
                        trail_mode TEXT,
                        trail_value REAL,
                        is_active INTEGER,
                        triggered INTEGER,
                        trace_id TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize trailing_stops table: %s", exc)

    def persist_stop(self, state: TrailingStopState) -> None:
        """Persist trailing stop state to SQLite."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO trailing_stops (
                        instrument_token, direction, entry_price, initial_stop,
                        current_stop, highest_price, lowest_price, trail_mode,
                        trail_value, is_active, triggered, trace_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(instrument_token) DO UPDATE SET
                        current_stop=excluded.current_stop,
                        highest_price=excluded.highest_price,
                        lowest_price=excluded.lowest_price,
                        is_active=excluded.is_active,
                        triggered=excluded.triggered,
                        updated_at=CURRENT_TIMESTAMP
                """, (
                    state.instrument_token, state.direction, state.entry_price,
                    state.initial_stop, state.current_stop, state.highest_price,
                    state.lowest_price, state.trail_mode, state.trail_value,
                    1 if state.is_active else 0, 1 if state.triggered else 0, state.trace_id
                ))
                conn.commit()
        except Exception as exc:
            logger.error("Failed to persist trailing stop %s: %s", state.instrument_token, exc)

    def restore_stops(self) -> None:
        """Restore active trailing stops on system restart."""
        try:
            if not self._db_path.exists():
                return
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT * FROM trailing_stops WHERE is_active = 1")
                rows = cur.fetchall()

                for r in rows:
                    rec = dict(r)
                    state = TrailingStopState(
                        instrument_token=rec["instrument_token"],
                        direction=rec["direction"],
                        entry_price=rec["entry_price"],
                        initial_stop=rec["initial_stop"],
                        current_stop=rec["current_stop"],
                        highest_price=rec["highest_price"],
                        lowest_price=rec["lowest_price"],
                        trail_mode=rec["trail_mode"],
                        trail_value=rec["trail_value"],
                        is_active=bool(rec["is_active"]),
                        triggered=bool(rec["triggered"]),
                        trace_id=rec.get("trace_id", str(uuid.uuid4())[:8]),
                    )
                    self._stops[state.instrument_token] = state

                if rows:
                    logger.info("Restored %d active trailing stops from SQLite.", len(rows))
        except Exception as exc:
            logger.error("Failed to restore trailing stops: %s", exc)

    def set_exit_callback(self, callback: ExitCallback) -> None:
        """Set the callback for triggering exits."""
        self._exit_callback = callback

    def attach(
        self,
        instrument_token: str,
        direction: str,
        entry_price: float,
        initial_stop: float,
        trail_mode: str = "atr",
        trail_value: float = 1.5,
        trace_id: str = "",
    ) -> TrailingStopState:
        """Attach trailing stop management to a position."""
        tr_id = trace_id or str(uuid.uuid4())[:8]
        state = TrailingStopState(
            instrument_token=instrument_token,
            direction=direction,
            entry_price=entry_price,
            initial_stop=initial_stop,
            current_stop=initial_stop,
            highest_price=entry_price,
            lowest_price=entry_price,
            trail_mode=trail_mode,
            trail_value=trail_value,
            trace_id=tr_id,
        )
        self._stops[instrument_token] = state
        self.persist_stop(state)

        logger.info(
            "[%s] Trailing stop attached: %s %s | Entry: %.2f | Stop: %.2f | Mode: %s (%.2f)",
            tr_id, instrument_token, direction, entry_price, initial_stop, trail_mode, trail_value,
        )
        return state

    def detach(self, instrument_token: str) -> None:
        """Remove trailing stop management for an instrument."""
        if instrument_token in self._stops:
            state = self._stops[instrument_token]
            state.is_active = False
            self.persist_stop(state)
            del self._stops[instrument_token]
            logger.info("Trailing stop detached: %s", instrument_token)

    async def on_price_update(
        self,
        instrument_token: str,
        current_price: float,
        current_atr: float = 0.0,
    ) -> Optional[str]:
        """Update trailing stop with a new price. Returns 'TRIGGERED' if stop hit."""
        state = self._stops.get(instrument_token)
        if state is None or not state.is_active:
            return None

        if state.direction == "LONG":
            return await self._update_long(state, current_price, current_atr)
        else:
            return await self._update_short(state, current_price, current_atr)

    async def _update_long(
        self, state: TrailingStopState, price: float, atr: float
    ) -> Optional[str]:
        """Update trailing stop for a long position."""
        if price > state.highest_price:
            state.highest_price = price

            if state.trail_mode == "atr" and atr > 0:
                new_stop = price - (atr * state.trail_value)
            elif state.trail_mode == "percentage":
                new_stop = price * (1.0 - (state.trail_value / 100.0))
            else:
                new_stop = price - state.trail_value

            if new_stop > state.current_stop:
                old_stop = state.current_stop
                state.current_stop = round(new_stop, 2)
                self.persist_stop(state)
                logger.info(
                    "[%s] Stop tightened (LONG): %s ₹%.2f → ₹%.2f (price: ₹%.2f)",
                    state.trace_id, state.instrument_token, old_stop, state.current_stop, price,
                )

        if price <= state.current_stop:
            return await self._trigger_exit(state, price)

        return None

    async def _update_short(
        self, state: TrailingStopState, price: float, atr: float
    ) -> Optional[str]:
        """Update trailing stop for a short position."""
        if price < state.lowest_price:
            state.lowest_price = price

            if state.trail_mode == "atr" and atr > 0:
                new_stop = price + (atr * state.trail_value)
            elif state.trail_mode == "percentage":
                new_stop = price * (1.0 + (state.trail_value / 100.0))
            else:
                new_stop = price + state.trail_value

            if new_stop < state.current_stop:
                old_stop = state.current_stop
                state.current_stop = round(new_stop, 2)
                self.persist_stop(state)
                logger.info(
                    "[%s] Stop tightened (SHORT): %s ₹%.2f → ₹%.2f (price: ₹%.2f)",
                    state.trace_id, state.instrument_token, old_stop, state.current_stop, price,
                )

        if price >= state.current_stop:
            return await self._trigger_exit(state, price)

        return None

    async def _trigger_exit(
        self, state: TrailingStopState, trigger_price: float
    ) -> str:
        """
        Handle stop-loss trigger — execute exit callback FIRST before marking inactive.
        If exit callback fails, retain is_active=True to allow retry on next tick.
        """
        slippage = abs(trigger_price - state.current_stop)
        slippage_pct = (slippage / state.current_stop * 100.0) if state.current_stop > 0 else 0.0
        logger.warning(
            "[%s] 🚨 TRAILING STOP HIT: %s | Direction: %s | Stop: ₹%.2f | Price: ₹%.2f | Gap Slippage: ₹%.2f (%.2f%%)",
            state.trace_id, state.instrument_token, state.direction, state.current_stop, trigger_price, slippage, slippage_pct,
        )

        exit_success = True
        if self._exit_callback:
            try:
                exit_side = "SELL" if state.direction == "LONG" else "BUY"
                await self._exit_callback(
                    state.instrument_token, exit_side, trigger_price
                )
            except Exception as exc:
                exit_success = False
                logger.error("[%s] Exit callback failed for %s: %s (Stop remains active for retry)", state.trace_id, state.instrument_token, exc)

        if exit_success:
            state.is_active = False
            state.triggered = True
            self.persist_stop(state)
            return "TRIGGERED"

        return "RETRY_REQUIRED"

    def get_all_stops(self) -> dict[str, TrailingStopState]:
        return {k: v for k, v in self._stops.items() if v.is_active}

    def get_stop(self, instrument_token: str) -> Optional[TrailingStopState]:
        return self._stops.get(instrument_token)
