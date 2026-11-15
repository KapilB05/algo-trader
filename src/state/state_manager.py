"""
State Manager — Broker-state recovery on restart.

On startup: queries portfolio positions, re-attaches trailing stops,
and reconciles local DB state with broker state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from zoneinfo import ZoneInfo

from src.broker.portfolio_api import PortfolioAPI, Position
from src.execution.trailing_stop import TrailingStopManager
from src.state.db import Database

logger = logging.getLogger(__name__)


class StateManager:
    """
    Recovers system state from the broker after a restart.

    Ensures that surviving positions have trailing stops re-attached
    and the local database reflects the actual broker state.
    """

    def __init__(
        self,
        portfolio_api: PortfolioAPI,
        trailing_stop_mgr: TrailingStopManager,
        db: Database,
    ) -> None:
        self._portfolio = portfolio_api
        self._trailing = trailing_stop_mgr
        self._db = db

    async def recover(self) -> dict:
        """
        Execute state recovery sequence.

        Returns a summary of recovered state.
        """
        logger.info("Starting broker-state recovery...")

        # 1. Fetch live positions from broker.
        # On failure, ABORT recovery — never treat a failed query as
        # "no positions" (that would mark live trades as stale/closed).
        try:
            live_positions = await self._portfolio.get_positions()
        except Exception as exc:
            logger.error(
                "State recovery aborted: could not fetch broker positions (%s). "
                "Local DB state left untouched — re-run recovery once the "
                "broker connection is healthy.",
                exc,
            )
            return {
                "live_positions": -1,
                "recovered": 0,
                "new_untracked": 0,
                "stale_closed": 0,
                "error": str(exc),
            }
        open_positions = [p for p in live_positions if p.net_qty != 0]

        # 2. Fetch local DB open trades
        db_open = await self._db.get_open_trades()
        db_tokens = {t["instrument_token"] for t in db_open}

        recovered = 0
        new_found = 0

        for pos in open_positions:
            if pos.instrument_token in db_tokens:
                # Known position — re-attach trailing stop
                direction = "LONG" if pos.net_qty > 0 else "SHORT"
                self._trailing.attach(
                    instrument_token=pos.instrument_token,
                    direction=direction,
                    entry_price=pos.average_price,
                    initial_stop=self._estimate_stop(pos),
                )
                recovered += 1
                logger.info(
                    "Recovered: %s %s @ %.2f (qty=%d)",
                    pos.symbol, direction, pos.average_price, pos.net_qty,
                )
            else:
                # Unknown position — found on broker but not in local DB
                new_found += 1
                logger.warning(
                    "Untracked position found: %s qty=%d @ %.2f",
                    pos.symbol, pos.net_qty, pos.average_price,
                )

        # 3. Check for local trades that no longer exist on broker
        live_tokens = {p.instrument_token for p in open_positions}
        stale = [t for t in db_open if t["instrument_token"] not in live_tokens]
        for trade in stale:
            logger.warning(
                "Stale local trade: %s (trade_id=%s) — marking as EXTERNALLY_CLOSED.",
                trade["instrument_token"], trade["trade_id"],
            )
            # Mark as closed in DB without corrupting exit price or leaving exit_time blank
            now_ist = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
            await self._db.update_trade_exit(
                trade_id=trade["trade_id"],
                exit_price=float(trade.get("entry_price", 0.0) or 0.0),
                exit_time=now_ist,
                pnl=0.0,
                status="EXTERNALLY_CLOSED",
            )

        summary = {
            "live_positions": len(open_positions),
            "recovered": recovered,
            "new_untracked": new_found,
            "stale_closed": len(stale),
        }

        logger.info("State recovery complete: %s", summary)
        return summary

    @staticmethod
    def _estimate_stop(position: Position) -> float:
        """Estimate a conservative stop for a recovered position."""
        # Use 2% of average price as default stop distance
        stop_distance = position.average_price * 0.02
        if position.net_qty > 0:  # Long
            return position.average_price - stop_distance
        else:  # Short
            return position.average_price + stop_distance
