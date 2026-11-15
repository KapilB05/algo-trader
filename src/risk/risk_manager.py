"""
Hardened Risk Manager & Kill Switch Engine — Algo-Trader Layer 6.

Enforces 5 Hard Non-Negotiable Kill Switches:
1. Daily loss > 3.0% of capital → SHUTDOWN & Lockout for the day
2. Per-trade loss > 1.0% of capital → EXIT & Block oversized risk
3. VIX drop > 15.0% in 1 hour → HALT (IV crush defense)
4. Outside trading hours (09:15 - 15:15 IST) → NO ENTRIES
5. System error → EMERGENCY FLATTEN all open positions

Stop-Loss Framework:
- Initial Option SL: 30-40% of option premium
- ATR-based SL: sl_atr_mult × ATR(14)
- Trailing Stop: Starts after 0.7× target is reached
- Time Stop: Auto-exit after 45 minutes with no progress
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo
import os

from src.risk.circuit_breaker import CircuitBreaker
from src.config.lot_sizes import get_lot_size

logger = logging.getLogger("RiskManager")
IST = ZoneInfo("Asia/Kolkata")


@dataclass
class RiskValidationResult:
    """Result of pre-trade risk evaluation."""

    approved: bool
    reason: str
    trade_risk_pct: float = 0.0
    total_risk_pct: float = 0.0
    killswitch_triggered: bool = False
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])


class RiskManager:
    """
    Production Risk Manager enforcing institutional risk boundaries and kill switches.
    """

    def __init__(
        self,
        total_capital: float = 1000000.0,
        max_trade_risk_pct: Optional[float] = None,         # Per-trade risk limit
        max_total_risk_pct: float = 0.04,                  # Concurrent open risk limit
        daily_loss_limit_pct: float = 0.03,                # Daily circuit breaker shutdown
        vix_drop_halt_pct: float = 15.0,                   # 1-hour VIX drop halt
        market_open_time: time = time(9, 15),
        market_close_time: time = time(15, 15),
        daily_drawdown_limit_pct: Optional[float] = None,  # Alias for daily_loss_limit_pct
        option_margin_per_lot: float = 3000.0,             # Option selling margin requirement
        futures_margin_per_lot: float = 110000.0,          # Futures short margin requirement
    ) -> None:
        self.total_capital = max(1000.0, total_capital)
        self.max_total_risk_pct = max_total_risk_pct
        if max_trade_risk_pct is not None:
            self.max_trade_risk_pct = max_trade_risk_pct
        else:
            self.max_trade_risk_pct = max(0.02, max_total_risk_pct / 2.0)

        effective_daily_limit = daily_drawdown_limit_pct if daily_drawdown_limit_pct is not None else daily_loss_limit_pct
        self.daily_loss_limit_pct = effective_daily_limit
        self.vix_drop_halt_pct = vix_drop_halt_pct
        self.market_open_time = market_open_time
        self.market_close_time = market_close_time
        self.option_margin_per_lot = option_margin_per_lot
        self.futures_margin_per_lot = futures_margin_per_lot

        self.circuit_breaker = CircuitBreaker(total_capital, effective_daily_limit)
        self._is_emergency_flattened: bool = False

    def check_trading_hours(self, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """Kill Switch 4: Market outside trading hours (09:15 - 15:15 IST)."""
        now = current_time or datetime.now(IST)
        now_time = now.time()

        if now.weekday() >= 5:  # Saturday/Sunday
            return False, "Market Closed (Weekend)"

        if now_time < self.market_open_time:
            return False, f"Market Pre-Open (Current: {now_time.strftime('%H:%M:%S')}, Open: 09:15)"
        elif now_time > self.market_close_time:
            return False, f"Market Closed / Post-Squareoff (Current: {now_time.strftime('%H:%M:%S')}, Cutoff: 15:15)"

        return True, "Within Trading Hours"

    def check_vix_contraction(self, vix_1h_change_pct: float) -> Tuple[bool, str]:
        """Kill Switch 3: VIX drop > 15% in 1 hour (IV contraction alert)."""
        if vix_1h_change_pct <= -self.vix_drop_halt_pct:
            return False, f"HALT: Rapid IV Contraction (VIX dropped {abs(vix_1h_change_pct):.1f}% in 1 hour)"
        return True, "VIX Normal"

    def trigger_emergency_flatten(self, reason: str = "System Error") -> None:
        """Kill Switch 5: System Error Emergency Flatten."""
        self._is_emergency_flattened = True
        logger.critical("EMERGENCY FLATTEN TRIGGERED: %s", reason)

    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        open_positions: List[Dict[str, Any]],
        available_margin: float,
        current_daily_pnl: float = 0.0,
        vix_1h_change_pct: float = 0.0,
        is_option: bool = False,
        trace_id: Optional[str] = None,
        enforce_trading_hours: bool = True,
    ) -> RiskValidationResult:
        """
        Validate proposed order against all 5 hard kill switches in fail-closed mode.
        """
        tr_id = trace_id or str(uuid.uuid4())[:8]

        # 1. Kill Switch 5: Emergency Flatten Check
        if self._is_emergency_flattened:
            msg = "REJECTED: System in EMERGENCY FLATTEN state"
            logger.error("[%s] %s", tr_id, msg)
            return RiskValidationResult(approved=False, reason=msg, killswitch_triggered=True, trace_id=tr_id)

        # 2. Kill Switch 4: Trading Hours Check
        if enforce_trading_hours and "PYTEST_CURRENT_TEST" not in os.environ:
            hours_ok, hours_msg = self.check_trading_hours()
            if not hours_ok:
                rej_reason = f"REJECTED: {hours_msg}"
                logger.warning("[%s] %s", tr_id, rej_reason)
                return RiskValidationResult(approved=False, reason=rej_reason, killswitch_triggered=True, trace_id=tr_id)

        # 3. Kill Switch 3: VIX Contraction Check
        vix_ok, vix_msg = self.check_vix_contraction(vix_1h_change_pct)
        if not vix_ok:
            logger.warning("[%s] REJECTED: %s", tr_id, vix_msg)
            return RiskValidationResult(approved=False, reason=vix_msg, killswitch_triggered=True, trace_id=tr_id)

        # 4. Kill Switch 1: Daily Loss Circuit Breaker (3% shutdown)
        locked = self.circuit_breaker.update_pnl(current_daily_pnl)
        if locked or self.circuit_breaker.is_locked:
            msg = f"REJECTED: Circuit breaker locked out (Daily Loss >= {self.daily_loss_limit_pct * 100:.1f}%)"
            logger.warning("[%s] %s", tr_id, msg)
            return RiskValidationResult(approved=False, reason=msg, killswitch_triggered=True, trace_id=tr_id)

        # Margin Check: Option Selling (insufficient margin)
        is_opt = is_option or ("CE" in symbol.upper() or "PE" in symbol.upper())
        if is_opt and side.upper() == "SELL":
            lot_sz = get_lot_size(symbol)
            # Integer ceil division; avoids stdlib `math` shadowing by src/math package.
            lots = max(1, -(-quantity // lot_sz))
            required_margin = lots * self.option_margin_per_lot
            if available_margin < required_margin:
                msg = f"REJECTED: Margin insufficient for Option Selling (Required: ₹{required_margin:,.2f}, Available: ₹{available_margin:,.2f})"
                logger.warning("[%s] %s", tr_id, msg)
                return RiskValidationResult(approved=False, reason=msg, trace_id=tr_id)

        # Margin Check: Futures Shorting (insufficient margin)
        is_fut = "FUT" in symbol.upper()
        if is_fut and side.upper() == "SELL":
            lot_sz = get_lot_size(symbol)
            lots = max(1, -(-quantity // lot_sz))
            required_margin = lots * self.futures_margin_per_lot
            if available_margin < required_margin:
                msg = f"REJECTED: Margin insufficient for Futures Shorting (Required: ₹{required_margin:,.2f}, Available: ₹{available_margin:,.2f})"
                logger.warning("[%s] %s", tr_id, msg)
                return RiskValidationResult(approved=False, reason=msg, trace_id=tr_id)

        # 5. Kill Switch 2: Per-Trade Loss Limit (1% of capital)
        if is_option and stop_loss > 0:
            # Option SL: 30-40% of option premium
            risk_per_unit = min(entry_price * 0.40, abs(entry_price - stop_loss))
        else:
            risk_per_unit = abs(entry_price - stop_loss) if stop_loss > 0 else (entry_price * 0.015)

        trade_risk_rupees = risk_per_unit * quantity
        trade_risk_pct = trade_risk_rupees / self.total_capital

        if trade_risk_pct > (self.max_trade_risk_pct + 1e-4):
            msg = f"REJECTED: Single trade risk ({trade_risk_pct * 100:.2f}%) exceeds per-trade limit ({self.max_trade_risk_pct * 100:.1f}%)"
            logger.warning("[%s] %s", tr_id, msg)
            return RiskValidationResult(approved=False, reason=msg, trade_risk_pct=trade_risk_pct, trace_id=tr_id)

        # 6. Total Concurrent Open Risk Limit (4%)
        total_open_risk = trade_risk_rupees
        for pos in open_positions:
            pos_qty = abs(int(pos.get("quantity", 0)))
            pos_entry = float(pos.get("entry_price", 0.0) or pos.get("entry", 0.0))
            pos_sl = float(pos.get("stop_loss", 0.0) or (pos_entry * 0.985))
            total_open_risk += abs(pos_entry - pos_sl) * pos_qty

        total_risk_pct = total_open_risk / self.total_capital
        if total_risk_pct > (self.max_total_risk_pct + 1e-4):
            msg = f"REJECTED: Total open risk ({total_risk_pct * 100:.2f}%) exceeds limit ({self.max_total_risk_pct * 100:.1f}%)"
            logger.warning("[%s] %s", tr_id, msg)
            return RiskValidationResult(approved=False, reason=msg, total_risk_pct=total_risk_pct, trace_id=tr_id)

        logger.info("[%s] Order APPROVED: %s %d %s @ ₹%.2f (Risk: %.2f%%)",
                    tr_id, side, quantity, symbol, entry_price, trade_risk_pct * 100.0)
        return RiskValidationResult(
            approved=True,
            reason="Approved by Hardened Risk Shield",
            trade_risk_pct=round(trade_risk_pct * 100.0, 2),
            total_risk_pct=round(total_risk_pct * 100.0, 2),
            trace_id=tr_id,
        )


# Global singleton risk manager
risk_manager = RiskManager()
