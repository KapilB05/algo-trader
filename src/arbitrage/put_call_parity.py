"""
Put-Call Parity Arbitrage Scanner — Box Spread Detector.

Mathematical Foundation:
  Put-Call Parity (PCP): C - P = S - K * e^(-r*T)
  
  Rearranged into a Box Spread:
    Buy  Call at K_low  + Sell Call at K_high  (Bull Call Spread)
    Buy  Put  at K_high + Sell Put  at K_low   (Bear Put Spread)
    
  Box Spread always pays exactly (K_high - K_low) at expiry — mathematically guaranteed.
  Today's fair value = (K_high - K_low) * e^(-r*T)

  If market_cost < fair_value → RISK-FREE PAPER PROFIT (no directional bet needed).

In paper trading mode, this engine:
  - Scans every pair of adjacent strikes in the Angel One option chain every 5 seconds
  - Detects Box Spread anomalies (after 0.5% cost buffer for slippage/brokerage)
  - Logs opportunities and executes virtual paper trades
  - Tracks all paper positions to expiry / early exit if anomaly reverts
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from src.config.lot_sizes import get_lot_size
from src.math.black_scholes import (
    BOX_TURNOVER_LEVY_PCT,
    RISK_FREE_RATE,
    box_net_edge,
    box_spread_fair_value,
    box_spread_market_cost,
    implied_volatility,
)

logger = logging.getLogger("PutCallParityScanner")
IST = ZoneInfo("Asia/Kolkata")

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Minimum NET edge (after all 4-leg costs) to qualify as a real arbitrage.
# Gross-only gates booked theoretical edges that fees consumed, so both bars
# apply to post-cost figures for 1 lot. Typical BANKNIFTY all-in costs run
# ≈ ₹140–150/lot; the bar sits just above that plus a safety margin.
MIN_NET_EDGE_ABS = 150.0   # Minimum ₹150 net edge per position (1 lot)
MIN_NET_EDGE_PCT = 0.003   # Minimum 0.3% net edge vs fair value

# Execution-realism gates (Phase 2).
PERSIST_SCANS = 3          # anomaly must survive N consecutive scans (flicker filter)
MAX_QUOTE_AGE_S = 120.0    # chain snapshot older than this is stale → no entries
MIN_DTE_ENTRY = 1.0        # no new boxes inside 1 day to expiry (pinning/exercise risk)

LEDGER_PATH = Path("data/box_trades_closed.jsonl")


@dataclass
class BoxSpreadOpportunity:
    """A detected Box Spread arbitrage opportunity."""
    detected_at: str              # ISO timestamp
    symbol: str                   # e.g. "BANKNIFTY"
    expiry: str                   # e.g. "18-SEP-2026"
    dte: float                    # Days to expiry
    strike_low: float             # Lower strike (e.g. 56000)
    strike_high: float            # Upper strike (e.g. 56500)
    spread: float                 # K_high - K_low (e.g. 500)
    fair_value: float             # PV of guaranteed payout (₹)
    market_cost: float            # Actual cost of all 4 legs (₹)
    edge: float                   # fair_value - market_cost (₹, GROSS pre-cost)
    edge_pct: float               # gross edge / fair_value (%)
    call_low_ltp: float
    call_high_ltp: float
    put_low_ltp: float
    put_high_ltp: float
    call_low_iv: float            # IV of CE at lower strike
    call_high_iv: float
    put_low_iv: float
    put_high_iv: float
    paper_trade_id: Optional[int] = None
    is_executed: bool = False
    paper_pnl: float = 0.0
    status: str = "DETECTED"      # DETECTED → EXECUTED → CLOSED
    # Post-cost economics (keyword-only in practice — always set post-init).
    edge_net_inr: float = 0.0     # post-cost edge for the paper position (₹)
    edge_net_pct: float = 0.0     # post-cost edge / fair_value (%)
    all_in_costs: float = 0.0     # 4-leg brokerage + levies + slippage (₹)
    quote_as_of: str = ""         # chain snapshot timestamp (freshness)


@dataclass
class PaperBoxTrade:
    """An active virtual Box Spread paper trade."""
    trade_id: int
    opportunity: BoxSpreadOpportunity
    entry_cost: float             # Total debit paid (₹)
    entry_time: str
    guaranteed_payout: float      # (K_high - K_low) — locked forever
    lots: int                     # Paper lots traded
    gross_pnl: float = 0.0
    status: str = "OPEN"          # OPEN → CLOSED
    exit_time: Optional[str] = None
    exit_reason: str = ""


class PutCallParityScanner:
    """
    Real-time Box Spread arbitrage scanner using Angel One option chain data.
    All trades are executed in paper mode only — no real orders.
    """

    def __init__(self, symbol: str = "BANKNIFTY", lots: int = 1) -> None:
        self.symbol = symbol.upper()
        self.lots = lots
        self._opportunities: List[BoxSpreadOpportunity] = []
        self._paper_trades: List[PaperBoxTrade] = []
        self._trade_id_counter = 1000
        self._scan_count = 0
        self._last_scan_ts = 0.0
        # Persistence registry: (strike_low, strike_high, expiry) →
        # {"consecutive": int, "last_scan": int, "opp": BoxSpreadOpportunity}
        self._pending: Dict[Any, Dict[str, Any]] = {}

        # Setup dedicated log file
        log_file = LOGS_DIR / "arb_scanner.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
        logger.addHandler(fh)
        logger.setLevel(logging.INFO)

    def _dte_from_expiry(self, expiry_str: str) -> float:
        """Compute days to expiry from string like '18-SEP-2026'."""
        try:
            exp_date = datetime.strptime(expiry_str.replace("-", "").upper(), "%d%b%Y").date()
            today = datetime.now(IST).date()
            return max(0.5, (exp_date - today).days)
        except Exception:
            return 1.0

    @staticmethod
    def _normalize_chain(chain_data: Dict[str, Any]) -> Tuple[float, str, Dict[float, Dict[str, Any]], str]:
        """Normalize provider chain payloads to the scanner's strike-dict schema.

        The live Angel provider returns ``strikes`` as a LIST of rows with
        ``ce_ltp``/``pe_ltp``/``ce_iv``/``pe_iv`` keys plus ``spot_price`` /
        ``selected_expiry`` / ``timestamp_ist`` — while older code expected a
        dict keyed by strike with CE/PE sub-dicts. Without this adapter the
        scanner silently evaluated nothing (``spot`` defaulted to 0.0 and the
        scan bailed before touching strikes). Both shapes are accepted; anything
        else yields an empty book.
        """
        if not isinstance(chain_data, dict):
            return 0.0, "", {}, ""
        spot = float(chain_data.get("spot", chain_data.get("spot_price", 0.0)) or 0.0)
        expiry = str(chain_data.get("expiry", chain_data.get("selected_expiry", "")) or "")
        as_of = str(chain_data.get("quote_as_of", chain_data.get("timestamp_ist", "")) or "")
        raw_strikes = chain_data.get("strikes", {})
        book: Dict[float, Dict[str, Any]] = {}
        if isinstance(raw_strikes, dict):
            for k, v in raw_strikes.items():
                try:
                    book[float(k)] = v if isinstance(v, dict) else {}
                except (TypeError, ValueError):
                    continue
        elif isinstance(raw_strikes, list):
            for row in raw_strikes:
                if not isinstance(row, dict):
                    continue
                try:
                    strike = float(row.get("strike", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                if strike <= 0:
                    continue
                book[strike] = {
                    "CE": {"ltp": row.get("ce_ltp", 0.0), "iv": row.get("ce_iv", 15.0)},
                    "PE": {"ltp": row.get("pe_ltp", 0.0), "iv": row.get("pe_iv", 15.0)},
                }
        return spot, expiry, book, as_of

    @staticmethod
    def _quote_age_s(as_of: str, now: Optional[datetime] = None) -> float:
        """Age of a ``HH:MM:SS IST`` chain snapshot in seconds (inf if unparseable)."""
        try:
            now = now or datetime.now(IST)
            parts = as_of.strip().split()
            t = datetime.strptime(parts[0], "%H:%M:%S").time()
            snap = now.replace(hour=t.hour, minute=t.minute, second=t.second, microsecond=0)
            age = (now - snap).total_seconds()
            return age if age >= 0 else float("inf")
        except Exception:
            return float("inf")

    def scan_once(self) -> List[BoxSpreadOpportunity]:
        """
        Fetch the current Angel One option chain and scan all adjacent strike pairs
        for Box Spread arbitrage opportunities.

        Returns a list of newly detected BoxSpreadOpportunity objects.
        """
        self._scan_count += 1
        new_opps: List[BoxSpreadOpportunity] = []

        try:
            from src.data.index_fo_data import fetch_angel_option_chain
            chain_data = fetch_angel_option_chain(self.symbol)
        except Exception as e:
            logger.warning("Option chain fetch failed: %s", e)
            return []

        if not chain_data:
            logger.debug("No option chain data returned for %s", self.symbol)
            return []

        spot, expiry, strike_data, quote_as_of = self._normalize_chain(chain_data)

        if spot <= 0 or not strike_data:
            return []

        lot_size = get_lot_size(self.symbol)
        dte = self._dte_from_expiry(expiry)
        strikes_sorted = sorted(strike_data.keys())

        # Entry gates: no new boxes into expiry-day pinning risk, and no entries
        # on stale snapshots (a minutes-old LTP prints phantom edges).
        allow_entries = dte >= MIN_DTE_ENTRY and self._quote_age_s(quote_as_of) <= MAX_QUOTE_AGE_S
        if dte < MIN_DTE_ENTRY:
            logger.debug("Box entries paused: DTE %.2f < %.1f (expiry-day risk)", dte, MIN_DTE_ENTRY)

        # Scan each adjacent strike pair (K_low, K_high)
        for i in range(len(strikes_sorted) - 1):
            k_low = strikes_sorted[i]
            k_high = strikes_sorted[i + 1]

            def _side(node: Any, side: str) -> Dict[str, Any]:
                return node.get(side, {}) if isinstance(node, dict) else {}

            ce_low, ce_high = _side(strike_data[k_low], "CE"), _side(strike_data[k_high], "CE")
            pe_low, pe_high = _side(strike_data[k_low], "PE"), _side(strike_data[k_high], "PE")

            # Need all 4 legs with valid prices
            c_l = float((ce_low.get("ltp", 0.0) if isinstance(ce_low, dict) else 0.0) or 0.0)
            c_h = float((ce_high.get("ltp", 0.0) if isinstance(ce_high, dict) else 0.0) or 0.0)
            p_l = float((pe_low.get("ltp", 0.0) if isinstance(pe_low, dict) else 0.0) or 0.0)
            p_h = float((pe_high.get("ltp", 0.0) if isinstance(pe_high, dict) else 0.0) or 0.0)

            if any(x <= 0 for x in [c_l, c_h, p_l, p_h]):
                continue

            fair_val = box_spread_fair_value(k_low, k_high, dte, RISK_FREE_RATE)
            market_cost = box_spread_market_cost(c_l, c_h, p_l, p_h)

            edge = fair_val - market_cost
            edge_pct = edge / fair_val if fair_val > 0 else 0.0

            # Net-of-cost gate: gross edges routinely evaporate in 4-leg fees.
            net_inr, costs = box_net_edge(
                market_cost, fair_val, lot_size, self.lots, (c_l, c_h, p_l, p_h))
            net_pct = net_inr / (fair_val * lot_size * self.lots) if fair_val > 0 else 0.0

            if net_inr >= MIN_NET_EDGE_ABS and net_pct >= MIN_NET_EDGE_PCT and allow_entries:
                # Extract IVs (use stored or recompute)
                cl_iv = float(ce_low.get("iv", 15.0) or 15.0) / 100.0
                ch_iv = float(ce_high.get("iv", 15.0) or 15.0) / 100.0
                pl_iv = float(pe_low.get("iv", 15.0) or 15.0) / 100.0
                ph_iv = float(pe_high.get("iv", 15.0) or 15.0) / 100.0

                opp = BoxSpreadOpportunity(
                    detected_at=datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
                    symbol=self.symbol,
                    expiry=expiry,
                    dte=dte,
                    strike_low=k_low,
                    strike_high=k_high,
                    spread=k_high - k_low,
                    fair_value=round(fair_val, 2),
                    market_cost=round(market_cost, 2),
                    edge=round(edge, 2),
                    edge_pct=round(edge_pct * 100.0, 3),
                    call_low_ltp=c_l,
                    call_high_ltp=c_h,
                    put_low_ltp=p_l,
                    put_high_ltp=p_h,
                    call_low_iv=round(cl_iv * 100.0, 1),
                    call_high_iv=round(ch_iv * 100.0, 1),
                    put_low_iv=round(pl_iv * 100.0, 1),
                    put_high_iv=round(ph_iv * 100.0, 1),
                    edge_net_inr=round(net_inr, 2),
                    edge_net_pct=round(net_pct * 100.0, 3),
                    all_in_costs=round(costs.total, 2),
                    quote_as_of=quote_as_of,
                )
                new_opps.append(opp)
                logger.info(
                    "📦 BOX SPREAD ANOMALY: %s | Strikes %s–%s | DTE %.1f | "
                    "Cost ₹%.2f < FairVal ₹%.2f | GROSS ₹%.2f | COSTS ₹%.2f | "
                    "NET ₹%.2f (%.3f%%)",
                    self.symbol, int(k_low), int(k_high), dte, market_cost, fair_val,
                    edge, costs.total, net_inr, net_pct * 100.0
                )

        if new_opps:
            confirmed = self._apply_persistence_filter(new_opps)
            if confirmed:
                self._execute_paper_trades(confirmed)
            self._opportunities.extend(new_opps)
        else:
            self._prune_pending()
        if not new_opps and self._scan_count % 12 == 0:
            logger.info("[Box Scan] %s | Scan #%d | No anomalies found (markets pricing fairly). Spot ≈ ₹%.2f, DTE %.1f",
                        self.symbol, self._scan_count, spot, dte)

        return new_opps

    def _apply_persistence_filter(
        self, candidates: List[BoxSpreadOpportunity]
    ) -> List[BoxSpreadOpportunity]:
        """Execute only anomalies surviving N consecutive scans (flicker filter).

        A single bad tick (stale leg LTP, partial chain refresh) prints a fake
        edge for exactly one scan. Tracking consecutive sightings per
        (strikes, expiry) key means only persistent mispricings become trades.
        """
        seen: set = set()
        confirmed: List[BoxSpreadOpportunity] = []
        for opp in candidates:
            key = (opp.strike_low, opp.strike_high, opp.expiry)
            seen.add(key)
            entry = self._pending.get(key)
            streak = (entry["consecutive"] + 1) if entry else 1
            self._pending[key] = {"consecutive": streak, "last_scan": self._scan_count, "opp": opp}
            if streak >= PERSIST_SCANS:
                confirmed.append(opp)
        self._prune_pending(seen)
        return confirmed

    def _prune_pending(self, seen: Optional[set] = None) -> None:
        """Drop persistence streaks not reconfirmed on the latest scan."""
        seen = seen or set()
        for key in [k for k, v in self._pending.items()
                    if k not in seen or v["last_scan"] < self._scan_count]:
            self._pending.pop(key, None)

    def _execute_paper_trades(self, opps: List[BoxSpreadOpportunity]) -> None:
        """Record virtual paper execution for each Box Spread opportunity."""
        for opp in opps:
            trade = PaperBoxTrade(
                trade_id=self._trade_id_counter,
                opportunity=opp,
                entry_cost=opp.market_cost,
                entry_time=opp.detected_at,
                guaranteed_payout=opp.fair_value,
                lots=self.lots,
                gross_pnl=opp.edge_net_inr,  # booked NET of all-in costs
            )
            self._trade_id_counter += 1
            self._paper_trades.append(trade)
            opp.paper_trade_id = trade.trade_id
            opp.is_executed = True
            opp.status = "EXECUTED"
            opp.paper_pnl = opp.edge_net_inr

            logger.info(
                "✅ PAPER BOX TRADE EXECUTED | ID=%d | %s %d–%d | Entry Cost ₹%.2f | Guaranteed Payout ₹%.2f | "
                "Costs ₹%.2f | Net Paper P&L ₹%+.2f (on %d lots)",
                trade.trade_id, self.symbol, int(opp.strike_low), int(opp.strike_high),
                opp.market_cost, opp.fair_value, opp.all_in_costs, opp.edge_net_inr, self.lots
            )

    def mark_to_market(self, chain_data: Optional[Dict[str, Any]] = None) -> None:
        """Reprice open box trades; settle at expiry.

        Interim value of liquidating now = (fair_value − current_box_cost) ×
        units − entry costs − estimated exit costs (informational only).
        Positions are NEVER scratched on interim noise: a box held to expiry
        pays exactly (K_high − K_low) — early exits only pay the spread twice
        and would manufacture losses. The single close path is
        EXPIRY_SETTLEMENT (DTE < 1): realized = spread × units − entry outlay
        − entry costs − settlement levy. Closes append to
        ``data/box_trades_closed.jsonl`` (git-ignored).
        """
        book: Dict[float, Dict[str, Any]] = {}
        if isinstance(chain_data, dict):
            _, _, book, _ = self._normalize_chain(chain_data)
        lot_size = get_lot_size(self.symbol)

        for trade in self._paper_trades:
            if trade.status == "CLOSED":
                continue
            opp = trade.opportunity
            units = float(lot_size * trade.lots)

            if book:
                def _ltp(node: Any, side: str) -> float:
                    try:
                        return float((node.get(side, {}) or {}).get("ltp", 0.0) or 0.0)
                    except (TypeError, ValueError, AttributeError):
                        return 0.0

                node_lo = book.get(opp.strike_low, {})
                node_hi = book.get(opp.strike_high, {})
                cur_legs = (_ltp(node_lo, "CE"), _ltp(node_hi, "CE"),
                            _ltp(node_lo, "PE"), _ltp(node_hi, "PE"))
                if all(x > 0 for x in cur_legs):
                    cur = box_spread_market_cost(*cur_legs)
                    _, exit_costs = box_net_edge(
                        cur, cur, lot_size, trade.lots, leg_ltps=cur_legs)
                    trade.gross_pnl = round(
                        (opp.fair_value - cur) * units - opp.all_in_costs - exit_costs.total, 2)

            if self._dte_from_expiry(opp.expiry) < MIN_DTE_ENTRY:
                payout = opp.spread * units
                settle_levy = round(payout * BOX_TURNOVER_LEVY_PCT, 2)
                trade.gross_pnl = round(
                    payout - opp.market_cost * units - opp.all_in_costs - settle_levy, 2)
                self._close_trade(trade, "EXPIRY_SETTLEMENT")

    def _close_trade(self, trade: PaperBoxTrade, reason: str) -> None:
        """Close a paper box trade and append it to the realized ledger."""
        trade.status = "CLOSED"
        trade.exit_time = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
        trade.exit_reason = reason
        opp = trade.opportunity
        opp.status = "CLOSED"
        record = {
            "trade_id": trade.trade_id,
            "symbol": opp.symbol,
            "strikes": f"{int(opp.strike_low)}-{int(opp.strike_high)}",
            "expiry": opp.expiry,
            "entry_time": trade.entry_time,
            "exit_time": trade.exit_time,
            "exit_reason": reason,
            "lots": trade.lots,
            "entry_cost_per_unit": opp.market_cost,
            "all_in_costs": opp.all_in_costs,
            "realized_pnl_inr": round(trade.gross_pnl, 2),
        }
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LEDGER_PATH, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning("Box ledger append failed: %s", exc)
        logger.info("📦 BOX TRADE CLOSED | ID=%d | %s | Realized ₹%+.2f",
                    trade.trade_id, reason, trade.gross_pnl)

    def get_summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary for the dashboard API."""
        total_detected = len(self._opportunities)
        open_trades = [t for t in self._paper_trades if t.status == "OPEN"]
        closed_trades = [t for t in self._paper_trades if t.status == "CLOSED"]
        gross_pnl = sum(t.gross_pnl for t in self._paper_trades)
        realized_pnl = sum(t.gross_pnl for t in closed_trades)
        open_unrealized = sum(t.gross_pnl for t in open_trades)

        recent_opps = []
        for opp in self._opportunities[-10:]:
            recent_opps.append({
                "detected_at": opp.detected_at,
                "symbol": opp.symbol,
                "strikes": f"{int(opp.strike_low)}-{int(opp.strike_high)}",
                "expiry": opp.expiry,
                "dte": opp.dte,
                "market_cost": opp.market_cost,
                "fair_value": opp.fair_value,
                "edge_inr": opp.edge,
                "edge_pct": opp.edge_pct,
                "edge_net_inr": opp.edge_net_inr,
                "edge_net_pct": opp.edge_net_pct,
                "all_in_costs": opp.all_in_costs,
                "quote_as_of": opp.quote_as_of,
                "status": opp.status,
                "paper_trade_id": opp.paper_trade_id,
            })

        return {
            "scanner": "PUT_CALL_PARITY_BOX_SPREAD",
            "symbol": self.symbol,
            "total_anomalies_detected": total_detected,
            "open_paper_trades": len(open_trades),
            "closed_paper_trades": len(closed_trades),
            "gross_paper_pnl_inr": round(gross_pnl, 2),
            "realized_paper_pnl_inr": round(realized_pnl, 2),
            "open_unrealized_pnl_inr": round(open_unrealized, 2),
            "scan_count": self._scan_count,
            "recent_opportunities": recent_opps,
        }
