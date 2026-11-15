"""
Paper Portfolio & Analytics Helper Module — Institutional Paper Trading Engine.

Handles:
1. Loading paper trading records from CSV (`data/paper_trading_log.csv`)
2. Calculating Realized P&L, Win Rate, Profit Factor, and Equity Drawdown
3. Extracting and tracking Open Positions with mark-to-market live valuation
4. Formatting Trade History with paginated / sliced limits
"""

from __future__ import annotations

import logging
from math import isinf, isnan  # type: ignore[attr-defined]
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger("PaperPortfolio")

DEFAULT_CSV_PATH = Path("data/paper_trading_log.csv")
DEFAULT_CAPITAL = 1000000.0  # ₹10,00,000 Institutional Baseline

COLUMNS = [
    "trade_id", "symbol", "direction", "entry_time", "entry_price",
    "exit_time", "exit_price", "quantity", "gross_pnl", "brokerage_stt",
    "net_pnl", "return_pct", "exit_reason", "strategies_triggered", "regime"
]


def _clean_num(val: Any, default: float = 0.0) -> float:
    """Sanitize float to guarantee valid JSON serialization (no NaN or Inf)."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (isnan(f) or isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _clean_str(val: Any, default: str = "") -> str:
    """Sanitize string to guarantee valid readable output (no 'nan' or 'None')."""
    if val is None:
        return default
    if isinstance(val, float) and isnan(val):
        return default
    s = str(val).strip()
    if s.lower() in ("nan", "none", "null", ""):
        return default
    return s


def classify_instrument(symbol: str) -> str:
    """Classify instrument into 'OPTIONS' or 'FUTURES'."""
    if not symbol:
        return "FUTURES"
    s = symbol.upper().strip()
    if s.endswith("CE") or s.endswith("PE") or " CE" in s or " PE" in s or "CALL" in s or "PUT" in s:
        return "OPTIONS"
    return "FUTURES"


def load_paper_trades(csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Read paper trading logs from SQLite database (with CSV fallback).
    """
    db_path = Path("data/trader.db")
    if db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades'")
            if cur.fetchone():
                df_sql = pd.read_sql_query("SELECT * FROM trades", conn)
                conn.close()
                if not df_sql.empty:
                    rename_map = {
                        "id": "trade_id",
                        "timestamp": "entry_time",
                        "exit_timestamp": "exit_time",
                        "side": "direction",
                        "pnl_rupees": "net_pnl",
                        "pnl_pct": "return_pct",
                        "strategy": "strategies_triggered",
                    }
                    for old_c, new_c in rename_map.items():
                        if old_c in df_sql.columns and new_c not in df_sql.columns:
                            df_sql[new_c] = df_sql[old_c]
                    for col in COLUMNS:
                        if col not in df_sql.columns:
                            df_sql[col] = np.nan
                    numeric_cols = ["entry_price", "exit_price", "quantity", "gross_pnl", "brokerage_stt", "net_pnl", "return_pct"]
                    for col in numeric_cols:
                        df_sql[col] = pd.to_numeric(df_sql[col], errors="coerce")
                    return df_sql[COLUMNS]
            conn.close()
        except Exception as exc:
            logger.debug("SQLite trades read fallback: %s", exc)

    p = Path(csv_path) if csv_path else DEFAULT_CSV_PATH
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)

    try:
        df = pd.read_csv(p, encoding="utf-8")
        # Ensure standard column set
        for col in COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        # Convert numeric columns
        numeric_cols = ["entry_price", "exit_price", "quantity", "gross_pnl", "brokerage_stt", "net_pnl", "return_pct"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["symbol"] = df["symbol"].astype(str).fillna("BANKNIFTY")
        df["direction"] = df["direction"].astype(str).fillna("BUY")
        df["entry_time"] = df["entry_time"].astype(str).fillna("")
        df["exit_time"] = df["exit_time"].astype(str).fillna("")
        df["exit_reason"] = df["exit_reason"].astype(str).fillna("")
        df["strategies_triggered"] = df["strategies_triggered"].astype(str).fillna("")
        df["regime"] = df["regime"].astype(str).fillna("")

        return df
    except Exception as exc:
        logger.error("Error reading paper trading log: %s", exc)
        return pd.DataFrame(columns=COLUMNS)


def calculate_summary(df: pd.DataFrame, initial_capital: float = DEFAULT_CAPITAL) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio metrics from completed paper trades.
    """
    if df.empty:
        return {
            "total_realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "average_trade_pnl": 0.0,
            "initial_capital": initial_capital,
            "current_capital": initial_capital,
            "return_on_capital_pct": 0.0,
        }

    # Filter completed trades (exit_price is present and > 0)
    completed = df[df["exit_price"].notna() & (df["exit_price"] > 0)].copy()

    if completed.empty:
        return {
            "total_realized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "average_trade_pnl": 0.0,
            "initial_capital": initial_capital,
            "current_capital": initial_capital,
            "return_on_capital_pct": 0.0,
        }

    total_trades = len(completed)
    pnl_series = completed["net_pnl"].fillna(0.0)
    total_realized_pnl = float(pnl_series.sum())

    wins = completed[completed["net_pnl"] > 0]
    losses = completed[completed["net_pnl"] < 0]
    breakevens = completed[completed["net_pnl"] == 0]

    winning_trades = len(wins)
    losing_trades = len(losses)
    breakeven_trades = len(breakevens)

    win_rate = round((winning_trades / total_trades) * 100.0, 2) if total_trades > 0 else 0.0

    gross_profit = float(wins["net_pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(abs(losses["net_pnl"].sum())) if not losses.empty else 0.0

    if gross_loss > 0:
        profit_factor = round(gross_profit / gross_loss, 2)
    elif gross_profit > 0:
        profit_factor = 10.0
    else:
        profit_factor = 0.0

    # Drawdown calculation on cumulative equity curve
    equity_curve = initial_capital + pnl_series.cumsum()
    peak = equity_curve.cummax()
    drawdown = peak - equity_curve
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    max_dd_pct = round((max_drawdown / float(peak.max())) * 100.0, 2) if float(peak.max()) > 0 else 0.0

    avg_pnl = round(total_realized_pnl / total_trades, 2) if total_trades > 0 else 0.0
    current_cap = round(initial_capital + total_realized_pnl, 2)
    roc_pct = round((total_realized_pnl / initial_capital) * 100.0, 2)

    return {
        "total_realized_pnl": round(total_realized_pnl, 2),
        "total_trades": total_trades,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "breakeven_trades": breakeven_trades,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_pct": max_dd_pct,
        "average_trade_pnl": avg_pnl,
        "initial_capital": initial_capital,
        "current_capital": current_cap,
        "return_on_capital_pct": roc_pct,
    }


def get_open_positions(
    df: pd.DataFrame,
    current_prices_dict: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Extract active open positions (where exit_price is null or NaN)
    and compute live mark-to-market unrealized P&L using live LTPs.
    """
    prices = current_prices_dict or {}
    open_positions: List[Dict[str, Any]] = []

    # 1. Check CSV rows with null/empty/NaN exit_price
    if not df.empty:
        open_mask = df["exit_price"].isna() | (df["exit_price"] <= 0) | (df["exit_price"] == "")
        open_df = df[open_mask].copy()

        for _, row in open_df.iterrows():
            sym = str(row.get("symbol") or "BANKNIFTY")
            direction = str(row.get("direction") or "BUY").upper()
            entry_p = float(row.get("entry_price") or 0.0)
            qty = int(row.get("quantity") or 15)

            # Match live LTP: exact symbol match, or fuzzy strike match
            ltp = prices.get(sym)
            if ltp is None:
                # Try finding in prices dictionary with upper/lower variations
                for k, v in prices.items():
                    if k.upper() in sym.upper() or sym.upper() in k.upper():
                        ltp = float(v)
                        break
            if ltp is None or ltp <= 0:
                ltp = entry_p  # fallback to entry

            if direction == "BUY":
                pos_pnl = (ltp - entry_p) * qty
                ret_pct = ((ltp - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0
            else:
                pos_pnl = (entry_p - ltp) * qty
                ret_pct = ((entry_p - ltp) / entry_p * 100.0) if entry_p > 0 else 0.0

            sl_val = float(row.get("stop_loss") or (entry_p * 0.985 if direction == "BUY" else entry_p * 1.015))
            inst_type = classify_instrument(sym)
            open_positions.append({
                "trade_id": str(row.get("trade_id") or ""),
                "id": str(row.get("trade_id") or ""),
                "symbol": sym,
                "instrument_type": inst_type,
                "is_future": (inst_type == "FUTURES"),
                "is_option": (inst_type == "OPTIONS"),
                "direction": direction,
                "side": direction,
                "entry_time": str(row.get("entry_time") or ""),
                "timestamp": str(row.get("entry_time") or ""),
                "entry_price": round(entry_p, 2),
                "avg_price": round(entry_p, 2),
                "price": round(entry_p, 2),
                "quantity": qty,
                "qty": qty,
                "current_price": round(ltp, 2),
                "ltp": round(ltp, 2),
                "unrealized_pnl": round(pos_pnl, 2),
                "pnl": round(pos_pnl, 2),
                "pnl_rupees": round(pos_pnl, 2),
                "net_pnl": round(pos_pnl, 2),
                "return_pct": round(ret_pct, 2),
                "pnl_pct": round(ret_pct, 2),
                "stop_loss": round(sl_val, 2),
                "stop_price": round(sl_val, 2),
                "strategy": str(row.get("strategies_triggered") or row.get("regime") or "Ensemble ML"),
                "product": "INTRADAY",
                "status": "OPEN",
            })

    # 2. Also check data/trader.db for any open positions not yet synced to CSV
    try:
        db_p = Path("data/trader.db")
        if db_p.exists():
            import sqlite3
            conn = sqlite3.connect(db_p)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM trades WHERE status = 'OPEN' OR exit_price IS NULL OR exit_price <= 0")
            rows = [dict(r) for r in cur.fetchall()]
            conn.close()

            existing_ids = {str(p.get("trade_id") or p.get("id")) for p in open_positions}
            for r in rows:
                tid = str(r.get("trade_id") or r.get("id") or "")
                if tid in existing_ids:
                    continue
                sym = str(r.get("symbol") or "BANKNIFTY")
                direction = str(r.get("direction") or r.get("side") or "BUY").upper()
                entry_p = float(r.get("entry_price") or r.get("price") or 0.0)
                qty = int(r.get("quantity") or 15)

                ltp = prices.get(sym, entry_p)
                pos_pnl = (ltp - entry_p) * qty if direction == "BUY" else (entry_p - ltp) * qty
                ret_pct = ((ltp - entry_p) / entry_p * 100.0) if entry_p > 0 else 0.0
                sl_val = float(r.get("stop_loss") or r.get("stop_price") or (entry_p * 0.985 if direction == "BUY" else entry_p * 1.015))
                inst_type_db = classify_instrument(sym)

                open_positions.append({
                    "trade_id": tid,
                    "id": tid,
                    "symbol": sym,
                    "instrument_type": inst_type_db,
                    "is_future": (inst_type_db == "FUTURES"),
                    "is_option": (inst_type_db == "OPTIONS"),
                    "direction": direction,
                    "side": direction,
                    "entry_time": str(r.get("entry_time") or r.get("timestamp") or ""),
                    "timestamp": str(r.get("entry_time") or r.get("timestamp") or ""),
                    "entry_price": round(entry_p, 2),
                    "avg_price": round(entry_p, 2),
                    "price": round(entry_p, 2),
                    "quantity": qty,
                    "qty": qty,
                    "current_price": round(ltp, 2),
                    "ltp": round(ltp, 2),
                    "unrealized_pnl": round(pos_pnl, 2),
                    "pnl": round(pos_pnl, 2),
                    "pnl_rupees": round(pos_pnl, 2),
                    "net_pnl": round(pos_pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "pnl_pct": round(ret_pct, 2),
                    "stop_loss": round(sl_val, 2),
                    "stop_price": round(sl_val, 2),
                    "strategy": _clean_str(r.get("strategy") or r.get("regime"), default="QUANT_ENSEMBLE"),
                    "product": str(r.get("product") or "INTRADAY"),
                    "status": "OPEN",
                })
    except Exception as exc:
        logger.debug("DB open position check fallback: %s", exc)

    return open_positions


def get_trade_history(df: pd.DataFrame, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Format completed trade history as a clean JSON-serializable list ordered latest first.
    Includes SQLite fallback if CSV is empty or has fewer trades than requested.
    """
    history: List[Dict[str, Any]] = []

    if df is not None and not df.empty and "exit_price" in df.columns:
        completed = df[df["exit_price"].notna() & (df["exit_price"] > 0)].copy()
        if not completed.empty:
            # Sort latest first (by index / exit_time)
            completed = completed.iloc[::-1]
            if limit and limit > 0:
                completed = completed.head(limit)

            for _, row in completed.iterrows():
                entry_p = _clean_num(row.get("entry_price") or row.get("price"))
                exit_p = _clean_num(row.get("exit_price"))
                qty = int(_clean_num(row.get("quantity"), 15))
                net_pnl = _clean_num(row.get("net_pnl") or row.get("pnl_rupees"))
                gross_pnl = _clean_num(row.get("gross_pnl"), net_pnl)
                brokerage = _clean_num(row.get("brokerage_stt") or row.get("fees_rupees"))
                ret_pct = _clean_num(row.get("return_pct") or row.get("pnl_pct"))
                strat = _clean_str(row.get("strategies_triggered") or row.get("strategy") or row.get("regime"), default="QUANT_ENSEMBLE")
                dir_val = str(row.get("direction") or row.get("side") or "BUY").upper()
                t_id = str(row.get("trade_id") or row.get("id") or "")
                time_val = str(row.get("exit_time") or row.get("exit_timestamp") or row.get("timestamp") or "")

                sym_val = str(row.get("symbol") or "BANKNIFTY")
                inst_type_csv = classify_instrument(sym_val)
                history.append({
                    "trade_id": t_id,
                    "id": t_id,
                    "timestamp": time_val,
                    "entry_time": str(row.get("entry_time") or row.get("timestamp") or ""),
                    "exit_time": time_val,
                    "symbol": sym_val,
                    "instrument_type": inst_type_csv,
                    "is_future": (inst_type_csv == "FUTURES"),
                    "is_option": (inst_type_csv == "OPTIONS"),
                    "direction": dir_val,
                    "side": dir_val,
                    "strategy": strat,
                    "entry_price": round(entry_p, 2),
                    "avg_price": round(entry_p, 2),
                    "price": round(entry_p, 2),
                    "exit_price": round(exit_p, 2),
                    "quantity": qty,
                    "qty": qty,
                    "gross_pnl": round(gross_pnl, 2),
                    "brokerage_stt": round(brokerage, 2),
                    "pnl": round(net_pnl, 2),
                    "pnl_rupees": round(net_pnl, 2),
                    "net_pnl": round(net_pnl, 2),
                    "return_pct": round(ret_pct, 2),
                    "pnl_pct": round(ret_pct, 2),
                    "exit_reason": str(row.get("exit_reason") or "SQUAREOFF"),
                    "status": "CLOSED",
                })

    # If CSV trades were fewer than limit, supplement with SQLite trades
    if len(history) < limit:
        try:
            db_p = Path("data/trader.db")
            if db_p.exists():
                import sqlite3
                conn = sqlite3.connect(db_p)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    "SELECT * FROM trades WHERE exit_price IS NOT NULL AND exit_price > 0 ORDER BY exit_timestamp DESC LIMIT ?",
                    (limit,)
                )
                rows = [dict(r) for r in cur.fetchall()]
                conn.close()

                existing_ids = {str(h.get("trade_id") or h.get("id")) for h in history}
                for r in rows:
                    t_id = str(r.get("id") or r.get("trade_id") or "")
                    if t_id in existing_ids:
                        continue
                    en = float(r.get("entry_price") or r.get("price") or 0.0)
                    ex = float(r.get("exit_price") or 0.0)
                    dir_val = str(r.get("side") or r.get("direction") or "BUY").upper()
                    pnl = float(r.get("pnl_rupees") or r.get("net_pnl") or 0.0)
                    ret_pct = float(r.get("pnl_pct") or 0.0)
                    if not ret_pct and en > 0:
                        ret_pct = round(((ex - en) / en) * 100.0, 2) if dir_val == "BUY" else round(((en - ex) / en) * 100.0, 2)
                    time_val = str(r.get("exit_timestamp") or r.get("timestamp") or "")
                    qty = int(r.get("quantity") or 15)
                    sym_sql = str(r.get("symbol") or "BANKNIFTY FUT")
                    inst_type_sql = classify_instrument(sym_sql)

                    history.append({
                        "trade_id": t_id,
                        "id": t_id,
                        "timestamp": time_val,
                        "entry_time": str(r.get("timestamp") or r.get("entry_time") or ""),
                        "exit_time": time_val,
                        "symbol": sym_sql,
                        "instrument_type": inst_type_sql,
                        "is_future": (inst_type_sql == "FUTURES"),
                        "is_option": (inst_type_sql == "OPTIONS"),
                        "direction": dir_val,
                        "side": dir_val,
                        "strategy": _clean_str(r.get("strategy") or r.get("regime") or r.get("strategies_triggered"), default="QUANT_ENSEMBLE"),
                        "entry_price": round(en, 2),
                        "avg_price": round(en, 2),
                        "price": round(en, 2),
                        "exit_price": round(ex, 2),
                        "quantity": qty,
                        "qty": qty,
                        "gross_pnl": round(pnl, 2),
                        "brokerage_stt": float(r.get("fees_rupees") or 20.0),
                        "pnl": round(pnl, 2),
                        "pnl_rupees": round(pnl, 2),
                        "net_pnl": round(pnl, 2),
                        "return_pct": round(ret_pct, 2),
                        "pnl_pct": round(ret_pct, 2),
                        "exit_reason": str(r.get("exit_reason") or "SQUARE_OFF"),
                        "status": "CLOSED",
                    })
                    if len(history) >= limit:
                        break
        except Exception as exc:
            logger.debug("SQLite closed trades fallback: %s", exc)

    return history[:limit]
