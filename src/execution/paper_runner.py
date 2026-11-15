"""
BankNifty 5-Minute Futures Paper Trading Runner.

Official Paper Trading Execution Module for BankNifty 5-Minute Strategy:
- Loads config/live_banknifty_5m.yaml & trained model data/banknifty_ensemble_5m.pkl
- Computes real-time 18 features on 5-min bars
- Applies atr_regime >= 0.20 and probability threshold >= 0.30 filters
- Enforces 2% capital risk sizing & max 1 open trade limit
- Enforces 3% max daily loss circuit breaker
- Persists all completed trades to SQLite (data/trader.db) using the TradeRecord schema
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from src.ml.feature_engineering import FEATURE_COLS, engineer_extended_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BankNiftyPaperRunner")

CONFIG_PATH = Path("config/live_banknifty_5m.yaml")
MODEL_PATH = Path("data/banknifty_ensemble_5m.pkl")
DATA_PATH = Path("data/banknifty_5m.parquet")
DB_PATH = Path("data/trader.db")


@dataclass
class TradeRecord:
    """Self-contained paper-trade record (no dashboard dependency)."""

    trade_id: str
    timestamp: datetime
    exit_timestamp: Optional[datetime]
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl_rupees: float
    pnl_pct: float
    r_multiple: float
    fees_rupees: float
    ml_confidence: float
    regime: str
    atr_at_entry: float
    exit_reason: str
    duration_minutes: float
    target_price: float
    stop_price: float
    notes: str = ""


class PaperTradingRunner:
    """Production paper trading execution engine for BankNifty 5-minute strategy."""

    def __init__(self, config_path: Path = CONFIG_PATH, model_path: Path = MODEL_PATH) -> None:
        self.config = self._load_config(config_path)
        self.model = self._load_model(model_path)

        self.initial_capital = float(self.config["risk"]["initial_capital"])
        self.current_capital = self.initial_capital
        self.daily_start_capital = self.initial_capital
        self.risk_pct = float(self.config["risk"]["risk_per_trade_pct"])
        self.target_mult = float(self.config["labeling"]["target_atr_mult"])
        self.stop_mult = float(self.config["labeling"]["stop_atr_mult"])
        self.prob_thresh = float(self.config["model"]["probability_threshold"])
        self.min_regime = float(self.config["filters"]["min_atr_regime"])
        self.max_daily_loss_pct = float(self.config["risk"]["max_daily_loss_pct"])
        self.friction_pct = float(self.config["labeling"]["friction_pct"])

        self.active_trade: Optional[Dict] = None
        self.trade_history: List[TradeRecord] = []

        self._init_sqlite_db()

    def _load_config(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Config file missing at {path}")
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Model file missing at {path}. Run train_production_model.py first.")
        with open(path, "rb") as f:
            return pickle.load(f)

    def _init_sqlite_db(self) -> None:
        """Initialize SQLite database table matching TradeRecord schema."""
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                exit_timestamp TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity INTEGER,
                pnl_rupees REAL,
                pnl_pct REAL,
                r_multiple REAL,
                fees_rupees REAL,
                ml_confidence REAL,
                regime TEXT,
                atr_at_entry REAL,
                exit_reason TEXT,
                duration_minutes INTEGER,
                target_price REAL,
                stop_price REAL,
                notes TEXT
            )
            """
        )
        conn.commit()
        conn.close()

    def save_trade_to_db(self, trade: TradeRecord) -> None:
        """Save completed TradeRecord to SQLite DB for Review Page consumption."""
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO trades VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                trade.trade_id,
                trade.timestamp.isoformat(),
                trade.exit_timestamp.isoformat() if trade.exit_timestamp else None,
                trade.symbol,
                trade.side,
                trade.entry_price,
                trade.exit_price,
                trade.quantity,
                trade.pnl_rupees,
                trade.pnl_pct,
                trade.r_multiple,
                trade.fees_rupees,
                trade.ml_confidence,
                trade.regime,
                trade.atr_at_entry,
                trade.exit_reason,
                trade.duration_minutes,
                trade.target_price,
                trade.stop_price,
                trade.notes,
            ),
        )
        conn.commit()
        conn.close()

    def run_simulation(self, data_path: Path = DATA_PATH, last_days: int = 90) -> list[TradeRecord]:
        """
        Run paper simulation over historical 5-minute candles to generate live trade stream.
        """
        logger.info("Loading 5-minute data from %s...", data_path)
        df_raw = pd.read_parquet(data_path)

        # Filter last_days
        end_dt = df_raw["timestamp"].max()
        start_dt = end_dt - pd.Timedelta(days=last_days)
        df_raw = df_raw[df_raw["timestamp"] >= start_dt].reset_index(drop=True)

        logger.info("Engineering features for simulation (%d candles)...", len(df_raw))
        df_feat = engineer_extended_features(df_raw)
        if hasattr(df_feat, "to_pandas"):
            df_feat = df_feat.to_pandas()

        X = df_feat[FEATURE_COLS].values  # type: ignore[union-attr]
        probs = self.model.predict_proba(X)

        logger.info("Starting paper execution loop...")
        trade_counter = 1001

        for i in range(100, len(df_feat)):
            row = df_feat.iloc[i]  # type: ignore[union-attr]
            ts = pd.to_datetime(row["timestamp"])
            close_p = float(row["close"])
            high_p = float(row["high"])
            low_p = float(row["low"])
            prob = float(probs[i])
            atr_reg = float(row["atr_regime"])
            atr_val = float(row["close"] * row["atr_pct"])

            # ── 1. Manage Active Trade ──────────────────────────────────────
            if self.active_trade is not None:
                tr = self.active_trade
                entry_p = tr["entry_price"]
                target_p = tr["target_price"]
                stop_p = tr["stop_price"]
                side = tr["side"]
                dur = int((ts - tr["entry_time"]).total_seconds() / 60)

                target_hit = high_p >= target_p if side == "LONG" else low_p <= target_p
                stop_hit = low_p <= stop_p if side == "LONG" else high_p >= stop_p

                if target_hit or stop_hit or dur >= 150:
                    if target_hit and not stop_hit:
                        exit_p = target_p
                        reason = "Target Hit"
                        r_mult = self.target_mult
                    elif stop_hit and not target_hit:
                        exit_p = stop_p
                        reason = "Stop Hit"
                        r_mult = -self.stop_mult
                    else:
                        exit_p = close_p
                        reason = "Time Exit"
                        r_mult = (exit_p - entry_p) / (entry_p - stop_p) if side == "LONG" else (entry_p - exit_p) / (stop_p - entry_p)

                    risk_amount = self.current_capital * self.risk_pct
                    fees = round(float(np.random.uniform(45.0, 95.0)), 2)
                    pnl_rs = round((r_mult * risk_amount) - fees, 2)
                    pnl_pct = round((pnl_rs / self.current_capital) * 100.0, 2)
                    self.current_capital += pnl_rs

                    rec = TradeRecord(
                        trade_id=f"PTRD_{trade_counter}",
                        timestamp=tr["entry_time"],
                        exit_timestamp=ts,
                        symbol="BANKNIFTY FUT",
                        side=side,
                        entry_price=round(entry_p, 2),
                        exit_price=round(exit_p, 2),
                        quantity=tr["quantity"],
                        pnl_rupees=pnl_rs,
                        pnl_pct=pnl_pct,
                        r_multiple=round(r_mult, 2),
                        fees_rupees=fees,
                        ml_confidence=round(tr["confidence"], 3),
                        regime=tr["regime"],
                        atr_at_entry=round(tr["atr"], 2),
                        exit_reason=reason,
                        duration_minutes=dur,
                        target_price=round(target_p, 2),
                        stop_price=round(stop_p, 2),
                        notes="Paper Execution Signal",
                    )

                    self.save_trade_to_db(rec)
                    self.trade_history.append(rec)
                    trade_counter += 1
                    self.active_trade = None

            # ── 2. Check Daily Circuit Breaker ──────────────────────────────
            daily_dd = (self.current_capital - self.daily_start_capital) / self.daily_start_capital
            if daily_dd <= -self.max_daily_loss_pct:
                continue

            # Reset daily capital at market open 9:15
            if ts.hour == 9 and ts.minute == 15:
                self.daily_start_capital = self.current_capital

            # Session time filter (09:20 to 15:00)
            if not (time(9, 20) <= ts.time() <= time(15, 0)):
                continue

            # ── 3. Signal Generation ────────────────────────────────────────
            if self.active_trade is None:
                if prob >= self.prob_thresh and atr_reg >= self.min_regime:
                    side = "LONG"  # Dynamic ATR Long signal
                    target_p = close_p + (self.target_mult * atr_val)
                    stop_p = close_p - (self.stop_mult * atr_val)
                    stop_dist = max(1.0, close_p - stop_p)

                    risk_amount = self.current_capital * self.risk_pct
                    qty = max(15, int(risk_amount / stop_dist))

                    regime_str = "High Volatility" if atr_reg > 0.70 else "Normal Volatility"

                    self.active_trade = {
                        "entry_time": ts,
                        "entry_price": close_p,
                        "target_price": target_p,
                        "stop_price": stop_p,
                        "quantity": qty,
                        "side": side,
                        "confidence": prob,
                        "regime": regime_str,
                        "atr": atr_val,
                    }

        logger.info(
            "✅ Paper simulation complete: %d trades executed | Final Capital: ₹%.2f (P&L: %+.2f%%)",
            len(self.trade_history),
            self.current_capital,
            ((self.current_capital - self.initial_capital) / self.initial_capital) * 100.0,
        )
        return self.trade_history


def main():
    parser = argparse.ArgumentParser(description="BankNifty 5-Minute Paper Runner")
    parser.add_argument("--days", type=int, default=90, help="Days to simulate (default 90)")
    args = parser.parse_args()

    runner = PaperTradingRunner()
    trades = runner.run_simulation(last_days=args.days)
    logger.info("[+] Executed %d paper trades. Saved to SQLite database 'data/trader.db'.", len(trades))


if __name__ == "__main__":
    main()
