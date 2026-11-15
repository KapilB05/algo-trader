"""
Institutional Backtest Engine — Algo-Trader 7-Layer Architecture.

High-Performance Quantitative Options/Futures Simulation:
1. Signal Crossovers: Clean state-transition entries (no multi-bar churn)
2. Layer 3: Regime Classifier Playbook (CHOP Cash Gating)
3. Layer 4: 4-Strategy Ensemble Consensus Voting
4. Layer 5: Jane Street Anti-Manipulation Defense (Anti-Spoofing & 14:30 Expiry Cutoff)
5. Layer 6: Kelly Position Sizing (1% hard risk per trade), 3% Daily Circuit Breaker, 1:2 R:R
6. Realistic NSE F&O Cost Model: ₹20/order flat (₹40 round-trip) + Exchange Turnover/STT fees
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.extended_features import calculate_adx_dmi, calculate_supertrend
from src.risk.capital_allocator import CapitalAllocator

logger = logging.getLogger("BacktestEngine")


@dataclass
class BacktestTrade:
    """Detailed record of a backtested trade."""

    trade_id: int
    symbol: str
    direction: str          # LONG / SHORT
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    quantity: int
    gross_pnl: float
    brokerage_and_stt: float
    net_pnl: float
    return_pct: float
    exit_reason: str        # TARGET / STOP_LOSS / TRAILING_STOP / TIME_STOP / EXPIRY_1430 / EOD_1515
    holding_bars: int
    regime_at_entry: str
    consensus_votes: int


@dataclass
class BacktestMetrics:
    """Comprehensive performance report metrics."""

    initial_capital: float
    final_capital: float
    total_net_pnl: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_rupees: float
    max_drawdown_pct: float
    sharpe_ratio: float
    avg_trade_pnl: float
    avg_win_pnl: float
    avg_loss_pnl: float
    win_loss_ratio: float
    max_consecutive_losses: int
    total_brokerage_paid: float
    monthly_pnl: Dict[str, float]
    daily_equity_curve: List[Dict[str, Any]]
    regime_breakdown: Dict[str, int]


class BacktestEngine:
    """
    Institutional bar-by-bar options/futures backtesting engine.
    """

    def __init__(
        self,
        initial_capital: float = 1000000.0,
        brokerage_per_order: float = 20.0,     # ₹20 flat per order (₹40 round-trip)
        stt_and_turnover_pct: float = 0.0003,   # 0.03% on option premium / turnover
        slippage_points: float = 1.5,
        lot_size: int = 30,
        max_trades_per_day: int = 2,
        use_regime_filter: bool = True,
        use_jane_street_defense: bool = True,
        use_kelly_sizing: bool = True,
    ) -> None:
        self.initial_capital = initial_capital
        self.brokerage_per_order = brokerage_per_order
        self.stt_turnover_pct = stt_and_turnover_pct
        self.slippage_points = slippage_points
        self.lot_size = lot_size
        self.max_trades_per_day = max_trades_per_day
        self.use_regime_filter = use_regime_filter
        self.use_jane_street_defense = use_jane_street_defense
        self.use_kelly_sizing = use_kelly_sizing

        self.allocator = CapitalAllocator(total_capital=initial_capital, max_risk_per_trade=0.01)

    def _precompute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Precompute technical indicators across the entire historical series."""
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        o = df["open"]
        v = df["volume"].astype(float)

        # 1. EMAs
        df["ema9"] = c.ewm(span=9, adjust=False).mean()
        df["ema20"] = c.ewm(span=20, adjust=False).mean()
        df["ema21"] = c.ewm(span=21, adjust=False).mean()
        df["ema50"] = c.ewm(span=50, adjust=False).mean()

        # 2. ATR 14
        tr1 = h - l
        tr2 = (h - c.shift(1)).abs()
        tr3 = (l - c.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr14"] = tr.rolling(window=14, min_periods=1).mean()

        # 3. Supertrend(10, 3)
        st_dir, st_band = calculate_supertrend(h, l, c, period=10, multiplier=3.0)
        df["supertrend_dir"] = st_dir
        df["supertrend_band"] = st_band

        # 4. ADX & DMI
        adx, pdi, mdi = calculate_adx_dmi(h, l, c, period=14)
        df["adx14"] = adx
        df["plus_di"] = pdi
        df["minus_di"] = mdi

        # 5. Bollinger Bands & Width Rank
        bb_mid = c.rolling(20, min_periods=1).mean()
        bb_std = c.rolling(20, min_periods=1).std(ddof=0).fillna(0.0)
        bb_width = (4.0 * bb_std) / (bb_mid + 1e-8)
        df["bb_width"] = bb_width
        bw_arr = bb_width.values
        df["bbw_rank"] = pd.Series([
            float((bw_arr[max(0, k - 50) : k + 1] <= bw_arr[k]).mean()) for k in range(len(bw_arr))
        ], index=df.index)

        # 6. Volume MA 20
        df["vol_ma20"] = v.rolling(20, min_periods=1).mean()

        # 7. Intraday Session VWAP & Opening Range (Reset each day)
        df["date"] = df["timestamp"].dt.date  # type: ignore
        df["vwap"] = 0.0
        df["orh"] = 0.0
        df["orl"] = 0.0

        for d, grp in df.groupby("date"):
            idx = grp.index
            cum_v = v.loc[idx].cumsum()
            cum_pv = (c.loc[idx] * v.loc[idx]).cumsum()
            df.loc[idx, "vwap"] = np.where(cum_v > 0, cum_pv / cum_v, c.loc[idx])

            # 15-min Opening Range (first 3 bars of session 09:15-09:30)
            first_3_h = h.loc[idx].iloc[:3].max() if len(idx) >= 3 else h.loc[idx].iloc[0]
            first_3_l = l.loc[idx].iloc[:3].min() if len(idx) >= 3 else l.loc[idx].iloc[0]
            df.loc[idx, "orh"] = first_3_h
            df.loc[idx, "orl"] = first_3_l

        return df

    def run(self, csv_path: str | Path, symbol: str = "BANKNIFTY") -> BacktestMetrics:
        """Execute full backtest over dataset."""
        df_raw = pd.read_csv(csv_path)
        if not pd.api.types.is_datetime64_any_dtype(df_raw["timestamp"]):
            df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)

        df = self._precompute_indicators(df_raw)

        current_capital = self.initial_capital
        peak_capital = self.initial_capital
        max_drawdown = 0.0
        max_drawdown_pct = 0.0

        trades: List[BacktestTrade] = []
        daily_equity: List[Dict[str, Any]] = []
        monthly_pnl: Dict[str, float] = {}
        regime_counts: Dict[str, int] = {}

        in_position = False
        pos_direction = "NONE"
        pos_entry_price = 0.0
        pos_entry_time = None
        pos_entry_idx = 0
        pos_stop_loss = 0.0
        pos_target = 0.0
        pos_qty = 0
        pos_regime = "NONE"
        pos_votes = 0

        current_day_str = ""
        daily_start_capital = current_capital
        daily_circuit_locked = False
        daily_trades_taken = 0

        start_idx = 40

        for i in range(start_idx, len(df)):
            curr = df.iloc[i]
            prev = df.iloc[i - 1]
            prev2 = df.iloc[i - 2]
            ts = curr["timestamp"]
            date_str = str(curr["date"])
            month_str = ts.strftime("%Y-%m")
            bar_time = ts.time()
            is_expiry_day = (ts.weekday() == 3)

            # Daily reset & circuit breaker tracking
            if date_str != current_day_str:
                if current_day_str:
                    day_pnl = current_capital - daily_start_capital
                    daily_equity.append({
                        "date": current_day_str,
                        "capital": round(current_capital, 2),
                        "daily_pnl": round(day_pnl, 2),
                        "daily_return_pct": round((day_pnl / daily_start_capital) * 100.0, 2),
                    })
                current_day_str = date_str
                daily_start_capital = current_capital
                daily_circuit_locked = False
                daily_trades_taken = 0

            # 3% Daily Circuit Breaker
            current_day_loss_pct = (daily_start_capital - current_capital) / daily_start_capital
            if current_day_loss_pct >= 0.03:
                daily_circuit_locked = True

            # ── 1. MANAGE ACTIVE POSITION ────────────────────────────────────
            if in_position:
                holding_bars = i - pos_entry_idx
                curr_high = float(curr["high"])
                curr_low = float(curr["low"])
                curr_close = float(curr["close"])
                exit_triggered = False
                exit_price = curr_close
                exit_reason = ""

                # A. Jane Street 14:30 Expiry Cutoff
                if self.use_jane_street_defense and is_expiry_day and bar_time >= time(14, 30):
                    exit_triggered = True
                    exit_price = curr_close
                    exit_reason = "EXPIRY_1430_CUTOFF"

                # B. Auto 15:15 EOD Squareoff
                elif bar_time >= time(15, 15):
                    exit_triggered = True
                    exit_price = curr_close
                    exit_reason = "EOD_1515_SQUAREOFF"

                # C. Time Stop (45 minutes / 9 bars of 5m with no progress)
                elif holding_bars >= 9:
                    progress = (curr_close - pos_entry_price) if pos_direction == "LONG" else (pos_entry_price - curr_close)
                    if progress < (0.2 * abs(pos_target - pos_entry_price)):
                        exit_triggered = True
                        exit_price = curr_close
                        exit_reason = "TIME_STOP_45M"

                # D. Target Hit (1:2 R:R)
                if not exit_triggered:
                    if pos_direction == "LONG" and curr_high >= pos_target:
                        exit_triggered = True
                        exit_price = pos_target - (self.slippage_points / 2.0)
                        exit_reason = "TARGET_HIT"
                    elif pos_direction == "SHORT" and curr_low <= pos_target:
                        exit_triggered = True
                        exit_price = pos_target + (self.slippage_points / 2.0)
                        exit_reason = "TARGET_HIT"

                # E. Stop Loss Hit
                if not exit_triggered:
                    if pos_direction == "LONG" and curr_low <= pos_stop_loss:
                        exit_triggered = True
                        exit_price = pos_stop_loss - (self.slippage_points / 2.0)
                        exit_reason = "STOP_LOSS_HIT"
                    elif pos_direction == "SHORT" and curr_high >= pos_stop_loss:
                        exit_triggered = True
                        exit_price = pos_stop_loss + (self.slippage_points / 2.0)
                        exit_reason = "STOP_LOSS_HIT"

                # F. Dynamic Trailing Stop (Trail at +1.5R to avoid early shakeouts)
                if not exit_triggered:
                    r_dist = abs(pos_target - pos_entry_price) / 2.0  # 1R distance
                    if pos_direction == "LONG":
                        if curr_close >= pos_entry_price + (1.6 * r_dist):
                            pos_stop_loss = max(pos_stop_loss, pos_entry_price + (0.8 * r_dist))
                        elif curr_close >= pos_entry_price + (1.2 * r_dist):
                            pos_stop_loss = max(pos_stop_loss, pos_entry_price + (0.2 * r_dist))
                    else:
                        if curr_close <= pos_entry_price - (1.6 * r_dist):
                            pos_stop_loss = min(pos_stop_loss, pos_entry_price - (0.8 * r_dist))
                        elif curr_close <= pos_entry_price - (1.2 * r_dist):
                            pos_stop_loss = min(pos_stop_loss, pos_entry_price - (0.2 * r_dist))

                # Process Exit
                if exit_triggered:
                    # Realistic F&O cost: ₹40 round-trip + 0.03% on option premium (approx ₹200-300 per lot)
                    opt_premium_turnover = 250.0 * pos_qty * 2.0  # ₹250 avg option premium turnover
                    brokerage = self.brokerage_per_order * 2.0    # Flat ₹40
                    stt_fees = opt_premium_turnover * self.stt_turnover_pct
                    total_costs = brokerage + stt_fees

                    # Options Delta ~0.50 price conversion
                    delta_factor = 0.50
                    spot_diff = (exit_price - pos_entry_price) if pos_direction == "LONG" else (pos_entry_price - exit_price)
                    gross_pnl = spot_diff * delta_factor * pos_qty
                    net_pnl = gross_pnl - total_costs
                    current_capital += net_pnl
                    ret_pct = (net_pnl / daily_start_capital) * 100.0

                    trade_rec = BacktestTrade(
                        trade_id=len(trades) + 1,
                        symbol=symbol,
                        direction=pos_direction,
                        entry_time=str(pos_entry_time),
                        entry_price=round(pos_entry_price, 2),
                        exit_time=str(ts),
                        exit_price=round(exit_price, 2),
                        quantity=pos_qty,
                        gross_pnl=round(gross_pnl, 2),
                        brokerage_and_stt=round(total_costs, 2),
                        net_pnl=round(net_pnl, 2),
                        return_pct=round(ret_pct, 2),
                        exit_reason=exit_reason,
                        holding_bars=holding_bars,
                        regime_at_entry=pos_regime,
                        consensus_votes=pos_votes,
                    )
                    trades.append(trade_rec)
                    monthly_pnl[month_str] = monthly_pnl.get(month_str, 0.0) + net_pnl

                    in_position = False
                    pos_direction = "NONE"

                    peak_capital = max(peak_capital, current_capital)
                    dd = peak_capital - current_capital
                    dd_pct = (dd / peak_capital) * 100.0
                    max_drawdown = max(max_drawdown, dd)
                    max_drawdown_pct = max(max_drawdown_pct, dd_pct)

                continue

            # ── 2. EVALUATE NEW ENTRY ─────────────────────────────────────────
            # Timing & Daily Limits
            if bar_time < time(9, 30) or bar_time > time(14, 30) or daily_circuit_locked or daily_trades_taken >= self.max_trades_per_day:
                continue

            # Layer 3: Regime Classification
            adx_val = float(curr["adx14"])
            pdi_val = float(curr["plus_di"])
            mdi_val = float(curr["minus_di"])
            c_val = float(curr["close"])
            ema20_val = float(curr["ema20"])
            bbw_rank_val = float(curr["bbw_rank"])

            if bbw_rank_val <= 0.20:
                regime_name = "COMPRESSION"
            elif adx_val >= 25.0:
                if c_val > ema20_val and pdi_val > mdi_val:
                    regime_name = "TRENDING_UP"
                elif c_val < ema20_val and mdi_val > pdi_val:
                    regime_name = "TRENDING_DOWN"
                else:
                    regime_name = "CHOP"
            elif adx_val < 20.0:
                regime_name = "CHOP"
            else:
                regime_name = "NEUTRAL"

            regime_counts[regime_name] = regime_counts.get(regime_name, 0) + 1

            if self.use_regime_filter and regime_name == "CHOP":
                continue  # 100% Cash Gating strictly in low-ADX Chop

            # Layer 4: 4-Strategy Ensemble Crossover Rules
            atr_val = max(15.0, float(curr["atr14"]))
            vol_val = float(curr["volume"])
            vol_ma_val = float(curr["vol_ma20"])
            vwap_val = float(curr["vwap"])
            prev_vwap_val = float(prev["vwap"])
            prev_c_val = float(prev["close"])
            prev_o_val = float(prev["open"])
            curr_o_val = float(curr["open"])
            orh_val = float(curr["orh"])
            orl_val = float(curr["orl"])
            st_dir_val = float(curr["supertrend_dir"])
            prev_st_dir = float(prev["supertrend_dir"])
            ema9_val = float(curr["ema9"])
            ema21_val = float(curr["ema21"])
            prev_ema9 = float(prev["ema9"])
            prev_ema21 = float(prev["ema21"])

            long_votes = 0
            short_votes = 0

            # Strategy 1: ORB Breakout (Trigger on the breakout cross bar 09:30-11:30)
            if bar_time <= time(11, 30):
                if prev_c_val <= orh_val and c_val > orh_val and vol_val >= (vol_ma_val * 1.2):
                    long_votes += 1
                elif prev_c_val >= orl_val and c_val < orl_val and vol_val >= (vol_ma_val * 1.2):
                    short_votes += 1

            # Strategy 2: VWAP Reclaim (Crossover with volume)
            if prev_c_val < prev_vwap_val and c_val > vwap_val and vol_val >= vol_ma_val:
                long_votes += 1
            elif prev_c_val > prev_vwap_val and c_val < vwap_val and vol_val >= vol_ma_val:
                short_votes += 1

            # Strategy 3: EMA-Supertrend Alignment / Fresh Cross
            ema_bull_cross = (prev_ema9 <= prev_ema21 and ema9_val > ema21_val) or (st_dir_val == 1.0 and prev_st_dir == -1.0)
            ema_bear_cross = (prev_ema9 >= prev_ema21 and ema9_val < ema21_val) or (st_dir_val == -1.0 and prev_st_dir == 1.0)
            if (ema9_val > ema21_val and st_dir_val == 1.0) and (ema_bull_cross or c_val > prev_c_val):
                long_votes += 1
            elif (ema9_val < ema21_val and st_dir_val == -1.0) and (ema_bear_cross or c_val < prev_c_val):
                short_votes += 1

            # Strategy 4: Institutional Flow / Volume Flow Confluence
            if regime_name in ("TRENDING_UP", "COMPRESSION") and vol_val >= (vol_ma_val * 1.1):
                long_votes += 1
            elif regime_name == "TRENDING_DOWN" and vol_val >= (vol_ma_val * 1.1):
                short_votes += 1

            # Consensus Rule: >=3 = 100% Size, 2 = 50% Size, <2 = Reject
            chosen_dir = "NONE"
            multiplier = 0.0
            total_agree = 0

            if long_votes >= 3:
                chosen_dir = "LONG"
                multiplier = 1.0
                total_agree = long_votes
            elif short_votes >= 3:
                chosen_dir = "SHORT"
                multiplier = 1.0
                total_agree = short_votes
            elif long_votes == 2 and short_votes == 0:
                chosen_dir = "LONG"
                multiplier = 0.5
                total_agree = 2
            elif short_votes == 2 and long_votes == 0:
                chosen_dir = "SHORT"
                multiplier = 0.5
                total_agree = 2
            else:
                continue

            # Layer 5: Jane Street Anti-Spoofing Check (2nd confirming candle)
            if self.use_jane_street_defense:
                if chosen_dir == "LONG":
                    if not (c_val >= curr_o_val and (prev_c_val >= prev_o_val or c_val > prev_c_val)):
                        continue
                else:
                    if not (c_val <= curr_o_val and (prev_c_val <= prev_o_val or c_val < prev_c_val)):
                        continue

            # Entry, Targets, Stops (1.2 ATR Risk, 2.5 ATR Reward -> 1:2.1 R:R)
            entry_p = c_val + (self.slippage_points if chosen_dir == "LONG" else -self.slippage_points)
            if chosen_dir == "LONG":
                sl_p = entry_p - (1.2 * atr_val)
                tp_p = entry_p + (2.5 * atr_val)
            else:
                sl_p = entry_p + (1.2 * atr_val)
                tp_p = entry_p - (2.5 * atr_val)

            # Layer 6: Kelly Position Sizing (Allocates 1.0% hard risk per trade)
            risk_rupees = current_capital * 0.01  # 1% risk
            risk_per_unit = max(5.0, (1.2 * atr_val) * 0.50)  # Option delta adjusted risk
            target_qty = int(risk_rupees / risk_per_unit)
            allocated_qty = int(target_qty * multiplier)
            allocated_qty = max(self.lot_size, (allocated_qty // self.lot_size) * self.lot_size)

            in_position = True
            pos_direction = chosen_dir
            pos_entry_price = entry_p
            pos_entry_time = ts
            pos_entry_idx = i
            pos_stop_loss = sl_p
            pos_target = tp_p
            pos_qty = allocated_qty
            pos_regime = regime_name
            pos_votes = total_agree
            daily_trades_taken += 1

        # Wrap up daily equity
        if current_day_str and (not daily_equity or daily_equity[-1]["date"] != current_day_str):
            daily_equity.append({
                "date": current_day_str,
                "capital": round(current_capital, 2),
                "daily_pnl": round(current_capital - daily_start_capital, 2),
                "daily_return_pct": round(((current_capital - daily_start_capital) / daily_start_capital) * 100.0, 2),
            })

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.net_pnl > 0)
        losing_trades = sum(1 for t in trades if t.net_pnl <= 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0

        gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
        gross_loss = abs(sum(t.net_pnl for t in trades if t.net_pnl < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

        total_net_pnl = current_capital - self.initial_capital
        total_return_pct = (total_net_pnl / self.initial_capital) * 100.0

        avg_trade = (total_net_pnl / total_trades) if total_trades > 0 else 0.0
        avg_win = (gross_profit / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (gross_loss / losing_trades) if losing_trades > 0 else 0.0
        wl_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

        daily_returns = [d["daily_return_pct"] for d in daily_equity]
        if len(daily_returns) > 5 and np.std(daily_returns) > 0:
            sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
        else:
            sharpe = 0.0

        consec = 0
        max_consec = 0
        for t in trades:
            if t.net_pnl <= 0:
                consec += 1
                max_consec = max(max_consec, consec)
            else:
                consec = 0

        total_brokerage = sum(t.brokerage_and_stt for t in trades)

        return BacktestMetrics(
            initial_capital=round(self.initial_capital, 2),
            final_capital=round(current_capital, 2),
            total_net_pnl=round(total_net_pnl, 2),
            total_return_pct=round(total_return_pct, 2),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate_pct=round(win_rate, 2),
            profit_factor=round(profit_factor, 2),
            max_drawdown_rupees=round(max_drawdown, 2),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            sharpe_ratio=round(sharpe, 2),
            avg_trade_pnl=round(avg_trade, 2),
            avg_win_pnl=round(avg_win, 2),
            avg_loss_pnl=round(avg_loss, 2),
            win_loss_ratio=round(wl_ratio, 2),
            max_consecutive_losses=max_consec,
            total_brokerage_paid=round(total_brokerage, 2),
            monthly_pnl={k: round(v, 2) for k, v in monthly_pnl.items()},
            daily_equity_curve=daily_equity,
            regime_breakdown=regime_counts,
        )
