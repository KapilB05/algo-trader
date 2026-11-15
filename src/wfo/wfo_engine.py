"""
Walk-Forward Optimization (WFO) Engine (src/wfo/wfo_engine.py).

Automated Walk-Forward Optimization pipeline:
- Rolling window optimization (70% In-Sample training, 30% Out-of-Sample validation).
- Calibrates strategy parameters: volume_surge_mult, vix_high_threshold, confluence_min_score, atr_trail_mult, min_rr_ratio.
- Generates versioned parameter store (data/wfo_parameters.json) and execution history (data/wfo_history.json).
- Supports hot-reloading parameters into memory without server restarts.
- Strict Safety Protocol: New parameters are deployed to Paper simulation mode first before promotion to Live.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

logger = logging.getLogger("WFOEngine")
IST = ZoneInfo("Asia/Kolkata")
WFO_STORE_PATH = Path("data/wfo_parameters.json")
WFO_HISTORY_PATH = Path("data/wfo_history.json")

# Absolute out-of-sample edge floor (annualized Sharpe, same units as the backtest
# engine and diagnostic simulator). The old purely-relative gate ("OOS >= 50% of IS")
# passed calibrations like IS=0.03/OOS=0.07 — noise with no tradable edge — and
# inflated the version history past v100 without ever approaching the 1.50
# promote-to-live bar. A window must now clear BOTH bars.
MIN_OOS_SHARPE_ABSOLUTE = 0.50


@dataclass
class WFOParameters:
    """Calibrated strategy parameters."""

    volume_surge_mult: float = 1.5
    vix_high_threshold: float = 18.0
    confluence_min_score: float = 0.65
    atr_trail_mult: float = 1.5
    min_rr_ratio: float = 2.0
    version: int = 1
    calibrated_at: str = field(default_factory=lambda: datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"))
    deployment_mode: str = "PAPER"  # Safety Rule: PAPER first before LIVE promotion
    in_sample_sharpe: float = 0.0
    out_of_sample_sharpe: float = 0.0
    profit_factor: float = 0.0
    win_rate_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_simulated_trades: int = 0


@dataclass
class WindowResult:
    """Performance metrics for an individual rolling window."""

    window_id: int
    is_window_days: int
    oos_window_days: int
    is_sharpe: float
    oos_sharpe: float
    is_win_rate_pct: float
    oos_win_rate_pct: float
    oos_profit_factor: float
    oos_max_dd_pct: float
    passed_validation: bool
    best_params: Dict[str, float]


class WFOEngine:
    """
    Production Walk-Forward Optimization Pipeline.
    """

    def __init__(self, store_path: Path = WFO_STORE_PATH, history_path: Path = WFO_HISTORY_PATH) -> None:
        self.store_path = store_path
        self.history_path = history_path
        self.active_parameters = WFOParameters()
        self.load_parameters()

    def load_parameters(self) -> WFOParameters:
        """Load latest approved parameters from versioned store."""
        try:
            if self.store_path.exists():
                with open(self.store_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.active_parameters = WFOParameters(**data)
                    logger.info("Loaded WFO parameters v%d (Mode: %s) calibrated at %s", self.active_parameters.version, self.active_parameters.deployment_mode, self.active_parameters.calibrated_at)
        except Exception as exc:
            logger.error("Failed to load WFO parameters: %s", exc)
        return self.active_parameters

    def save_parameters(self, params: WFOParameters) -> None:
        """Persist calibrated parameter set and append to version history."""
        try:
            self.store_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.store_path, "w", encoding="utf-8") as f:
                json.dump(asdict(params), f, indent=2)
            self.active_parameters = params

            # Append to history
            history = []
            if self.history_path.exists():
                try:
                    with open(self.history_path, "r", encoding="utf-8") as hf:
                        history = json.load(hf)
                except Exception:
                    history = []
            history.append(asdict(params))
            with open(self.history_path, "w", encoding="utf-8") as hf:
                json.dump(history, hf, indent=2)

            logger.info("WFO parameters v%d saved and archived to history.", params.version)
        except Exception as exc:
            logger.error("Failed to save WFO parameters: %s", exc)

    def promote_to_live(self) -> Tuple[bool, str]:
        """
        Safety Promotion: Explicitly promotes validated PAPER parameters to LIVE mode.
        """
        if self.active_parameters.deployment_mode == "LIVE":
            return True, "Parameters already promoted to LIVE mode."

        if self.active_parameters.out_of_sample_sharpe < 1.50:
            return False, f"Promotion blocked: OOS Sharpe {self.active_parameters.out_of_sample_sharpe:.2f} < 1.50 minimum threshold (Phase 4 live-readiness bar)."

        self.active_parameters.deployment_mode = "LIVE"
        self.save_parameters(self.active_parameters)
        logger.warning("🚀 WFO Parameters v%d PROMOTED TO LIVE TRADING MODE", self.active_parameters.version)
        return True, f"WFO parameters v{self.active_parameters.version} promoted to LIVE mode successfully."

    def run_walk_forward_optimization(
        self,
        historical_candles: Optional[List[Dict[str, Any]]] = None,
        window_days: int = 90,
        n_splits: int = 3,
        in_sample_ratio: float = 0.70,
        trace_id: str = "",
    ) -> Tuple[WFOParameters, Dict[str, Any]]:
        """
        Execute Walk-Forward Optimization across rolling multi-window time series.

        Runs an honest event-driven backtest of the confluence entry model
        (volume-surge gate + VWAP directional filter, ATR-trailing / min-RR exits)
        over the supplied candles. Requires real historical data — refuses to
        fabricate results when none is available.
        Returns newly calibrated WFOParameters and performance report dict.
        """
        tr_id = trace_id or str(uuid.uuid4())[:8]

        rows = [c for c in (historical_candles or []) if self._valid_candle(c)]
        min_required = 300
        if len(rows) < min_required:
            raise ValueError(
                f"WFO requires real historical candles ({min_required}+ usable rows, "
                f"got {len(rows)}). Refusing to simulate results without data."
            )

        logger.info(
            "[%s] 🧪 Starting Walk-Forward Optimization (%d candles, %d splits, %.0f%% IS / %.0f%% OOS)...",
            tr_id, len(rows), n_splits, in_sample_ratio * 100, (1 - in_sample_ratio) * 100,
        )

        # Candidate parameter grid search space
        param_grid = [
            {"volume_surge_mult": 1.2, "vix_high_threshold": 16.0, "confluence_min_score": 0.60, "atr_trail_mult": 1.2, "min_rr_ratio": 1.8},
            {"volume_surge_mult": 1.5, "vix_high_threshold": 18.0, "confluence_min_score": 0.65, "atr_trail_mult": 1.5, "min_rr_ratio": 2.0},
            {"volume_surge_mult": 1.8, "vix_high_threshold": 20.0, "confluence_min_score": 0.70, "atr_trail_mult": 1.8, "min_rr_ratio": 2.2},
            {"volume_surge_mult": 2.0, "vix_high_threshold": 22.0, "confluence_min_score": 0.75, "atr_trail_mult": 2.0, "min_rr_ratio": 2.5},
        ]

        import pandas as pd

        df_all = pd.DataFrame(rows)
        df_all = df_all.sort_values("time").reset_index(drop=True)

        fold_size = len(df_all) // n_splits
        is_days = int(window_days * in_sample_ratio)
        oos_days = int(window_days * (1 - in_sample_ratio))

        windows_results: List[WindowResult] = []

        for w_idx in range(n_splits):
            fold = df_all.iloc[w_idx * fold_size:(w_idx + 1) * fold_size]
            
            # --- 2026 Adaptive Windowing Research Integration ---
            # Adjust IS ratio dynamically based on recent market volatility
            if "india_vix" in fold.columns:
                avg_vol = float(fold["india_vix"].mean())
            elif "close" in fold.columns:
                returns = fold["close"].astype(float).pct_change().dropna()
                avg_vol = float(returns.std() * math.sqrt(252 * 75) * 100) if len(returns) > 0 else 15.0
            else:
                avg_vol = 15.0

            if avg_vol > 18.0:
                current_is_ratio = 0.50 # High vol: adapt faster with 50% split
            elif avg_vol < 13.0:
                current_is_ratio = 0.80 # Low vol: train longer with 80% split
            else:
                current_is_ratio = in_sample_ratio
                
            split = int(len(fold) * current_is_ratio)
            is_df = fold.iloc[:split].reset_index(drop=True)
            oos_df = fold.iloc[split:].reset_index(drop=True)
            
            is_days = int(window_days * current_is_ratio)
            oos_days = int(window_days * (1 - current_is_ratio))

            best_candidate: Optional[Dict[str, float]] = None
            best_is_sharpe = float("-inf")
            best_is_metrics: Dict[str, Any] = {}
            best_oos_metrics: Dict[str, Any] = {}

            for cand in param_grid:
                is_m = self._backtest(is_df, cand)
                # In-sample selection gate: enough trades and positive Sharpe
                if is_m["trades"] < 10 or is_m["sharpe"] <= 0:
                    continue
                if is_m["sharpe"] > best_is_sharpe:
                    best_is_sharpe = is_m["sharpe"]
                    best_candidate = cand
                    best_is_metrics = is_m

            if best_candidate is None:
                logger.info("[%s] Window %d: no candidate passed in-sample gates.", tr_id, w_idx + 1)
                continue

            best_oos_metrics = self._backtest(oos_df, best_candidate)

            # Overfitting Validation Gate:
            # 1. Enough OOS trades to be meaningful
            # 2. Absolute OOS edge: annualized Sharpe >= 0.50 (same units as the
            #    backtest engine; the old per-trade Sharpe read ~0.02 for this
            #    strategy, which annualizes to ~0.25 — correctly rejected)
            # 3. OOS Sharpe >= 50% of IS Sharpe (no cliff-edge decay)
            passed_gate = (
                best_oos_metrics["trades"] >= 5
                and best_oos_metrics["sharpe"] >= MIN_OOS_SHARPE_ABSOLUTE
                and best_oos_metrics["sharpe"] >= (best_is_metrics["sharpe"] * 0.50)
            )

            windows_results.append(WindowResult(
                window_id=w_idx + 1,
                is_window_days=is_days,
                oos_window_days=oos_days,
                is_sharpe=round(best_is_metrics["sharpe"], 2),
                oos_sharpe=round(best_oos_metrics["sharpe"], 2),
                is_win_rate_pct=round(best_is_metrics["win_rate_pct"], 1),
                oos_win_rate_pct=round(best_oos_metrics["win_rate_pct"], 1),
                oos_profit_factor=round(best_oos_metrics["profit_factor"], 2),
                oos_max_dd_pct=round(best_oos_metrics["max_dd_pct"], 2),
                passed_validation=passed_gate,
                best_params=best_candidate,
            ))

        passing_windows = [w for w in windows_results if w.passed_validation]

        if not passing_windows:
            report = {
                "status": "NO_PASSING_WINDOW",
                "trace_id": tr_id,
                "total_windows": len(windows_results),
                "passed_windows": 0,
                "rolling_windows": [asdict(w) for w in windows_results],
                "active_version": self.active_parameters.version,
                "message": (
                    "No parameter set passed out-of-sample validation on real data — "
                    "existing parameters left unchanged."
                ),
            }
            logger.warning("[%s] ⚠️ WFO completed with no passing windows. Parameters unchanged.", tr_id)
            return self.active_parameters, report

        selected_window = passing_windows[-1]
        chosen_params = selected_window.best_params

        new_version = self.active_parameters.version + 1
        calibrated = WFOParameters(
            volume_surge_mult=chosen_params["volume_surge_mult"],
            vix_high_threshold=chosen_params["vix_high_threshold"],
            confluence_min_score=chosen_params["confluence_min_score"],
            atr_trail_mult=chosen_params["atr_trail_mult"],
            min_rr_ratio=chosen_params["min_rr_ratio"],
            version=new_version,
            deployment_mode="PAPER",  # Safety First
            in_sample_sharpe=selected_window.is_sharpe,
            out_of_sample_sharpe=selected_window.oos_sharpe,
            profit_factor=selected_window.oos_profit_factor,
            win_rate_pct=selected_window.oos_win_rate_pct,
            max_drawdown_pct=selected_window.oos_max_dd_pct,
            total_simulated_trades=sum(
                self._backtest(
                    df_all.iloc[w_idx * fold_size:((w_idx + 1) * fold_size)].iloc[int(fold_size * in_sample_ratio):],
                    chosen_params,
                )["trades"]
                for w_idx in range(n_splits)
            ),
        )

        self.save_parameters(calibrated)

        report = {
            "status": "SUCCESS",
            "trace_id": tr_id,
            "version": new_version,
            "calibrated_parameters": asdict(calibrated),
            "rolling_windows": [asdict(w) for w in windows_results],
            "total_windows": len(windows_results),
            "passed_windows": len(passing_windows),
            "in_sample_sharpe": calibrated.in_sample_sharpe,
            "out_of_sample_sharpe": calibrated.out_of_sample_sharpe,
            "profit_factor": calibrated.profit_factor,
            "win_rate_pct": calibrated.win_rate_pct,
            "max_drawdown_pct": calibrated.max_drawdown_pct,
            "safety_protocol": "Parameters deployed to PAPER simulation mode first before LIVE promotion.",
        }

        logger.info("[%s] ✅ Walk-Forward Optimization Completed: Version v%d active in PAPER mode.", tr_id, new_version)
        return calibrated, report

    # ── Honest backtest core ──────────────────────────────────────────────
    @staticmethod
    def _valid_candle(c: Any) -> bool:
        """A candle must carry OHLC (+volume when present) as numbers."""
        if not isinstance(c, dict):
            return False
        try:
            for k in ("open", "high", "low", "close"):
                v = c.get(k)
                if v is None or not isinstance(v, (int, float)) or isinstance(v, bool):
                    return False
            return True
        except Exception:
            return False

    @staticmethod
    def _backtest(df: Any, params: Dict[str, float]) -> Dict[str, Any]:
        """
        Deterministic single-position backtest of the entry/exit model.

        Entry: volume surge (vol > SMA20(vol) * volume_surge_mult) combined with
               a VWAP directional filter (close above VWAP → LONG, below → SHORT).
        Exit:  ATR(14) trailing stop (atr_trail_mult) or min_rr_ratio target,
               measured from entry price with 1×ATR initial risk.
        No randomness — identical inputs always produce identical metrics.
        """
        import pandas as pd

        n = len(df)
        empty = {"trades": 0, "sharpe": 0.0, "win_rate_pct": 0.0, "profit_factor": 0.0, "max_dd_pct": 0.0}
        if n < 60:
            return empty

        o = df["open"].astype(float).reset_index(drop=True)
        h = df["high"].astype(float).reset_index(drop=True)
        l = df["low"].astype(float).reset_index(drop=True)
        c = df["close"].astype(float).reset_index(drop=True)
        v = df["volume"].astype(float).reset_index(drop=True) if "volume" in df else pd.Series([1.0] * n)

        vol_sma = v.rolling(20).mean()
        prev_close = c.shift(1)
        tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        cum_vol = v.cumsum()
        cum_pv = (c * v).cumsum()
        vwap = (cum_pv / cum_vol.replace(0, pd.NA)).ffill()

        trades: List[float] = []
        trade_days: List[Any] = []
        try:
            _dates = pd.to_datetime(df["time"]).dt.date
            _has_dates = True
        except Exception:
            _has_dates = False
        position = 0  # 0 flat, +1 long, -1 short
        entry_price = stop = target = 0.0
        trail_stop = 0.0

        for i in range(20, n):
            price = float(c.iloc[i])
            atr_i = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0
            vol_ok = (
                pd.notna(vol_sma.iloc[i])
                and float(vol_sma.iloc[i]) > 0
                and float(v.iloc[i]) > float(vol_sma.iloc[i]) * params["volume_surge_mult"]
            )
            vw = float(vwap.iloc[i]) if pd.notna(vwap.iloc[i]) else price

            if position != 0:
                if position == 1:
                    trail_stop = max(trail_stop, price - params["atr_trail_mult"] * atr_i) if atr_i > 0 else trail_stop
                    hit_stop = price <= trail_stop
                    hit_target = price >= target
                else:
                    trail_stop = min(trail_stop, price + params["atr_trail_mult"] * atr_i) if atr_i > 0 else trail_stop
                    hit_stop = price >= trail_stop
                    hit_target = price <= target
                if hit_stop or hit_target:
                    ret = (price - entry_price) / entry_price * position
                    trades.append(ret)
                    trade_days.append(_dates.iloc[i] if _has_dates else i // 75)
                    position = 0
                continue

            if vol_ok and atr_i > 0:
                if price > vw:  # LONG
                    position, entry_price = 1, price
                    stop = price - atr_i
                    trail_stop = stop
                    target = price + params["min_rr_ratio"] * atr_i
                elif price < vw:  # SHORT
                    position, entry_price = -1, price
                    stop = price + atr_i
                    trail_stop = stop
                    target = price - params["min_rr_ratio"] * atr_i

        if not trades:
            return empty

        rets = pd.Series(trades)
        wins = rets[rets > 0]
        losses = rets[rets <= 0]
        gross_win = float(wins.sum())
        gross_loss = abs(float(losses.sum()))
        equity = (1 + rets).cumprod()
        peak = equity.cummax()
        dd = ((equity / peak) - 1.0).min()

        # Annualized Sharpe from daily-aggregated returns (same units as the
        # backtest engine and diagnostic simulator: mean/std * sqrt(252)).
        # The old per-trade Sharpe understated readings ~10-40x and made the
        # "Config vs backtest Sharpe" gap look like a data bug instead of a
        # genuinely weak strategy.
        daily_pnl: dict[Any, float] = {}
        for d, r in zip(trade_days, trades, strict=True):
            daily_pnl[d] = daily_pnl.get(d, 0.0) + float(r)
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 5 and float(pd.Series(daily_returns).std(ddof=1)) > 0:
            sharpe = float(
                pd.Series(daily_returns).mean() / pd.Series(daily_returns).std(ddof=1)
            ) * math.sqrt(252)
        else:
            sharpe = 0.0
        return {
            "trades": len(trades),
            "sharpe": sharpe,
            "win_rate_pct": float((rets > 0).mean() * 100.0),
            "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else (10.0 if gross_win > 0 else 0.0),
            "max_dd_pct": float(abs(dd) * 100.0) if pd.notna(dd) else 0.0,
        }


# Global singleton instance
wfo_engine = WFOEngine()

