"""
Continuous Auto-Learner — Configuration Version Controller.

Manages strategy parameter deployment, backups, and audit history:
1. Archives previous live config to `archive/v1/config/config_v{timestamp}.yaml`
2. Atomically deploys candidate parameters to `config/optimized_strategy_config.yaml`
3. Records all parameter promotions to `config/version_history.csv`
"""

from __future__ import annotations

import csv
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from src.config.lot_sizes import get_lot_size

logger = logging.getLogger("AutoLearner.VersionController")
CONFIG_PATH = Path("config/optimized_strategy_config.yaml")
ARCHIVE_DIR = Path("archive/v1/config")
VERSION_CSV = Path("config/version_history.csv")

HISTORY_HEADER = [
    "timestamp", "version", "old_sharpe", "new_sharpe",
    "improvement_pct", "sl_atr", "tp_atr", "vix_gate",
    "consensus_rule", "promoted_by", "reason",
    "alpha_annual_pct", "info_ratio", "excess_return_pct",
    "bench_return_pct", "n_sessions", "significant",
]

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


PENDING_PROMOTION_PATH = Path("data/pending_promotion.yaml")

# Phase 4 live-readiness bar — a candidate must clear ALL of these before a
# "HumanOperator" promotion row may be written. Mirrors src/main.py's
# validate_live_readiness gate so the promote path can never bypass it.
MIN_PROMOTABLE_ALPHA_SIGNIFICANT = True
MIN_PROMOTABLE_N_SESSIONS = 20
MIN_PROMOTABLE_OOS_SHARPE = 1.5


class VersionController:
    """
    Manages safe strategy parameter promotion, archival, rollback, and manual human sign-off.
    """

    def __init__(self, require_manual_approval: bool = True) -> None:
        self.require_manual_approval = require_manual_approval
        self._init_history_csv()

    def _init_history_csv(self) -> None:
        """Ensure CSV audit log header exists and carries the alpha columns."""
        if not VERSION_CSV.exists():
            with open(VERSION_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(HISTORY_HEADER)
            return

        # Migrate existing file: add alpha columns if header is stale
        with open(VERSION_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if rows and rows[0] != HISTORY_HEADER:
            old = rows[0]
            pad = ["", "", "", "", "", ""]  # 6 new trailing columns
            migrated = [HISTORY_HEADER] + [r + pad[:len(HISTORY_HEADER) - len(r)] for r in rows[1:]]
            with open(VERSION_CSV, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerows(migrated)
            logger.info("Migrated version_history.csv header to include alpha attribution columns.")

    def promote_new_configuration(
        self,
        new_params: Dict[str, Any],
        old_sharpe: float,
        new_sharpe: float,
        improvement_pct: float,
        alpha_annual_pct: float = 0.0,
        information_ratio: float = 0.0,
        excess_return_pct: float = 0.0,
        bench_return_pct: float = 0.0,
        n_sessions: int = 0,
        significant: bool = False,
        benchmark: str = "NAIVE_LONG",
        reason: str = "Weekly Auto-Learner Walk-Forward Optimization",
    ) -> Tuple[bool, str, Path]:
        """
        Stage or deploy new YAML configuration, archive old, and log to CSV.
        If require_manual_approval is True, stages to data/pending_promotion.yaml.
        """
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_path = ARCHIVE_DIR / f"config_v{ts_str}.yaml"

        # 1. Archive current active config if present
        if CONFIG_PATH.exists():
            shutil.copy(CONFIG_PATH, archived_path)
            logger.info("Archived current live config to %s", archived_path)

        # 2. Build new version configuration payload
        new_version_tag = f"2.{int(datetime.now().timestamp()) % 10000}.0-auto-optimized"
        full_config = {
            "version": new_version_tag,
            "generated_at": datetime.now().isoformat(),
            "primary_asset": {
                "symbol": "BANKNIFTY",
                "exchange": "NFO",
                "instrument_type": "OPTIDX",
                "lot_size": get_lot_size("BANKNIFTY"),
                "status": "ACTIVE",
            },
            "secondary_asset": {
                "symbol": "NIFTY",
                "status": "DISABLED",
                "reason": "Negative mathematical expectancy in intraday breakout regime",
            },
            "ensemble_configuration": {
                "active_strategies": ["ORB", "SUPERTREND", "FLOW"],
                "excluded_strategies": ["VWAP"],
                "voting_rule": new_params.get("consensus_rule", "BENCHMARK"),
                "consensus_matrix": {
                    "3_of_3_agree": 1.0,
                    "2_of_3_agree": 0.5,
                    "less_than_2": 0.0,
                },
            },
            "risk_and_execution_parameters": {
                "stop_loss_atr_multiplier": float(new_params.get("sl_atr_mult", 1.0)),
                "take_profit_atr_multiplier": float(new_params.get("tp_atr_mult", 2.0)),
                "risk_reward_ratio": round(float(new_params.get("tp_atr_mult", 2.0)) / float(new_params.get("sl_atr_mult", 1.0)), 2),
                "max_risk_per_trade_pct": 0.01,
                "max_daily_loss_pct": 0.03,
                "trailing_stop": {
                    "activation_r_multiple": 1.6,
                    "trail_lock_r_multiple": 0.8,
                },
            },
            "regime_and_volatility_filters": {
                "chop_cash_gating_enabled": True,
                "min_india_vix_entry": float(new_params.get("min_vix_gate", 11.5)),
                "max_india_vix_entry": 24.0,
                "adx_trend_threshold": 25.0,
                "adx_chop_threshold": 20.0,
                "hysteresis_bars": 3,
            },
            "jane_street_defense": {
                "anti_spoofing_2nd_candle_confirmation": True,
                "expiry_mandatory_cutoff_time_ist": "14:30",
                "eod_auto_squareoff_time_ist": "15:15",
                "max_oi_trap_trim_pct": 0.005,
            },
            "backtested_performance": {
                "out_of_sample_sharpe": float(new_sharpe),
                "previous_sharpe": float(old_sharpe),
                "improvement_pct": float(improvement_pct),
                "benchmark": benchmark,
                "alpha_annual_pct": float(alpha_annual_pct),
                "information_ratio": float(information_ratio),
                "excess_return_pct": float(excess_return_pct),
                "benchmark_return_pct": float(bench_return_pct),
                "n_sessions": n_sessions,
                "alpha_significant": significant,
            }
        }

        # 3. Handle manual approval gate vs direct promotion
        target_file = PENDING_PROMOTION_PATH if self.require_manual_approval else CONFIG_PATH
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        reason_str = f"{reason} (Awaiting Manual Sign-off)" if self.require_manual_approval else reason

        # 4. Log to version history CSV
        with open(VERSION_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                new_version_tag,
                round(old_sharpe, 2),
                round(new_sharpe, 2),
                f"{improvement_pct:+.1f}%",
                new_params.get("sl_atr_mult"),
                new_params.get("tp_atr_mult"),
                new_params.get("min_vix_gate"),
                new_params.get("consensus_rule"),
                "AutoLearner",
                reason_str,
                round(alpha_annual_pct, 2),
                round(information_ratio, 2),
                round(excess_return_pct, 2),
                round(bench_return_pct, 2),
                n_sessions,
                "1" if significant else "0",
            ])

        if self.require_manual_approval:
            logger.info("STAGED: Candidate parameters staged to %s. Awaiting manual dashboard sign-off.", target_file)
        else:
            logger.info("PROMOTED: Deployed new live configuration %s (Sharpe: %.2f -> %.2f, +%.1f%%)",
                        new_version_tag, old_sharpe, new_sharpe, improvement_pct)

        return True, new_version_tag, target_file

    def apply_pending_promotion(self) -> Tuple[bool, str]:
        """Apply staged configuration from data/pending_promotion.yaml to live config."""
        if not PENDING_PROMOTION_PATH.exists():
            return False, "No pending promotion staged in data/pending_promotion.yaml."

        try:
            with open(PENDING_PROMOTION_PATH, "r", encoding="utf-8") as f:
                pending_data = yaml.safe_load(f) or {}

            # Phase 4 safety validation gate: refuse to promote candidates that
            # do not clear the same bar as the LIVE SAFETY GATE in src/main.py.
            # This makes a phantom 'HumanOperator' promotion impossible.
            perf = pending_data.get("backtested_performance", {}) or {}
            alpha_significant = bool(perf.get("alpha_significant", False))
            n_sessions = int(perf.get("n_sessions", 0) or 0)
            oos_sharpe = float(perf.get("out_of_sample_sharpe", 0.0) or 0.0)
            if not alpha_significant or n_sessions < MIN_PROMOTABLE_N_SESSIONS or oos_sharpe < MIN_PROMOTABLE_OOS_SHARPE:
                logger.warning(
                    "SIGN-OFF REFUSED: pending candidate %s fails live-readiness validation "
                    "(alpha_significant=%s, n_sessions=%d, oos_sharpe=%.2f). Promotion blocked.",
                    pending_data.get("version", "unknown"),
                    alpha_significant, n_sessions, oos_sharpe,
                )
                return False, (
                    "Promotion blocked: candidate does not meet the live-readiness bar "
                    "(alpha_significant=True, n_sessions>=20, out_of_sample_sharpe>=1.5). "
                    "Rerun backtest + WFO and promote an approved candidate."
                )

            version_tag = pending_data.get("version", "unknown")
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            archived_path = ARCHIVE_DIR / f"config_v{ts_str}.yaml"

            if CONFIG_PATH.exists():
                shutil.copy(CONFIG_PATH, archived_path)

            shutil.copy(PENDING_PROMOTION_PATH, CONFIG_PATH)
            PENDING_PROMOTION_PATH.unlink()

            # Record manual sign-off in CSV
            with open(VERSION_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    version_tag,
                    "", "", "", "", "", "", "",
                    "HumanOperator",
                    "Approved and Promoted to Live Production",
                    "", "", "", "", "", "",
                ])

            logger.info("HUMAN SIGN-OFF: Successfully promoted pending configuration %s to live production.", version_tag)
            return True, f"Successfully promoted {version_tag} to live production."
        except Exception as e:
            logger.exception("Error applying pending promotion: %s", e)
            return False, f"Failed to apply pending promotion: {e}"


# Global instance
version_controller = VersionController()

if __name__ == "__main__":
    logger.info("Version Controller Ready. Audit log: %s", VERSION_CSV)
