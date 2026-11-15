"""
Continuous Auto-Learner — Master Pipeline Scheduler & Coordinator.

Orchestrates the entire automated self-optimization cycle:
1. Historical Data Extraction & Parquet Ingestion
2. Rolling Walk-Forward Grid Search & Parameter Ranking
3. Machine Learning (XGBoost & RL) Model Retraining
4. Shadow Mode Safety Validation against Live Configuration
5. Atomic Version Archival & Promotion
6. Recurring Sunday 10:00 AM IST scheduling
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

from src.auto_learner.data_extractor import data_extractor
from src.auto_learner.ml_retrainer import ml_retrainer
from src.auto_learner.optimization_runner import optimization_runner
from src.auto_learner.shadow_evaluator import shadow_evaluator
from src.auto_learner.version_controller import version_controller

IST = ZoneInfo("Asia/Kolkata")
LOGS_DIR = Path("logs")
AUTO_LEARN_LOG = LOGS_DIR / "auto_learning.log"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("AutoLearner.Scheduler")
logger.setLevel(logging.INFO)

# Setup log handler for auto-learning
file_handler = logging.FileHandler(AUTO_LEARN_LOG, encoding="utf-8")
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class AutoLearnerScheduler:
    """
    Automated Weekly Self-Optimization Coordinator.
    """

    def run_full_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete end-to-end auto-learning and optimization cycle.
        """
        start_ts = datetime.now(IST)
        logger.info("=" * 80)
        logger.info("🏛️ AUTO-LEARNER PIPELINE EXECUTION STARTED — %s", start_ts.strftime("%Y-%m-%d %H:%M:%S IST"))
        logger.info("=" * 80)

        results: Dict[str, Any] = {
            "start_time": start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "data_updated": False,
            "new_bars": 0,
            "ml_retrained": False,
            "ml_promoted": False,
            "candidate_evaluated": False,
            "shadow_approved": False,
            "config_promoted": False,
        }

        # ── STEP 1: DATA EXTRACTION & INGESTION ───────────────────────────────
        logger.info("[STEP 1/5] Extracting & Merging Latest Historical Data...")
        try:
            df_merged, new_count = data_extractor.extract_and_merge()
            results["data_updated"] = True
            results["new_bars"] = new_count
            logger.info("Step 1 Data Extractor: %d total bars (+%d new).", len(df_merged), new_count)
        except Exception as e:
            logger.error("Step 1 Data Extractor failed: %s", e, exc_info=True)

        # ── STEP 2: ML & RL RETRAINING ────────────────────────────────────────
        logger.info("[STEP 2/5] Retraining Machine Learning Classifiers & RL Exit Policy...")
        try:
            ml_promoted, train_auc, val_auc, model_path = ml_retrainer.retrain_xgboost_classifier()
            ml_retrainer.retrain_rl_exit_agent()
            results["ml_retrained"] = True
            results["ml_promoted"] = ml_promoted
            logger.info("Step 2 ML Retraining: Train AUC=%.3f, Val AUC=%.3f, Promoted=%s", train_auc, val_auc, ml_promoted)
        except Exception as e:
            logger.error("Step 2 ML Retraining failed: %s", e, exc_info=True)

        # ── STEP 3: WALK-FORWARD OPTIMIZATION GRID SEARCH ──────────────────────
        logger.info("[STEP 3/5] Running Rolling Walk-Forward Parameter Optimization...")
        try:
            best_cand, all_cands = optimization_runner.run_walk_forward_optimization()
            results["candidate_evaluated"] = True
            results["best_candidate"] = {
                "sl_atr_mult": best_cand.sl_atr_mult,
                "tp_atr_mult": best_cand.tp_atr_mult,
                "min_vix_gate": best_cand.min_vix_gate,
                "consensus_rule": best_cand.consensus_rule,
                "oos_sharpe": best_cand.out_of_sample_sharpe,
                "oos_pf": best_cand.out_of_sample_pf,
                "is_promotable": best_cand.is_promotable,
            }
            logger.info("Step 3 Optimization: Top candidate OOS Sharpe=%.2f, PF=%.2f, Improvement=%.1f%%, Promotable=%s",
                        best_cand.out_of_sample_sharpe, best_cand.out_of_sample_pf, best_cand.improvement_pct, best_cand.is_promotable)
        except Exception as e:
            logger.error("Step 3 Walk-Forward Optimization failed: %s", e, exc_info=True)
            return results

        # ── STEP 4: SHADOW MODE SAFETY VALIDATION ─────────────────────────────
        logger.info("[STEP 4/5] Running Shadow Mode Safety Verification on Recent 3 Months...")
        cand_dict = {
            "sl_atr_mult": best_cand.sl_atr_mult,
            "tp_atr_mult": best_cand.tp_atr_mult,
            "min_vix_gate": best_cand.min_vix_gate,
            "consensus_rule": best_cand.consensus_rule,
        }
        shadow_res = shadow_evaluator.evaluate_candidate(cand_dict)
        results["shadow_approved"] = shadow_res.approved
        logger.info("Step 4 Shadow Check: Approved=%s, Reason=%s (Live Sharpe=%.2f vs Candidate Sharpe=%.2f)",
                    shadow_res.approved, shadow_res.reason, shadow_res.live_sharpe, shadow_res.candidate_sharpe)

        # ── STEP 5: CONFIGURATION VERSIONING & DEPLOYMENT ─────────────────────
        logger.info("[STEP 5/5] Parameter Promotion & Version Control...")
        if best_cand.is_promotable and shadow_res.approved:
            promo_ok, ver_tag, archive_p = version_controller.promote_new_configuration(
                new_params=cand_dict,
                old_sharpe=shadow_res.live_sharpe,
                new_sharpe=shadow_res.candidate_sharpe,
                improvement_pct=shadow_res.improvement_sharpe_pct,
                alpha_annual_pct=shadow_res.candidate_alpha_annual_pct,
                information_ratio=shadow_res.candidate_information_ratio,
                excess_return_pct=shadow_res.candidate_excess_return_pct,
                bench_return_pct=shadow_res.benchmark_total_return_pct,
                n_sessions=shadow_res.n_sessions,
                significant=shadow_res.candidate_significant,
                benchmark=shadow_res.benchmark,
                reason="Weekly Walk-Forward Auto-Learner Promotion",
            )
            results["config_promoted"] = promo_ok
            results["promotion"] = {
                "version": ver_tag,
                "alpha_annual_pct": shadow_res.candidate_alpha_annual_pct,
                "excess_return_pct": shadow_res.candidate_excess_return_pct,
                "bench_return_pct": shadow_res.benchmark_total_return_pct,
                "n_sessions": shadow_res.n_sessions,
                "significant": shadow_res.candidate_significant,
            }
            logger.info("Step 5 Deployment: Promoted version %s to production (Archived to %s). Alpha=%.2f%%/yr, excess=%.2f%%, significant=%s",
                        ver_tag, archive_p, shadow_res.candidate_alpha_annual_pct,
                        shadow_res.candidate_excess_return_pct, shadow_res.candidate_significant)
        else:
            logger.info("Step 5 Deployment: Current live config maintained (No promotion triggered). "
                        "promotable=%s shadow_approved=%s",
                        best_cand.is_promotable, shadow_res.approved)

        end_ts = datetime.now(IST)
        duration_sec = (end_ts - start_ts).total_seconds()
        logger.info("=" * 80)
        logger.info("AUTO-LEARNER CYCLE FINISHED IN %.1f SECONDS — Logged to %s", duration_sec, AUTO_LEARN_LOG)
        logger.info("=" * 80)

        return results

    def start_weekly_schedule(self) -> None:
        """
        Run continuous background scheduler firing every Sunday at 10:00 AM IST.
        """
        logger.info("Auto-Learner Scheduler Active. Scheduled: Every Sunday at 10:00 AM IST.")

        while True:
            now = datetime.now(IST)
            # Sunday = weekday 6, 10:00 AM
            if now.weekday() == 6 and now.hour == 10 and now.minute == 0:
                logger.info("Sunday 10:00 AM Auto-Learner Trigger Activated!")
                self.run_full_cycle()
                time.sleep(65)  # Avoid duplicate trigger in same minute

            time.sleep(30)


# Global instance
auto_learner_scheduler = AutoLearnerScheduler()

if __name__ == "__main__":
    auto_learner_scheduler.run_full_cycle()
