"""
Dual-Model Calibrated Ensemble — Algo-Trader ML Scorer Module.

Combines XGBoost & LightGBM with Platt scaling / CalibratedClassifierCV probability calibration,
scale_pos_weight class imbalance handling, and SHAP tree feature importance logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import joblib
except ImportError:
    import pickle as joblib

try:
    from sklearn.calibration import CalibratedClassifierCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

HAS_SHAP = False
try:
    import shap  # type: ignore[import-untyped]
    HAS_SHAP = True
except ImportError:
    shap = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

ENSEMBLE_PATH = Path("data/ensemble_model.joblib")
ENSEMBLE_PATH.parent.mkdir(parents=True, exist_ok=True)


class CalibratedEnsemble:
    """
    Dual-Model Calibrated Ensemble Classifier combining XGBoost and LightGBM.
    """

    def __init__(self, use_lightgbm: bool = True) -> None:
        self.use_lightgbm = use_lightgbm
        self.xgb_model: Optional[object] = None
        self.lgbm_model: Optional[object] = None
        self.is_fitted: bool = False
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> None:
        """Fit calibrated XGBoost and LightGBM models with class imbalance scaling."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.feature_names = feature_names

        n_pos = int(np.sum(y_arr == 1))
        n_neg = int(np.sum(y_arr == 0))
        scale_pos_weight = float(n_neg / max(1, n_pos))
        logger.info(
            "Training Calibrated Ensemble | Pos: %d, Neg: %d | scale_pos_weight: %.2f",
            n_pos, n_neg, scale_pos_weight,
        )

        # ── 1. Base XGBoost ───────────────────────────────────────────────────
        if HAS_XGBOOST:
            try:
                base_xgb = XGBClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric="logloss",
                    random_state=42,
                    verbosity=0,
                )
                base_xgb.fit(X_arr, y_arr)
                if HAS_SKLEARN:
                    self.xgb_model = CalibratedClassifierCV(
                        estimator=base_xgb, method="sigmoid", cv=3
                    )
                    self.xgb_model.fit(X_arr, y_arr)
                else:
                    self.xgb_model = base_xgb

                self._log_shap_importance(base_xgb, X_arr, feature_names, model_name="XGBoost")
                logger.info("XGBoost calibrated model fitted successfully.")
            except Exception as exc:
                logger.error("Failed to fit XGBoost model: %s", exc)

        # ── 2. Base LightGBM ──────────────────────────────────────────────────
        if self.use_lightgbm and HAS_LIGHTGBM:
            try:
                base_lgbm = LGBMClassifier(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbosity=-1,
                )
                base_lgbm.fit(X_arr, y_arr)
                if HAS_SKLEARN:
                    self.lgbm_model = CalibratedClassifierCV(
                        estimator=base_lgbm, method="sigmoid", cv=3
                    )
                    self.lgbm_model.fit(X_arr, y_arr)
                else:
                    self.lgbm_model = base_lgbm

                self._log_shap_importance(base_lgbm, X_arr, feature_names, model_name="LightGBM")
                logger.info("LightGBM calibrated model fitted successfully.")
            except Exception as exc:
                logger.warning("Failed to fit LightGBM model: %s", exc)

        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray | float:
        """Compute consensus ensemble probability P(WIN)."""
        if not self.is_fitted:
            return 1.0 if np.ndim(X) == 1 else np.ones(len(X))

        is_single = np.ndim(X) == 1
        X_arr = np.atleast_2d(X)

        probs_list = []

        if self.xgb_model is not None and hasattr(self.xgb_model, "predict_proba"):
            p_xgb = self.xgb_model.predict_proba(X_arr)[:, 1]
            probs_list.append(p_xgb)

        if self.lgbm_model is not None and hasattr(self.lgbm_model, "predict_proba"):
            p_lgbm = self.lgbm_model.predict_proba(X_arr)[:, 1]
            probs_list.append(p_lgbm)

        if probs_list:
            consensus = np.mean(probs_list, axis=0)
        else:
            consensus = np.full(len(X_arr), 0.5)

        return float(consensus[0]) if is_single else consensus

    def predict_consensus(self, X: np.ndarray, threshold: float = 0.65) -> Tuple[bool, float]:
        """Check if consensus probability exceeds threshold."""
        p_ensemble = float(self.predict_proba(X)) if np.ndim(X) == 1 else self.predict_proba(X)
        return bool(p_ensemble >= threshold), float(p_ensemble)

    def _log_shap_importance(
        self, base_model: object, X_sample: np.ndarray, feature_names: list[str], model_name: str = "Model"
    ) -> None:
        """Compute and log true SHAP feature importances from tree models."""
        if not HAS_SHAP:
            return
        assert shap is not None  # narrows type from module | None
        try:
            explainer = shap.TreeExplainer(base_model)
            shap_vals = explainer.shap_values(X_sample[:300])

            if isinstance(shap_vals, list):
                shap_abs = np.abs(shap_vals[1]).mean(axis=0)
            elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                shap_abs = np.abs(shap_vals[:, :, 1]).mean(axis=0)
            else:
                shap_abs = np.abs(shap_vals).mean(axis=0)

            sorted_imp = sorted(zip(feature_names, shap_abs), key=lambda x: x[1], reverse=True)

            logger.info("=== Tree SHAP Feature Importance (%s) ===", model_name)
            print(f"\n  Real Tree SHAP Feature Importance ({model_name}):")
            for name, val in sorted_imp:
                print(f"    - {name:<28}: {val:.4f}")
            print("")
        except Exception as exc:
            logger.debug("SHAP importance computation skipped/failed for %s: %s", model_name, exc)

    def save(self, path: Path = ENSEMBLE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            joblib.dump(self, f)
        logger.info("Ensemble model saved to %s", path)

    @classmethod
    def load(cls, path: Path = ENSEMBLE_PATH) -> "CalibratedEnsemble":
        path_obj = Path(path)
        if not path_obj.exists():
            logger.warning("No ensemble model checkpoint found at %s. Returning uninitialized.", path)
            return cls()

        if path_obj.suffix.lower() == ".json":
            try:
                ensemble = cls()
                if HAS_XGBOOST:
                    from xgboost import XGBClassifier
                    base_xgb = XGBClassifier()
                    base_xgb.load_model(str(path_obj))
                    ensemble.xgb_model = base_xgb
                    ensemble.is_fitted = True
                    return ensemble
            except Exception as exc:
                logger.warning("Failed to load raw JSON XGBoost model: %s", exc)

        try:
            with open(path_obj, "rb") as f:
                model = joblib.load(f)
            logger.info("Loaded CalibratedEnsemble model from %s", path_obj)
            return model
        except Exception as exc:
            logger.warning("joblib.load failed for %s: %s. Returning uninitialized.", path_obj, exc)
            return cls()
