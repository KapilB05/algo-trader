"""
ML Scorer — Live Calibrated Ensemble & XGBoost Inference.

Provides real-time P(Win) confidence scoring for signal gating using the calibrated dual-model ensemble.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from src.ml.ensemble import ENSEMBLE_PATH, CalibratedEnsemble
from src.ml.feature_engineering import FEATURE_COLS

logger = logging.getLogger(__name__)


class XGBoostScorer:
    """
    Live inference scorer for signal confidence gating.

    Delegates to the calibrated dual-model ensemble (XGBoost + LightGBM)
    while maintaining 100% backward compatibility with existing code.
    """

    def __init__(self, model_path: Path = ENSEMBLE_PATH) -> None:
        self.model_path = model_path
        self._ensemble: Optional[CalibratedEnsemble] = None
        self._available: bool = False
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load calibrated ensemble from disk."""
        if not self.model_path.exists():
            if self.model_path == ENSEMBLE_PATH:
                legacy_path = Path("data/xgboost_model.json")
                if legacy_path.exists():
                    logger.info("Loading legacy XGBoost model from %s", legacy_path)
                    try:
                        import xgboost as xgb
                        model = xgb.XGBClassifier()
                        model.load_model(str(legacy_path))
                        self._legacy_model = model
                        self._available = True
                        return
                    except Exception as exc:
                        logger.error("Failed loading legacy XGBoost model: %s", exc)

            logger.warning("ML Ensemble model not found at %s.", self.model_path)
            self._available = False
            return

        ensemble = CalibratedEnsemble.load(self.model_path)
        if ensemble is not None and ensemble.is_fitted:
            self._ensemble = ensemble
            self._available = True
            logger.info("Calibrated Ensemble scorer loaded from %s", self.model_path)
        else:
            self._available = False

    def predict_confidence(self, feature_dict: dict) -> float:
        """
        Predict calibrated P(Win) confidence score for a given trade setup.

        Args:
            feature_dict: Feature dictionary containing FEATURE_COLS keys.

        Returns:
            float: P(Win) probability score in [0.0, 1.0].
                    Returns 1.0 ONLY when the model is intentionally
                    unavailable/disabled (gate off).
                    Inference ERRORS fail closed with 0.0 so a broken model
                    can never wave every signal through.
        """
        if not self._available:
            logger.debug("ML scorer unavailable — ML gate ACTIVE (pass-through blocked by max-conviction policy).")
            return 0.0

        try:
            features = np.array(
                [[float(feature_dict.get(col, 0.0)) for col in FEATURE_COLS]],
                dtype=np.float32,
            )

            if self._ensemble is not None:
                proba = self._ensemble.predict_proba(features)
                if isinstance(proba, np.ndarray):
                    return float(proba[0]) if proba.ndim > 0 else float(proba)
                return float(proba)

            if hasattr(self, "_legacy_model") and self._legacy_model is not None:
                # Fallback to legacy 5-feature XGBoost
                legacy_cols = ["vwap_dist_pct", "vol_ratio", "rsi", "time_of_day_min", "vix"]
                leg_features = np.array(
                    [[float(feature_dict.get(col, 0.0)) for col in legacy_cols]],
                    dtype=np.float32,
                )
                proba = self._legacy_model.predict_proba(leg_features)[0][1]
                return float(np.clip(proba, 0.0, 1.0))

            return 0.0
        except Exception as exc:
            logger.error("Inference failed: %s — failing CLOSED (confidence=0.0)", exc)
            return 0.0

    @property
    def is_available(self) -> bool:
        return self._available

    def reload(self) -> None:
        """Hot-reload the model from disk."""
        self._load_model()
