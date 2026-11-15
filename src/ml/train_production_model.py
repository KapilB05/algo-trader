"""
Train Production Calibrated Ensemble Model for BankNifty 5-Minute Futures.

Persists the trained model artifact to data/banknifty_ensemble_5m.pkl.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.ensemble import CalibratedEnsemble
from src.ml.feature_engineering import (
    FEATURE_COLS,
    engineer_extended_features,
    generate_dynamic_atr_labels,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BANKNIFTY_PATH = Path("data/banknifty_5m.parquet")
MODEL_OUTPUT_PATH = Path("data/banknifty_ensemble_5m.pkl")


def train_and_save_model() -> None:
    if not BANKNIFTY_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {BANKNIFTY_PATH}. Please run data downloader first.")

    logger.info("Loading BankNifty 5-minute dataset from %s...", BANKNIFTY_PATH)
    df_raw = pd.read_parquet(BANKNIFTY_PATH)

    logger.info("Engineering 18 zero-lookahead features...")
    df_feat = engineer_extended_features(df_raw)

    # Convert Polars DataFrame to Pandas DataFrame if needed
    if hasattr(df_feat, "to_pandas"):
        df_feat = df_feat.to_pandas()

    logger.info("Generating Dynamic ATR labels (Target: 2.5x ATR, Stop: 1.0x ATR)...")
    labels, _, _ = generate_dynamic_atr_labels(
        df_feat,  # type: ignore[arg-type]
        target_atr_mult=2.5,
        stop_atr_mult=1.0,
        forward_window=30,
        min_rr=1.5,
        friction_pct=0.03,
    )
    df_feat["label"] = labels

    df_feat_pd: pd.DataFrame = df_feat  # type: ignore[assignment]
    X = df_feat_pd[FEATURE_COLS].copy()
    y = df_feat_pd["label"].copy()

    logger.info("Dataset shape: %s | Positive Win Labels: %d (%.2f%%)", X.shape, y.sum(), (y.mean() * 100))

    # Initialize Calibrated Ensemble
    ensemble = CalibratedEnsemble(use_lightgbm=True)

    logger.info("Fitting Calibrated Ensemble model (XGBoost + LightGBM)...")
    ensemble.fit(np.asarray(X), np.asarray(y), feature_names=FEATURE_COLS)

    # Save artifact
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(ensemble, f)

    logger.info("✅ Production model successfully trained and saved to %s", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    train_and_save_model()
