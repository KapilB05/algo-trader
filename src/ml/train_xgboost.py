"""
XGBoost & Dual-Model Calibrated Ensemble Training Entry Point.

Delegates directly to src.ml.train_pipeline.run_training_pipeline.
"""

from __future__ import annotations

import logging

# Public re-export shim: legacy consumers (tests, notebooks) import these
# names from this module; they are re-exported from the data layer.
from src.ml.data_fetcher import (  # noqa: F401
    FEATURE_COLS,
    TRAINING_SYMBOLS,
    engineer_features,
    engineer_features_polars,
    fetch_historical_candles_yfinance,
    fetch_india_vix,
    generate_labels,
    generate_synthetic_candles,
)
from src.ml.train_pipeline import run_training_pipeline

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logger.info("Executing training pipeline via train_xgboost entry point...")
    run_training_pipeline()


if __name__ == "__main__":
    main()
