"""
ML Training Pipeline — Algo-Trader Machine Learning Module.

Trains the calibrated dual-model ensemble (XGBoost + LightGBM) with zero-lookahead feature
engineering, configurable friction-adjusted labels (default friction=0.18%), scale_pos_weight
class imbalance scaling, and SHAP feature importance logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.ml.data_fetcher import (
    TRAINING_SYMBOLS,
    fetch_historical_candles_yfinance,
    fetch_india_vix,
    generate_synthetic_candles,
)
from src.ml.ensemble import ENSEMBLE_PATH, CalibratedEnsemble
from src.ml.feature_engineering import (
    FEATURE_COLS,
    engineer_extended_features,
    generate_friction_adjusted_labels,
)

logger = logging.getLogger(__name__)


def run_training_pipeline(
    use_synthetic_fallback: bool = True,
    target_pct: float = 2.0,
    stop_pct: float = 1.0,
    friction_pct: float = 0.18,
    forward_window: int = 20,
    save_path: Path = ENSEMBLE_PATH,
) -> CalibratedEnsemble:
    """
    Run the complete friction-aware calibrated ensemble training pipeline.
    """
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    logger.info("=" * 65)
    logger.info("Calibrated Ensemble Training Pipeline (Friction: %.2f%%)", friction_pct * 100)
    logger.info("=" * 65)

    # ── 1. Fetch historical candles ─────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []

    logger.info("[1/5] Fetching historical OHLCV candles...")
    for symbol, ticker in TRAINING_SYMBOLS.items():
        df = fetch_historical_candles_yfinance(ticker, interval="1m")
        if df is not None and len(df) > 50:
            logger.info("  %s: %d candles (yfinance)", symbol, len(df))
            all_frames.append(df)
        elif use_synthetic_fallback:
            ref_prices = {
                "NIFTY 50": 24500.0, "BANKNIFTY": 51500.0,
                "RELIANCE": 2890.0, "HDFCBANK": 1620.0, "INFY": 1780.0,
            }
            df_synth = generate_synthetic_candles(
                symbol=symbol,
                n=1000,
                base_price=ref_prices.get(symbol, 1000.0),
            )
            logger.info("  %s: %d synthetic candles (fallback)", symbol, len(df_synth))
            all_frames.append(df_synth)

    if not all_frames:
        raise RuntimeError("No training candles available.")

    df_all = pd.concat(all_frames, ignore_index=True)
    logger.info("  Total candles: %d", len(df_all))

    # ── 2. Fetch VIX ────────────────────────────────────────────────────────
    logger.info("[2/5] Fetching India VIX...")
    vix = fetch_india_vix()
    logger.info("  India VIX: %.2f", vix)

    # ── 3. Zero-Lookahead Feature Engineering ────────────────────────────────
    logger.info("[3/5] Engineering zero-lookahead features with Polars...")
    pl_feat = engineer_extended_features(df_all, vix=vix)
    df_feat = pl_feat.drop_nulls(subset=FEATURE_COLS).to_pandas()
    logger.info("  Features: %s", FEATURE_COLS)
    logger.info("  Clean rows: %d", len(df_feat))

    # ── 4. Friction-Adjusted Labeling ───────────────────────────────────────
    logger.info("[4/5] Generating labels (Target=+%.2f%%, Stop=-%.2f%%)...", target_pct + friction_pct, stop_pct + friction_pct)
    labels = generate_friction_adjusted_labels(
        df_feat,  # type: ignore[arg-type]
        target_pct=target_pct,
        stop_pct=stop_pct,
        friction_pct=friction_pct,
        forward_window=forward_window,
    )
    win_rate = labels.mean() * 100
    logger.info("  WIN=1: %d | LOSS=0: %d | Net Win Rate: %.1f%%", labels.sum(), (labels == 0).sum(), win_rate)

    # ── 5. Fit Calibrated Ensemble ───────────────────────────────────────────
    logger.info("[5/5] Fitting Calibrated Ensemble (XGBoost + LightGBM)...")
    X = df_feat[FEATURE_COLS].values.astype("float32")
    y = labels.astype(int)

    ensemble = CalibratedEnsemble(use_lightgbm=True)
    ensemble.fit(X, y, feature_names=FEATURE_COLS)
    ensemble.save(save_path)

    logger.info("=" * 65)
    logger.info("Training Complete! Model saved to: %s", save_path.resolve())
    logger.info("=" * 65)

    return ensemble


if __name__ == "__main__":
    run_training_pipeline()
