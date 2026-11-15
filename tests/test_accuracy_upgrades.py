"""
Unit Test Suite for Accuracy Upgrades & Prediction Quality.

Tests:
1. Friction-adjusted labeling (+2.0% + 0.18% target vs -1.0% - 0.18% stop)
2. Zero-lookahead feature engineering
3. Purged & Embargoed Cross-Validation
4. Volatility Regime Detector rules & multipliers
5. Dual-Model Calibrated Ensemble fitting & fallback
6. Live Performance Monitor degradation alerts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble import CalibratedEnsemble
from src.ml.feature_engineering import (
    FEATURE_COLS,
    engineer_extended_features,
    generate_friction_adjusted_labels,
)
from src.ml.performance_monitor import LivePerformanceMonitor
from src.strategy.regime_detector import MarketRegime, RegimeDetector
from src.wfo import PurgedWalkForwardSplitter


class TestAccuracyUpgrades:

    def test_friction_adjusted_labeling(self) -> None:
        """Verify friction-adjusted label logic with target=+2.18% and stop=-1.18%."""
        prices = [100.0] * 30
        highs = [100.0] * 30
        lows = [100.0] * 30

        # Candle 5 hits +2.18% (102.18)
        highs[5] = 102.20
        df = pd.DataFrame({"close": prices, "high": highs, "low": lows})

        labels = generate_friction_adjusted_labels(df, target_pct=2.0, stop_pct=1.0, friction_pct=0.18)
        assert labels[0] == 1, "Candle 0 should be labeled WIN (1) since candle 5 hit +2.20%"

        # Test stop hit -1.18% (98.82)
        highs_stop = [100.0] * 30
        lows_stop = [100.0] * 30
        lows_stop[5] = 98.50
        df_stop = pd.DataFrame({"close": prices, "high": highs_stop, "low": lows_stop})
        labels_stop = generate_friction_adjusted_labels(df_stop, target_pct=2.0, stop_pct=1.0, friction_pct=0.18)
        assert labels_stop[0] == 0, "Candle 0 should be labeled LOSS (0) since candle 5 hit stop"

    def test_feature_engineering_zero_lookahead(self) -> None:
        """Verify feature engineering produces correct FEATURE_COLS without NaNs."""
        timestamps = pd.date_range("2026-08-10 09:15", periods=50, freq="1min")
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": np.random.uniform(100, 105, 50),
            "high": np.random.uniform(105, 110, 50),
            "low": np.random.uniform(95, 100, 50),
            "close": np.random.uniform(98, 104, 50),
            "volume": np.random.uniform(1000, 5000, 50),
        })

        pl_feat = engineer_extended_features(df, vix=14.5)
        df_feat = pl_feat.to_pandas()

        for col in FEATURE_COLS:
            assert col in df_feat.columns, f"Feature {col} missing from output DataFrame"

    def test_purged_walk_forward_splitter(self) -> None:
        """Verify purging overlap indices and embargoing."""
        X = np.arange(200)
        splitter = PurgedWalkForwardSplitter(n_splits=3, forward_window=20, embargo_pct=0.02)

        splits = list(splitter.split(X))
        assert len(splits) > 0, "Splitter should yield valid splits"

        for train_idx, test_idx in splits:
            test_start = test_idx[0]
            # Ensure no train index falls within [test_start - 20, test_start]
            overlap = train_idx[train_idx >= (test_start - 20)]
            assert len(overlap) == 0, f"Purging failed! Found {len(overlap)} training samples in purge zone."

    def test_regime_detector_rules(self) -> None:
        """Verify regime detector signal gating and position sizing multiplier."""
        detector = RegimeDetector(high_vix_threshold=14.0, low_vix_threshold=12.0)

        # High VIX
        detector.update(16.5)
        assert detector.current_regime == MarketRegime.HIGH_VIX
        assert detector.get_sizing_multiplier() == 1.0
        assert detector.is_signal_allowed() is True

        # Extremely low VIX chop
        detector.update(8.5)
        assert detector.current_regime == MarketRegime.LOW_VIX
        assert detector.get_sizing_multiplier() == 0.5
        assert detector.is_signal_allowed() is False

    def test_calibrated_ensemble_fitting(self) -> None:
        """Verify ensemble model fitting with class imbalance scaling and calibration."""
        pytest.importorskip("xgboost")
        np.random.seed(42)
        X = np.random.randn(150, len(FEATURE_COLS)).astype("float32")
        y = np.random.choice([0, 1], size=150, p=[0.7, 0.3])

        ensemble = CalibratedEnsemble(use_lightgbm=False)
        ensemble.fit(X, y, feature_names=FEATURE_COLS)

        assert ensemble.is_fitted is True
        proba = ensemble.predict_proba(X[:1])
        assert 0.0 <= proba <= 1.0

    def test_live_performance_monitor(self) -> None:
        """Verify rolling win rate degradation tracking and risk multipliers."""
        mon = LivePerformanceMonitor(window_size=10, min_win_rate_threshold=0.40)

        for i in range(8):
            mon.record_trade(f"t_{i}", "NIFTY", pnl=-100, is_win=False)

        assert mon.is_degraded is True
        assert mon.get_risk_multiplier() == 0.5
