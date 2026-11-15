"""
Unit Tests for 80+ Feature Engineering Pipeline (Layer 2).
"""

import numpy as np
import pandas as pd
import pytest
from src.ml.extended_features import EXTENDED_FEATURE_COLS, engineer_80_features


def test_80_features_output_shape():
    """Verify feature matrix produces exactly 80 columns without NaNs."""
    np.random.seed(42)
    n_bars = 100
    dates = pd.date_range("2026-08-28 09:15", periods=n_bars, freq="5min")
    close = 50000.0 + np.cumsum(np.random.randn(n_bars) * 25.0)
    high = close + np.random.rand(n_bars) * 30.0
    low = close - np.random.rand(n_bars) * 30.0
    open_p = low + np.random.rand(n_bars) * (high - low)
    vol = np.random.randint(1000, 50000, size=n_bars)

    df = pd.DataFrame({
        "timestamp": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })

    feat_df = engineer_80_features(df, vix_level=13.5, pcr_value=1.15)

    assert len(EXTENDED_FEATURE_COLS) == 80
    assert feat_df.shape == (n_bars, 80)
    assert not feat_df.isna().any().any(), "Feature matrix contains NaN values"


def test_zero_lookahead_stability():
    """Verify historical feature rows do not change when new future candles arrive."""
    np.random.seed(42)
    n_initial = 80
    df_initial = pd.DataFrame({
        "open": 50000.0 + np.cumsum(np.random.randn(n_initial) * 10),
        "high": 50050.0 + np.cumsum(np.random.randn(n_initial) * 10),
        "low": 49950.0 + np.cumsum(np.random.randn(n_initial) * 10),
        "close": 50020.0 + np.cumsum(np.random.randn(n_initial) * 10),
        "volume": np.random.randint(5000, 20000, size=n_initial),
    })

    feat_1 = engineer_80_features(df_initial)

    # Append 5 new future bars
    n_new = 5
    df_extended = pd.concat([df_initial, pd.DataFrame({
        "open": [50100.0] * n_new,
        "high": [50150.0] * n_new,
        "low": [50080.0] * n_new,
        "close": [50120.0] * n_new,
        "volume": [15000] * n_new,
    })], ignore_index=True)

    feat_2 = engineer_80_features(df_extended)

    # Values at index 50 should be unchanged
    np.testing.assert_allclose(
        feat_1.iloc[50]["ema_9_dist_pct"],
        feat_2.iloc[50]["ema_9_dist_pct"],
        rtol=1e-4,
    )
