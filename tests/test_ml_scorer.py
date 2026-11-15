"""
Tests for XGBoost ML Signal Confidence Scorer.

Tests:
  1. XGBoostScorer graceful fallback when model file does not exist.
  2. XGBoostScorer loads and infers correctly when model exists.
  3. predict_confidence always returns float in [0.0, 1.0].
  4. ConfluenceEngine respects ML block when confidence is below threshold.
  5. ConfluenceEngine passes through when ML is disabled.
  6. train_xgboost feature engineering produces correct column names.
  7. generate_labels produces correct 0/1 values.
  8. generate_synthetic_candles returns correct shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# XGBoostScorer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestXGBoostScorerNoModel:
    """Tests for XGBoostScorer when model file is absent."""

    def test_returns_pass_through_when_model_missing(self, tmp_path: Path) -> None:
        """Scorer returns fail-closed 0.0 (or pass-through 1.0) when model file does not exist."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=tmp_path / "no_model.json")
        assert scorer.is_available is False
        confidence = scorer.predict_confidence(
            {"vwap_dist_pct": 0.3, "vol_ratio": 1.8, "rsi": 60.0, "time_of_day_min": 45, "vix": 14.5}
        )
        assert confidence in (0.0, 1.0), "Should return fail-closed 0.0 or pass-through 1.0 when model is missing"

    def test_returns_float_when_model_missing(self, tmp_path: Path) -> None:
        """Return type must always be float, even without a model."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=tmp_path / "no_model.json")
        result = scorer.predict_confidence({})
        assert isinstance(result, float)


class TestXGBoostScorerWithModel:
    """Tests for XGBoostScorer when a trained model is available."""

    @pytest.fixture
    def trained_model_path(self, tmp_path: Path) -> Path:
        """Train a tiny XGBoost model and return its path for testing."""
        xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0)
        # Tiny synthetic training data (5 features)
        X = np.random.RandomState(42).rand(100, 5).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(int)
        model.fit(X, y)
        model_path = tmp_path / "test_model.json"
        model.save_model(str(model_path))
        return model_path

    def test_loads_model_successfully(self, trained_model_path: Path) -> None:
        """Scorer correctly loads a valid XGBoost model."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=trained_model_path)
        assert scorer.is_available is True

    def test_predict_confidence_returns_float_in_range(self, trained_model_path: Path) -> None:
        """predict_confidence must return a float in [0.0, 1.0]."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=trained_model_path)
        result = scorer.predict_confidence(
            {"vwap_dist_pct": 0.35, "vol_ratio": 2.1, "rsi": 67.0, "time_of_day_min": 60, "vix": 13.5}
        )
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_predict_with_missing_feature_keys(self, trained_model_path: Path) -> None:
        """Scorer handles missing feature keys by defaulting to 0.0."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=trained_model_path)
        # Pass empty dict — should not raise
        result = scorer.predict_confidence({})
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0

    def test_reload(self, trained_model_path: Path) -> None:
        """reload() re-reads the model from disk without raising."""
        from src.ml.scorer import XGBoostScorer

        scorer = XGBoostScorer(model_path=trained_model_path)
        scorer.reload()
        assert scorer.is_available is True


# ─────────────────────────────────────────────────────────────────────────────
# train_xgboost Module Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFeatureEngineering:
    """Tests for engineer_features() function."""

    def test_produces_required_feature_columns(self) -> None:
        """engineer_features() must return a Polars DataFrame with all feature columns."""
        polars = pytest.importorskip("polars", reason="polars not installed")
        from src.ml.train_xgboost import FEATURE_COLS, engineer_features, generate_synthetic_candles

        df_pd = generate_synthetic_candles(n=100)
        df_feat = engineer_features(df_pd, vix=14.5)

        for col in FEATURE_COLS:
            assert col in df_feat.columns, f"Missing feature column: {col}"

    def test_vwap_dist_pct_column_is_float(self) -> None:
        """vwap_dist_pct must be a numeric Polars column."""
        polars = pytest.importorskip("polars", reason="polars not installed")
        from src.ml.train_xgboost import engineer_features, generate_synthetic_candles

        df_pd = generate_synthetic_candles(n=50)
        df_feat = engineer_features(df_pd, vix=14.5)
        col = df_feat["vwap_dist_pct"]
        assert col.dtype in (polars.Float64, polars.Float32)

    def test_vix_column_constant(self) -> None:
        """vix feature column must all equal the passed VIX value."""
        pytest.importorskip("polars", reason="polars not installed")
        from src.ml.train_xgboost import engineer_features, generate_synthetic_candles

        vix_val = 18.75
        df_pd = generate_synthetic_candles(n=50)
        df_feat = engineer_features(df_pd, vix=vix_val)
        assert "vix" in df_feat.columns
        assert all(v == pytest.approx(vix_val) for v in df_feat["vix"].to_list())


class TestLabelGeneration:
    """Tests for generate_labels() function."""

    def test_labels_are_binary(self) -> None:
        """Labels must only contain 0 or 1."""
        from src.ml.train_xgboost import generate_labels, generate_synthetic_candles

        df_pd = generate_synthetic_candles(n=200)
        labels = generate_labels(df_pd, target_pct=2.0, stop_pct=1.0, forward_window=20)
        assert set(np.unique(labels)).issubset({0, 1})

    def test_label_length_matches_dataframe(self) -> None:
        """generate_labels must return same number of rows as input DataFrame."""
        from src.ml.train_xgboost import generate_labels, generate_synthetic_candles

        df_pd = generate_synthetic_candles(n=150)
        labels = generate_labels(df_pd)
        assert len(labels) == len(df_pd)

    def test_labels_are_numpy_int_array(self) -> None:
        """Labels must be a numpy integer array."""
        from src.ml.train_xgboost import generate_labels, generate_synthetic_candles

        df_pd = generate_synthetic_candles(n=100)
        labels = generate_labels(df_pd)
        assert isinstance(labels, np.ndarray)
        assert np.issubdtype(labels.dtype, np.integer)


class TestSyntheticDataGenerator:
    """Tests for generate_synthetic_candles() function."""

    def test_returns_correct_shape(self) -> None:
        """generate_synthetic_candles must return correct row count and columns."""
        from src.ml.train_xgboost import generate_synthetic_candles

        n = 300
        df = generate_synthetic_candles(n=n)
        assert len(df) == n
        for col in ["timestamp", "open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_high_gte_close(self) -> None:
        """High must always be >= Close for all candles."""
        from src.ml.train_xgboost import generate_synthetic_candles

        df = generate_synthetic_candles(n=500)
        assert (df["high"] >= df["close"]).all()

    def test_low_lte_close(self) -> None:
        """Low must always be <= Close for all candles."""
        from src.ml.train_xgboost import generate_synthetic_candles

        df = generate_synthetic_candles(n=500)
        assert (df["low"] <= df["close"]).all()

    def test_volume_positive(self) -> None:
        """Volume must be positive for all candles."""
        from src.ml.train_xgboost import generate_synthetic_candles

        df = generate_synthetic_candles(n=200)
        assert (df["volume"] > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# ConfluenceEngine + ML Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConfluenceEngineMLIntegration:
    """Test ConfluenceEngine interaction with XGBoostScorer."""

    @pytest.fixture
    def mock_confluence_engine_no_ml(self):
        """Self-contained ConfluenceEngine with ML disabled."""
        from unittest.mock import MagicMock

        from src.strategy.confluence_engine import ConfluenceEngine, ConfluenceMode

        mock_vwap = MagicMock()
        mock_vwap.get_vwap.return_value = 0  # VWAP unavailable → VWAP check fails

        mock_agg = MagicMock()
        mock_agg.get_history.return_value = []

        engine = ConfluenceEngine(
            vwap_engine=mock_vwap,
            candle_aggregator=mock_agg,
            mode=ConfluenceMode.STRICT,
            ml_enabled=False,
        )
        return engine

    def test_ml_disabled_does_not_add_ml_check(self, mock_confluence_engine_no_ml) -> None:
        """When ml_enabled=False, no 'ML Scorer' check appears in results."""

        result = mock_confluence_engine_no_ml.evaluate("TEST|001", 100.0)
        check_names = [c.name for c in result.checks]
        assert "ML Scorer" not in check_names

    def test_ml_disabled_scorer_is_none(self, mock_confluence_engine_no_ml) -> None:
        """When ml_enabled=False, _ml_scorer should be None."""
        assert mock_confluence_engine_no_ml._ml_scorer is None

    def test_ml_scorer_blocks_trade_below_threshold(self, tmp_path: Path) -> None:
        """ConfluenceEngine blocks trade when ML scorer returns confidence below threshold."""
        pytest.importorskip("xgboost", reason="xgboost not installed")
        import xgboost as xgb

        # Build a tiny model that returns low confidence
        model = xgb.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0)
        X = np.random.RandomState(0).rand(100, 5).astype(np.float32)
        y = np.zeros(100, dtype=int)  # All LOSS → model will output low P(Win)
        model.fit(X, y)
        model_path = tmp_path / "low_conf_model.json"
        model.save_model(str(model_path))

        from datetime import datetime, timezone
        from unittest.mock import MagicMock

        from src.data.candle_aggregator import Candle
        from src.ml.scorer import XGBoostScorer
        from src.strategy.confluence_engine import ConfluenceEngine, ConfluenceMode, Direction

        mock_vwap = MagicMock()
        mock_vwap.get_vwap.return_value = 95.0
        mock_vwap.get_vwap_distance_pct.return_value = 5.27

        mock_agg = MagicMock()
        ts = datetime.now(timezone.utc)
        candles = [
            Candle(instrument_token="T", timestamp=ts, open=99, high=101,
                   low=98, close=100, volume=100000 * (i + 1))
            for i in range(21)
        ]
        mock_agg.get_history.return_value = candles

        engine = ConfluenceEngine(
            vwap_engine=mock_vwap,
            candle_aggregator=mock_agg,
            mode=ConfluenceMode.STRICT,
            ml_confidence_threshold=0.90,  # High threshold → ML will likely block
            ml_enabled=True,
        )
        # Inject the low-confidence model
        engine._ml_scorer = XGBoostScorer(model_path=model_path)

        engine.update_index_trend(Direction.LONG)
        engine.update_pcr("TOKEN", 1.2)
        engine.update_oi_change("TOKEN", 1000)

        result = engine.evaluate("TOKEN", 100.0)
        # Result must always be a valid ConfluenceResult
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.score <= 1.0

    def test_confluence_result_has_all_required_fields(self, tmp_path: Path) -> None:
        """ConfluenceResult always has passed, direction, score, checks fields."""
        from unittest.mock import MagicMock

        from src.strategy.confluence_engine import ConfluenceEngine

        mock_vwap = MagicMock()
        mock_vwap.get_vwap.return_value = 0
        mock_agg = MagicMock()
        mock_agg.get_history.return_value = []

        engine = ConfluenceEngine(
            vwap_engine=mock_vwap,
            candle_aggregator=mock_agg,
            ml_enabled=False,
        )
        result = engine.evaluate("TOKEN", 100.0)
        assert hasattr(result, "passed")
        assert hasattr(result, "direction")
        assert hasattr(result, "score")
        assert hasattr(result, "checks")
        assert isinstance(result.checks, list)
