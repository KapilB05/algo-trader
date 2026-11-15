"""
Unit test suite for Part 2: Real Walk-Forward Optimization Automation (tests/test_wfo_automation.py).
"""

from pathlib import Path
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dashboard_v2.server import app
from src.wfo.wfo_engine import WFOEngine, WFOParameters

client = TestClient(app)

PARQUET_PATH = Path("data/banknifty_5m.parquet")


def _load_real_candles() -> list[dict]:
    """Load real candles from parquet for WFO testing."""
    if not PARQUET_PATH.exists():
        pytest.skip("banknifty_5m.parquet not available")
    df = pd.read_parquet(PARQUET_PATH)
    rows = df.to_dict(orient="records")
    for r in rows:
        if "timestamp" in r and not isinstance(r["timestamp"], str):
            r["timestamp"] = str(r["timestamp"])
        if "time" not in r and "timestamp" in r:
            r["time"] = r["timestamp"]
    return rows


def test_wfo_rolling_window_execution(tmp_path: Path) -> None:
    candles = _load_real_candles()
    if len(candles) < 300:
        pytest.skip("Insufficient candle data for WFO")

    store_file = tmp_path / "wfo_test_params.json"
    hist_file = tmp_path / "wfo_test_hist.json"

    engine = WFOEngine(store_path=store_file, history_path=hist_file)
    calibrated, report = engine.run_walk_forward_optimization(
        historical_candles=candles, window_days=90, n_splits=3,
    )

    assert report["status"] in ("SUCCESS", "NO_PASSING_WINDOW")
    assert "rolling_windows" in report
    assert len(report["rolling_windows"]) >= 1
    assert report["total_windows"] >= 1

    assert calibrated.deployment_mode == "PAPER"
    assert report["calibrated_parameters"]["deployment_mode"] == "PAPER"

    assert store_file.exists()
    assert hist_file.exists()


def test_wfo_parameters_hot_reload_and_promotion(tmp_path: Path) -> None:
    candles = _load_real_candles()
    if len(candles) < 300:
        pytest.skip("Insufficient candle data for WFO")

    store_file = tmp_path / "wfo_reload_params.json"
    hist_file = tmp_path / "wfo_reload_hist.json"

    engine = WFOEngine(store_path=store_file, history_path=hist_file)
    engine.run_walk_forward_optimization(
        historical_candles=candles, window_days=60, n_splits=2,
    )

    assert engine.active_parameters.deployment_mode == "PAPER"

    ok, msg = engine.promote_to_live()
    if engine.active_parameters.out_of_sample_sharpe >= 1.20:
        assert ok is True
        assert engine.active_parameters.deployment_mode == "LIVE"

        reloaded_engine = WFOEngine(store_path=store_file, history_path=hist_file)
        assert reloaded_engine.active_parameters.deployment_mode == "LIVE"
    else:
        assert ok is False
        assert engine.active_parameters.deployment_mode == "PAPER"


def test_wfo_api_endpoints() -> None:
    # 1. GET active parameters
    res_get = client.get("/api/wfo/parameters")
    assert res_get.status_code == 200
    data_get = res_get.json()
    assert data_get["status"] == "SUCCESS"
    assert "parameters" in data_get
    assert data_get["parameters"]["deployment_mode"] in ("PAPER", "LIVE")

    # 2. POST run WFO pipeline — may return 200 (success) or 503 (no data available)
    res_run = client.post("/api/wfo/run")
    assert res_run.status_code in (200, 503)
    data_run = res_run.json()
    if res_run.status_code == 200:
        assert data_run["status"] in ("SUCCESS", "NO_PASSING_WINDOW")
        if data_run["status"] == "SUCCESS":
            assert "calibrated_parameters" in data_run
            assert data_run["calibrated_parameters"]["deployment_mode"] == "PAPER"
        else:
            assert "message" in data_run

    # 3. POST hot reload
    res_reload = client.post("/api/wfo/reload")
    assert res_reload.status_code == 200
    assert res_reload.json()["status"] == "SUCCESS"

    # 4. POST promote to live — may return 200 or 400 if OOS Sharpe < 1.20
    res_promote = client.post("/api/wfo/promote")
    assert res_promote.status_code in (200, 400)
    if res_promote.status_code == 200:
        assert res_promote.json()["deployment_mode"] == "LIVE"
