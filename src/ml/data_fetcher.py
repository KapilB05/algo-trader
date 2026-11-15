"""
Market Data & Synthetic Candles Helper for ML Training & Scorer.
"""

from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from src.ml.feature_engineering import (
    FEATURE_COLS,  # noqa: F401  (legacy re-export for tests)
    engineer_extended_features,
    generate_friction_adjusted_labels,
)

logger = logging.getLogger(__name__)

# Legacy aliases for tests
generate_labels = generate_friction_adjusted_labels
engineer_features_polars = engineer_extended_features
engineer_features = engineer_extended_features

TRAINING_SYMBOLS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "RELIANCE": "RELIANCE.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "INFY": "INFY.NS",
}


def fetch_historical_candles_yfinance(
    ticker_symbol: str,
    days: int = 7,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch 1-minute historical candles from yfinance."""
    try:
        import yfinance as yf

        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)

        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(start=start_dt, end=end_dt, interval=interval)
        if df.empty:
            logger.warning("yfinance returned empty DataFrame for %s.", ticker_symbol)
            return pd.DataFrame()

        df = df.reset_index()
        col_map = {
            "Datetime": "timestamp",
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = df.rename(columns={c: col_map[c] for c in df.columns if c in col_map})
        keep_cols = ["timestamp", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep_cols if c in df.columns]]
        logger.info("Fetched %d candles for %s from yfinance.", len(df), ticker_symbol)
        return df

    except Exception as exc:
        logger.error("Failed to fetch yfinance data for %s: %s", ticker_symbol, exc)
        return pd.DataFrame()


def fetch_india_vix() -> float:
    """Fetch current India VIX value from yfinance."""
    try:
        import yfinance as yf
        vix_ticker = yf.Ticker("^INDIAVIX")
        hist = vix_ticker.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as exc:
        logger.warning("Could not fetch India VIX: %s. Using default 14.5.", exc)
    return 14.5


def generate_synthetic_candles(
    n_candles: int = 1000,
    n: Optional[int] = None,
    symbol: str = "SYNTH",
    base_price: float = 24000.0,
) -> pd.DataFrame:
    """Generate realistic synthetic 1-minute OHLCV candles for training/testing."""
    if n is not None:
        n_candles = n
    logger.debug("Generating %d synthetic candles for %s @ base %.2f", n_candles, symbol, base_price)

    random.seed(42)
    np.random.seed(42)

    now = datetime.now(timezone.utc)
    timestamps = [now - timedelta(minutes=n_candles - i) for i in range(n_candles)]

    prices = [base_price]
    for _ in range(n_candles - 1):
        change = np.random.normal(0, 15.0)
        prices.append(max(1000.0, prices[-1] + change))

    highs, lows, opens, closes, volumes = [], [], [], [], []
    for p in prices:
        spread = abs(np.random.normal(5, 3))
        o = p + np.random.normal(0, 2)
        c = p
        h = max(o, c) + spread
        l = min(o, c) - spread
        v = int(abs(np.random.normal(5000, 2000))) + 100

        opens.append(round(o, 2))
        highs.append(round(h, 2))
        lows.append(round(l, 2))
        closes.append(round(c, 2))
        volumes.append(v)

    # Dynamic mean-reverting VIX (Mean = 15.0, Std ~ 2.5)
    vix_vals = [14.5]
    for _ in range(n_candles - 1):
        v_next = vix_vals[-1] + 0.05 * (15.0 - vix_vals[-1]) + np.random.normal(0, 0.25)
        vix_vals.append(round(float(np.clip(v_next, 10.5, 28.0)), 2))

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
        "vix": vix_vals,
    })
