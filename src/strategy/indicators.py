"""
Technical Indicators — TA library wrappers on Pandas DataFrames.

Per spec: Pandas/NumPy for technical indicator calculation via the `ta` library.
Converts Polars candle output → Pandas for indicator computation → returns enriched DF.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
import polars as pl
import ta.momentum
import ta.trend
import ta.volatility

logger = logging.getLogger(__name__)


def polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars DataFrame of candles to Pandas for TA computation."""
    return df.to_pandas()  # pyrefly: ignore[bad-return]


def enrich_with_indicators(
    df: pd.DataFrame,
    ema_periods: Optional[list[int]] = None,
    rsi_period: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
    atr_period: int = 14,
    volume_ma_period: int = 20,
) -> pd.DataFrame:
    """
    Enrich a candle DataFrame with technical indicators.

    Input columns required: open, high, low, close, volume
    Returns the same DataFrame with indicator columns appended.
    """
    if df.empty or len(df) < 2:
        return df

    df = df.copy()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── EMA Ribbons ───────────────────────────────────────────────────
    if ema_periods is None:
        ema_periods = [9, 21, 50, 200]

    for period in ema_periods:
        if len(df) >= period:
            df[f"ema_{period}"] = ta.trend.ema_indicator(close, window=period)  # pyrefly: ignore[unsupported-operation, bad-argument-type]

    # ── RSI ───────────────────────────────────────────────────────────
    if len(df) >= rsi_period:
        df["rsi"] = ta.momentum.rsi(close, window=rsi_period)  # pyrefly: ignore[unsupported-operation, bad-argument-type]

    # ── MACD ──────────────────────────────────────────────────────────
    if len(df) >= macd_slow:
        macd = ta.trend.MACD(
            close, window_fast=macd_fast, window_slow=macd_slow, window_sign=macd_signal  # pyrefly: ignore[bad-argument-type]
        )
        df["macd"] = macd.macd()  # pyrefly: ignore[unsupported-operation]
        df["macd_signal"] = macd.macd_signal()  # pyrefly: ignore[unsupported-operation]
        df["macd_histogram"] = macd.macd_diff()  # pyrefly: ignore[unsupported-operation]

    # ── Bollinger Bands ───────────────────────────────────────────────
    if len(df) >= bb_period:
        bb = ta.volatility.BollingerBands(close, window=bb_period, window_dev=int(bb_std))  # pyrefly: ignore[bad-argument-type]
        df["bb_upper"] = bb.bollinger_hband()  # pyrefly: ignore[unsupported-operation]
        df["bb_middle"] = bb.bollinger_mavg()  # pyrefly: ignore[unsupported-operation]
        df["bb_lower"] = bb.bollinger_lband()  # pyrefly: ignore[unsupported-operation]
        df["bb_width"] = bb.bollinger_wband()  # pyrefly: ignore[unsupported-operation]

    # ── ATR ───────────────────────────────────────────────────────────
    if len(df) >= atr_period:
        df["atr"] = ta.volatility.average_true_range(
            high, low, close, window=atr_period
        )

    # ── Volume MA ─────────────────────────────────────────────────────
    df["volume_ma"] = volume.rolling(window=volume_ma_period, min_periods=1).mean()
    df["volume_surge"] = volume / df["volume_ma"]

    # ── ADX (Trend Strength) ──────────────────────────────────────────
    if len(df) >= 14:
        df["adx"] = ta.trend.adx(high, low, close, window=14)

    # ── Supertrend (simplified via ATR) ───────────────────────────────
    if "atr" in df.columns:
        df["supertrend_upper"] = (high + low) / 2 + 2 * df["atr"]
        df["supertrend_lower"] = (high + low) / 2 - 2 * df["atr"]

    return df


def compute_volume_surge(
    df: pd.DataFrame,
    multiplier: float = 1.5,
    lookback: int = 20,
) -> pd.Series:
    """
    Compute whether current volume exceeds the threshold.

    Returns a boolean Series: True where volume > multiplier × lookback MA.
    """
    volume_ma = df["volume"].rolling(window=lookback, min_periods=1).mean()
    return df["volume"] > (multiplier * volume_ma)


def compute_trend_direction(df: pd.DataFrame, ema_short: int = 9, ema_long: int = 21) -> str:
    """
    Determine trend direction from EMA crossover.

    Returns: 'BULLISH', 'BEARISH', or 'NEUTRAL'
    """
    if len(df) < ema_long:
        return "NEUTRAL"

    short_col = f"ema_{ema_short}"
    long_col = f"ema_{ema_long}"

    if short_col not in df.columns or long_col not in df.columns:
        df = enrich_with_indicators(df, ema_periods=[ema_short, ema_long])

    if short_col not in df.columns or long_col not in df.columns:
        return "NEUTRAL"

    latest_short = df[short_col].iloc[-1]
    latest_long = df[long_col].iloc[-1]

    if pd.isna(latest_short) or pd.isna(latest_long):
        return "NEUTRAL"

    if latest_short > latest_long:
        return "BULLISH"
    elif latest_short < latest_long:
        return "BEARISH"
    return "NEUTRAL"
