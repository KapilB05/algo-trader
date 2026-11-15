"""
Feature Engineering & Friction-Adjusted Label Generator — Algo-Trader ML Pipeline.

Implements zero-lookahead feature engineering (PDH/PDL, Opening Range, Relative Strength,
Realized Volatility, Cyclical Time Encoding, PCR Delta) and realistic transaction-friction
label generation (+2.0% target + friction vs -1.0% stop - friction).
"""

from __future__ import annotations

import logging
from math import pi, sqrt  # type: ignore[attr-defined]
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

logger = logging.getLogger(__name__)

# Primary feature columns list used across training and inference
FEATURE_COLS = [
    "vwap_dist_pct",
    "vwap_slope",
    "atr_pct",
    "atr_regime",
    "realized_vol_20",
    "vol_ratio",
    "volume_surge",
    "oi_change_pct",
    "oi_change_12",
    "rsi",
    "pdh_dist_pct",
    "pdl_dist_pct",
    "orh_dist_pct",
    "orl_dist_pct",
    "sin_time_of_day",
    "cos_time_of_day",
    "session_period",
    "relative_strength_index_pct",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Friction-Aware Label Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_friction_adjusted_labels(
    df_pd: pd.DataFrame,
    target_pct: float = 2.0,
    stop_pct: float = 1.0,
    friction_pct: float = 0.18,
    forward_window: int = 20,
) -> np.ndarray:
    """
    Simulate a 1:2 R:R trade for each candle with pure market price triggers and transaction friction.

    Market Price Triggers:
    - Target Price = entry * (1 + target_pct / 100)  (e.g., +2.00%)
    - Stop Price   = entry * (1 - stop_pct / 100)    (e.g., -1.00%)

    Account P&L after costs (used in evaluation):
    - Net Reward = target_pct - friction_pct
    - Net Risk   = stop_pct + friction_pct

    Args:
        df_pd: DataFrame with 'close', 'high', 'low' columns.
        target_pct: Gross target gain % (default 2.0%).
        stop_pct: Gross stop loss % (default 1.0%).
        friction_pct: Transaction friction + slippage % (default 0.18%).
        forward_window: Max candles to look ahead (default 20).

    Returns:
        np.ndarray of binary labels (0 or 1).
    """
    closes = df_pd["close"].to_numpy()
    highs = df_pd["high"].to_numpy()
    lows = df_pd["low"].to_numpy()
    n = len(closes)
    labels = np.zeros(n, dtype=int)

    target_mult = 1.0 + target_pct / 100.0
    stop_mult = 1.0 - stop_pct / 100.0

    for i in range(n - 1):
        entry = closes[i]
        if entry <= 0:
            continue
        target_price = entry * target_mult
        stop_price = entry * stop_mult

        for j in range(i + 1, min(i + forward_window + 1, n)):
            target_hit = highs[j] >= target_price
            stop_hit = lows[j] <= stop_price

            if target_hit and not stop_hit:
                labels[i] = 1
                break
            if stop_hit and not target_hit:
                labels[i] = 0
                break
            if target_hit and stop_hit:
                # Same candle hit target and stop — conservative label: LOSS
                labels[i] = 0
                break

    return labels


def generate_dynamic_atr_labels(
    df_pd: pd.DataFrame,
    target_atr_mult: float = 2.0,
    stop_atr_mult: float = 1.0,
    forward_window: int = 30,
    min_rr: float = 1.5,
    friction_pct: float = 0.03,
) -> tuple[np.ndarray, pd.Series, pd.Series]:
    """
    Simulate trade labeling using dynamic Wilder ATR(14) bounds on 5-minute bars.

    Rules:
    - Market Target Price = entry + (target_atr_mult * ATR_14)  (e.g., +2.0 * ATR)
    - Market Stop Price   = entry - (stop_atr_mult * ATR_14)    (e.g., -1.0 * ATR)
    - Enforces minimum R:R ratio floor: target_atr_mult / stop_atr_mult >= min_rr (default 1.5)
    - P&L side:
        WIN  -> +(target_distance_pct - friction_pct)
        LOSS -> -(stop_distance_pct + friction_pct)

    Args:
        df_pd: DataFrame with 'close', 'high', 'low' columns.
        target_atr_mult: Target multiplier of ATR(14) (default 2.0).
        stop_atr_mult: Stop multiplier of ATR(14) (default 1.0).
        forward_window: Max 5-minute candles to look ahead (default 30 = 2.5 hours).
        min_rr: Minimum Reward-to-Risk ratio floor (default 1.5).
        friction_pct: Transaction friction + slippage % (default 0.03%).

    Returns:
        tuple of (binary_labels_ndarray, net_reward_pct_series, net_risk_pct_series)
    """
    rr_ratio = target_atr_mult / max(1e-6, stop_atr_mult)
    if rr_ratio < min_rr:
        raise ValueError(
            f"Target/Stop multiplier ratio ({rr_ratio:.2f}) violates minimum R:R floor ({min_rr:.2f})"
        )

    # Compute Wilder's ATR(14)
    high = df_pd["high"]
    low = df_pd["low"]
    close_prev = df_pd["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close_prev).abs()
    tr3 = (low - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    closes = df_pd["close"].to_numpy()
    highs = df_pd["high"].to_numpy()
    lows = df_pd["low"].to_numpy()
    atrs = atr.to_numpy()
    n = len(closes)

    labels = np.zeros(n, dtype=int)
    net_rewards = np.zeros(n, dtype=float)
    net_risks = np.zeros(n, dtype=float)

    for i in range(n - 1):
        entry = closes[i]
        curr_atr = atrs[i]

        if entry <= 0 or np.isnan(curr_atr) or curr_atr <= 0:
            labels[i] = 0
            continue

        target_price = entry + (target_atr_mult * curr_atr)
        stop_price = entry - (stop_atr_mult * curr_atr)

        target_dist_pct = ((target_price - entry) / entry) * 100.0
        stop_dist_pct = ((entry - stop_price) / entry) * 100.0

        net_rewards[i] = target_dist_pct - friction_pct
        net_risks[i] = stop_dist_pct + friction_pct

        for j in range(i + 1, min(i + forward_window + 1, n)):
            target_hit = highs[j] >= target_price
            stop_hit = lows[j] <= stop_price

            if target_hit and not stop_hit:
                labels[i] = 1
                break
            if stop_hit and not target_hit:
                labels[i] = 0
                break
            if target_hit and stop_hit:
                labels[i] = 0
                break

    return labels, pd.Series(net_rewards, index=df_pd.index), pd.Series(net_risks, index=df_pd.index)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Zero-Lookahead Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def engineer_extended_features(
    df_pd: pd.DataFrame,
    index_df_pd: Optional[pd.DataFrame] = None,
    vix: float = 14.5,
    pcr_series: Optional[pd.Series] = None,
) -> pl.DataFrame:
    """
    Engineer comprehensive technical, time, and market structure features.

    Zero-lookahead guarantee:
      - PDH/PDL calculated strictly from prior session's candles.
      - Opening Range (ORH/ORL) calculated strictly from 9:15-9:30 AM candles.

    Returns:
      Polars DataFrame containing all FEATURE_COLS.
    """
    df = df_pd.copy()
    df["vix"] = float(vix)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if hasattr(df["timestamp"].dt, "tz") and df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # type: ignore[attr-defined]

    # Calculate PDH / PDL (Previous Day High / Low) using shift(1) per session date
    df["date"] = df["timestamp"].dt.date if "timestamp" in df.columns else pd.date_range("2026-01-01", periods=len(df), freq="1min").date  # type: ignore[attr-defined]

    daily_stats = df.groupby("date").agg(
        pdh=("high", "max"),
        pdl=("low", "min"),
    ).shift(1)  # Shift 1 day to strictly prevent lookahead

    df = df.merge(daily_stats, on="date", how="left")
    df["pdh"] = df["pdh"].bfill().ffill()
    df["pdl"] = df["pdl"].bfill().ffill()

    # Calculate Opening Range (ORH / ORL) from 9:15 - 9:30 AM IST
    if "timestamp" in df.columns:
        df["time_str"] = df["timestamp"].dt.strftime("%H:%M")  # type: ignore[attr-defined]
        or_candles = df[(df["time_str"] >= "09:15") & (df["time_str"] <= "09:30")]
        or_stats = or_candles.groupby("date").agg(
            orh=("high", "max"),
            orl=("low", "min"),
        )
        df = df.merge(or_stats, on="date", how="left")
        # Mask ORH/ORL before 9:30 AM to prevent lookahead
        df.loc[df["time_str"] < "09:30", ["orh", "orl"]] = np.nan
    else:
        df["orh"] = df["high"].expanding().max()
        df["orl"] = df["low"].expanding().min()

    # Calculate Wilder's ATR(14) & atr_pct & atr_regime (100-bar rolling percentile)
    high_s = df["high"]
    low_s = df["low"]
    close_prev = df["close"].shift(1)
    tr1 = high_s - low_s
    tr2 = (high_s - close_prev).abs()
    tr3 = (low_s - close_prev).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    df["atr_pct"] = ((atr_14 / df["close"]) * 100.0).fillna(0.0)

    atr_min100 = atr_14.rolling(window=100, min_periods=20).min()
    atr_max100 = atr_14.rolling(window=100, min_periods=20).max()
    df["atr_regime"] = ((atr_14 - atr_min100) / (atr_max100 - atr_min100 + 1e-9)).clip(0.0, 1.0).fillna(0.5)

    # Calculate Open Interest (OI) % change (4-bar / 20-min & 12-bar / 60-min)
    if "oi" in df.columns:
        oi_s = df["oi"].astype(float)
        oi_prev4 = oi_s.shift(4)
        oi_prev12 = oi_s.shift(12)
        df["oi_change_pct"] = (((oi_s - oi_prev4) / (oi_prev4.abs() + 1e-9)) * 100.0).fillna(0.0)
        df["oi_change_12"] = (((oi_s - oi_prev12) / (oi_prev12.abs() + 1e-9)) * 100.0).fillna(0.0)
    else:
        df["oi_change_pct"] = 0.0
        df["oi_change_12"] = 0.0

    # Session period (0: Morning 9:15-10:30, 1: Mid-day 10:30-14:15, 2: Closing 14:15-15:30)
    if "timestamp" in df.columns:
        t_str = df["timestamp"].dt.strftime("%H:%M")  # type: ignore[attr-defined]
        conds = [
            (t_str < "10:30"),
            (t_str >= "10:30") & (t_str < "14:15"),
            (t_str >= "14:15"),
        ]
        choices = [0.0, 1.0, 2.0]
        df["session_period"] = np.select(conds, choices, default=1.0)
    else:
        df["session_period"] = 1.0

    # Convert to Polars for columnar features
    pl_df = pl.from_pandas(df)  # type: ignore[arg-type]

    # ── 1. Typical price, VWAP & VWAP Slope (3-bar % change) ─────────────
    pl_df = pl_df.with_columns([
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("tp"),
    ])
    pl_df = pl_df.with_columns([
        (pl.col("tp") * pl.col("volume")).alias("tp_vol"),
        pl.col("volume").cum_sum().alias("cum_vol"),
    ])
    pl_df = pl_df.with_columns([
        pl.when(pl.col("cum_vol") > 0)
        .then(pl.col("tp_vol").cum_sum() / pl.col("cum_vol"))
        .otherwise(pl.col("tp").cum_sum() / (pl.cum_count("tp") + 1e-9))
        .alias("vwap"),
    ])
    pl_df = pl_df.with_columns([
        (((pl.col("close") - pl.col("vwap")) / (pl.col("vwap") + 1e-9)) * 100).alias("vwap_dist_pct"),
    ])

    vwap_s = pl_df["vwap"].to_pandas()
    vwap_prev3 = vwap_s.shift(3)
    vwap_slope_series = (((vwap_s - vwap_prev3) / (vwap_prev3 + 1e-9)) * 100.0).fillna(0.0)
    pl_df = pl_df.with_columns([pl.Series("vwap_slope", vwap_slope_series)])

    # ── 2. Volume ratio & Volume Surge ratio ──────────────────────────────
    vol_ma = pl.col("volume").rolling_mean(window_size=20, min_samples=1)
    pl_df = pl_df.with_columns([
        (pl.col("volume") / (vol_ma + 1.0)).alias("vol_ratio"),
        (pl.col("volume") / (vol_ma + 1.0)).alias("volume_surge"),
    ])

    # ── 3. RSI-14 ─────────────────────────────────────────────────────────
    delta = pl.col("close").diff()
    gain = delta.clip(lower_bound=0).alias("gain")
    loss = (-delta).clip(lower_bound=0).alias("loss")
    pl_df = pl_df.with_columns([gain, loss])
    pl_df = pl_df.with_columns([
        pl.col("gain").rolling_mean(window_size=14, min_samples=1).alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14, min_samples=1).alias("avg_loss"),
    ])
    pl_df = pl_df.with_columns([
        (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9)))).alias("rsi"),
    ])

    # ── 4. PDH / PDL distance % ───────────────────────────────────────────
    pl_df = pl_df.with_columns([
        (((pl.col("close") - pl.col("pdh")) / (pl.col("pdh") + 1e-9)) * 100).alias("pdh_dist_pct"),
        (((pl.col("close") - pl.col("pdl")) / (pl.col("pdl") + 1e-9)) * 100).alias("pdl_dist_pct"),
    ])

    # ── 5. Opening Range (ORH / ORL) distance % ─────────────────────────
    pl_df = pl_df.with_columns([
        pl.when(pl.col("orh").is_not_null())
        .then(((pl.col("close") - pl.col("orh")) / (pl.col("orh") + 1e-9)) * 100)
        .otherwise(0.0)
        .alias("orh_dist_pct"),
        pl.when(pl.col("orl").is_not_null())
        .then(((pl.col("close") - pl.col("orl")) / (pl.col("orl") + 1e-9)) * 100)
        .otherwise(0.0)
        .alias("orl_dist_pct"),
    ])

    # ── 6. Relative Strength vs Peer Index (12-bar return spread) ─────────
    if index_df_pd is not None and len(index_df_pd) == len(df_pd):
        idx_ret = index_df_pd["close"].pct_change(12).fillna(0.0).to_numpy()
        stock_ret = df_pd["close"].pct_change(12).fillna(0.0).to_numpy()
        rel_str = (stock_ret - idx_ret) * 100.0
        pl_df = pl_df.with_columns([pl.Series("relative_strength_index_pct", rel_str)])
    else:
        pl_df = pl_df.with_columns([pl.lit(0.0).alias("relative_strength_index_pct")])

    # ── 7. Realized Volatility (20-bar rolling) ───────────────────────────
    log_ret = (pl.col("close") / pl.col("close").shift(1)).log().fill_null(0.0)
    pl_df = pl_df.with_columns([log_ret.alias("_log_ret")])
    realized_vol = pl.col("_log_ret").rolling_std(window_size=20, min_samples=2).fill_null(0.0) * sqrt(75 * 252) * 100.0
    pl_df = pl_df.with_columns([realized_vol.alias("realized_vol_20")])

    # ── 8. Cyclical Intraday Time Features ───────────────────────────────
    if "timestamp" in df.columns:
        hours = df_pd["timestamp"].dt.hour.to_numpy()  # type: ignore[attr-defined]
        minutes = df_pd["timestamp"].dt.minute.to_numpy()  # type: ignore[attr-defined]
        mins_since_open = np.maximum(0, (hours - 9) * 60 + minutes - 15)
        angles = (mins_since_open / 375.0) * (2.0 * pi)
        pl_df = pl_df.with_columns([
            pl.Series("sin_time_of_day", np.sin(angles)),
            pl.Series("cos_time_of_day", np.cos(angles)),
        ])
    else:
        seq = np.linspace(0, 2 * pi, len(df_pd))
        pl_df = pl_df.with_columns([
            pl.Series("sin_time_of_day", np.sin(seq)),
            pl.Series("cos_time_of_day", np.cos(seq)),
        ])

    return pl_df
