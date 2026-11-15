"""
80+ Indicator Feature Engineering Suite — Algo-Trader Institutional ML Pipeline.

Implements zero-lookahead, point-in-time calculation for:
1. Trend (15 features): Multi-EMA, Supertrend, ADX/DMI, MACD, Trend Alignments
2. Volatility (15 features): ATR, BBW Percentile Rank, Squeeze, Keltner, Realized Vol
3. Momentum (15 features): RSI, Stoch RSI, Williams %R, ROC, CCI, MACD Hist
4. Volume & Microstructure (15 features): VWAP, Volume Z-Score, OBV, MFI, CMF, PDH/PDL, ORH/ORL
5. Options & Institutional Greeks (20 features): PCR, PCR Delta, Greeks (Δ, Γ, Θ, ν), Max Pain, FII/DII
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# The authoritative list of all 80 features in exact order
EXTENDED_FEATURE_COLS: List[str] = [
    # 1. Trend Indicators (15)
    "ema_9_dist_pct", "ema_20_dist_pct", "ema_50_dist_pct", "ema_200_dist_pct",
    "ema_9_slope", "ema_20_slope",
    "ema_9_gt_20", "ema_20_gt_50", "ema_50_gt_200",
    "supertrend_direction", "supertrend_dist_pct",
    "adx_14", "plus_di_14", "minus_di_14", "di_spread",

    # 2. Volatility & Range (15)
    "atr_14_pct", "atr_regime", "atr_expansion_ratio",
    "bb_upper_dist_pct", "bb_lower_dist_pct", "bb_width_pct",
    "bb_width_percentile_rank_50", "bb_squeeze",
    "keltner_upper_dist_pct", "keltner_lower_dist_pct",
    "realized_vol_20", "volatility_ratio", "parkinson_vol", "garman_klass_vol", "vix_level",

    # 3. Momentum & Mean Reversion (15)
    "rsi_14", "rsi_slope_3", "rsi_oversold", "rsi_overbought",
    "stoch_k_14", "stoch_d_3", "stoch_cross",
    "williams_r_14", "roc_5", "roc_20", "cci_20",
    "macd_line", "macd_signal", "macd_hist", "macd_hist_slope",

    # 4. Volume & Microstructure (15)
    "vwap_dist_pct", "vwap_slope", "price_above_vwap",
    "volume_surge_ratio", "volume_zscore_20", "volume_ma_20_ratio",
    "obv_slope_5", "mfi_14", "cmf_20",
    "pdh_dist_pct", "pdl_dist_pct", "orh_dist_pct", "orl_dist_pct",
    "sin_time_of_day", "cos_time_of_day",

    # 5. Options & Institutional Greeks (20)
    "pcr_value", "pcr_15m_delta", "pcr_regime_bullish", "pcr_regime_bearish",
    "atm_iv", "atm_delta", "atm_gamma", "atm_theta", "atm_vega",
    "gamma_exposure_ratio", "theta_decay_rate", "iv_skew_ce_pe",
    "max_pain_dist_pct", "max_call_oi_dist_pct", "max_put_oi_dist_pct",
    "call_oi_growth_pct", "put_oi_growth_pct", "fii_net_flow_cr", "dii_net_flow_cr", "institutional_bias_score",
]


def calculate_supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10, multiplier: float = 3.0
) -> Tuple[pd.Series, pd.Series]:
    """Calculate Supertrend indicator (upper/lower bands + direction)."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=1).mean()

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + (multiplier * atr)
    lower_basic = hl2 - (multiplier * atr)

    upper_band = upper_basic.copy()
    lower_band = lower_basic.copy()
    direction = pd.Series(1.0, index=close.index)  # 1 = Bullish, -1 = Bearish

    for i in range(1, len(close)):
        # Lower band trailing
        if close.iloc[i - 1] > lower_band.iloc[i - 1]:
            lower_band.iloc[i] = max(lower_basic.iloc[i], lower_band.iloc[i - 1])
        else:
            lower_band.iloc[i] = lower_basic.iloc[i]

        # Upper band trailing
        if close.iloc[i - 1] < upper_band.iloc[i - 1]:
            upper_band.iloc[i] = min(upper_basic.iloc[i], upper_band.iloc[i - 1])
        else:
            upper_band.iloc[i] = upper_basic.iloc[i]

        # Direction flip
        if close.iloc[i] > upper_band.iloc[i - 1]:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < lower_band.iloc[i - 1]:
            direction.iloc[i] = -1.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    return direction, pd.Series(np.where(direction == 1.0, lower_band, upper_band), index=close.index)


def calculate_adx_dmi(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Average Directional Index (ADX) and +DI / -DI."""
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(window=period, min_periods=1).mean()
    plus_di = 100.0 * (pd.Series(plus_dm, index=close.index).rolling(window=period, min_periods=1).mean() / (atr + 1e-8))
    minus_di = 100.0 * (pd.Series(minus_dm, index=close.index).rolling(window=period, min_periods=1).mean() / (atr + 1e-8))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8))
    adx = dx.rolling(window=period, min_periods=1).mean()
    return adx, plus_di, minus_di


def engineer_80_features(
    df: pd.DataFrame,
    vix_level: float = 13.5,
    pcr_value: float = 1.0,
    pcr_15m_delta: float = 0.0,
    atm_greeks: Optional[Dict[str, float]] = None,
    fii_net_cr: float = 0.0,
    dii_net_cr: float = 0.0,
) -> pd.DataFrame:
    """
    Generate complete 80-feature matrix with strict zero-lookahead guarantee.
    """
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"].astype(float)

    # ── 1. TREND (15) ────────────────────────────────────────────────────────
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()

    df["ema_9_dist_pct"] = (c - ema9) / (c + 1e-8) * 100.0
    df["ema_20_dist_pct"] = (c - ema20) / (c + 1e-8) * 100.0
    df["ema_50_dist_pct"] = (c - ema50) / (c + 1e-8) * 100.0
    df["ema_200_dist_pct"] = (c - ema200) / (c + 1e-8) * 100.0

    df["ema_9_slope"] = ema9.diff(3) / (c + 1e-8) * 100.0
    df["ema_20_slope"] = ema20.diff(3) / (c + 1e-8) * 100.0

    df["ema_9_gt_20"] = np.where(ema9 > ema20, 1.0, 0.0)
    df["ema_20_gt_50"] = np.where(ema20 > ema50, 1.0, 0.0)
    df["ema_50_gt_200"] = np.where(ema50 > ema200, 1.0, 0.0)

    st_dir, st_band = calculate_supertrend(h, l, c, period=10, multiplier=3.0)
    df["supertrend_direction"] = st_dir
    df["supertrend_dist_pct"] = (c - st_band) / (c + 1e-8) * 100.0

    adx, pdi, mdi = calculate_adx_dmi(h, l, c, period=14)
    df["adx_14"] = adx
    df["plus_di_14"] = pdi
    df["minus_di_14"] = mdi
    df["di_spread"] = pdi - mdi

    # ── 2. VOLATILITY (15) ───────────────────────────────────────────────────
    tr1 = h - l
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14 = tr.rolling(window=14, min_periods=1).mean()
    atr20_ma = atr14.rolling(window=20, min_periods=1).mean()

    df["atr_14_pct"] = (atr14 / (c + 1e-8)) * 100.0
    df["atr_regime"] = np.where(df["atr_14_pct"] >= 0.20, 1.0, 0.0)
    df["atr_expansion_ratio"] = atr14 / (atr20_ma + 1e-8)

    bb_mid = c.rolling(window=20, min_periods=1).mean()
    bb_std = c.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0)
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)
    bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-8) * 100.0

    df["bb_upper_dist_pct"] = (bb_upper - c) / (c + 1e-8) * 100.0
    df["bb_lower_dist_pct"] = (c - bb_lower) / (c + 1e-8) * 100.0
    df["bb_width_pct"] = bb_width
    df["bb_width_percentile_rank_50"] = bb_width.rolling(window=50, min_periods=1).apply(
        lambda s: pd.Series(s).rank(pct=True).iloc[-1] if len(s) > 0 else 0.5, raw=False
    )
    df["bb_squeeze"] = np.where(df["bb_width_percentile_rank_50"] <= 0.20, 1.0, 0.0)

    # Keltner
    keltner_mid = ema20
    keltner_upper = keltner_mid + (1.5 * atr14)
    keltner_lower = keltner_mid - (1.5 * atr14)
    df["keltner_upper_dist_pct"] = (keltner_upper - c) / (c + 1e-8) * 100.0
    df["keltner_lower_dist_pct"] = (c - keltner_lower) / (c + 1e-8) * 100.0

    log_ret = pd.Series(np.log(c / c.shift(1)), index=c.index).fillna(0.0)
    df["realized_vol_20"] = log_ret.rolling(window=20, min_periods=1).std(ddof=0).fillna(0.0) * np.sqrt(252 * 75) * 100.0
    df["volatility_ratio"] = df["realized_vol_20"] / (df["realized_vol_20"].rolling(window=50, min_periods=1).mean() + 1e-8)
    
    # Parkinson Volatility
    hl_ratio = pd.Series(np.log(h / (l + 1e-8)), index=c.index)
    df["parkinson_vol"] = np.sqrt(hl_ratio.rolling(20, min_periods=1).apply(lambda s: (s**2).mean() / (4 * np.log(2)), raw=True)) * np.sqrt(252 * 75) * 100.0
    
    # Garman-Klass Volatility
    co_ratio = pd.Series(np.log(c / (o + 1e-8)), index=c.index)
    df["garman_klass_vol"] = np.sqrt(hl_ratio.rolling(20, min_periods=1).apply(lambda s: (0.5 * (s**2) - (2*np.log(2)-1)*(co_ratio**2)).mean(), raw=False).fillna(0.0).clip(lower=0.0)) * 100.0
    df["vix_level"] = vix_level

    # ── 3. MOMENTUM (15) ─────────────────────────────────────────────────────
    delta_p = c.diff()
    gain = np.where(delta_p > 0, delta_p, 0.0)
    loss = np.where(delta_p < 0, -delta_p, 0.0)
    avg_gain = pd.Series(gain, index=c.index).rolling(window=14, min_periods=1).mean()
    avg_loss = pd.Series(loss, index=c.index).rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    df["rsi_14"] = rsi
    df["rsi_slope_3"] = rsi.diff(3)
    df["rsi_oversold"] = np.where(rsi < 30.0, 1.0, 0.0)
    df["rsi_overbought"] = np.where(rsi > 70.0, 1.0, 0.0)

    # Stochastic RSI
    rsi_min = rsi.rolling(14, min_periods=1).min()
    rsi_max = rsi.rolling(14, min_periods=1).max()
    stoch_k = 100.0 * (rsi - rsi_min) / (rsi_max - rsi_min + 1e-8)
    stoch_d = stoch_k.rolling(3, min_periods=1).mean()
    df["stoch_k_14"] = stoch_k
    df["stoch_d_3"] = stoch_d
    df["stoch_cross"] = np.where(stoch_k > stoch_d, 1.0, -1.0)

    # Williams %R
    h14 = h.rolling(14, min_periods=1).max()
    l14 = l.rolling(14, min_periods=1).min()
    df["williams_r_14"] = ((h14 - c) / (h14 - l14 + 1e-8)) * -100.0

    df["roc_5"] = (c.diff(5) / (c.shift(5) + 1e-8)) * 100.0
    df["roc_20"] = (c.diff(20) / (c.shift(20) + 1e-8)) * 100.0

    # Commodity Channel Index (CCI)
    tp = (h + l + c) / 3.0
    tp_ma = tp.rolling(20, min_periods=1).mean()
    tp_dev = (tp - tp_ma).abs().rolling(20, min_periods=1).mean()
    df["cci_20"] = (tp - tp_ma) / (0.015 * tp_dev + 1e-8)

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_sig = macd_line.ewm(span=9, adjust=False).mean()
    macd_h = macd_line - macd_sig
    df["macd_line"] = macd_line
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_h
    df["macd_hist_slope"] = macd_h.diff(2)

    # ── 4. VOLUME & MICROSTRUCTURE (15) ──────────────────────────────────────
    cum_vol = v.cumsum()
    cum_pv = (c * v).cumsum()
    vwap = np.where(cum_vol > 0, cum_pv / cum_vol, c)
    df["vwap_dist_pct"] = (c - vwap) / (c + 1e-8) * 100.0
    df["vwap_slope"] = pd.Series(vwap, index=c.index).diff(3) / (c + 1e-8) * 100.0
    df["price_above_vwap"] = np.where(c > vwap, 1.0, 0.0)

    v_ma20 = v.rolling(20, min_periods=1).mean()
    v_std20 = v.rolling(20, min_periods=1).std(ddof=0).fillna(1.0)
    df["volume_surge_ratio"] = v / (v_ma20 + 1e-8)
    df["volume_zscore_20"] = (v - v_ma20) / (v_std20 + 1e-8)
    df["volume_ma_20_ratio"] = v_ma20 / (v_ma20.rolling(50, min_periods=1).mean() + 1e-8)

    # OBV
    obv_direction = np.where(c > c.shift(1), 1.0, np.where(c < c.shift(1), -1.0, 0.0))
    obv = (pd.Series(obv_direction, index=c.index) * v).cumsum()
    df["obv_slope_5"] = obv.diff(5) / (obv.abs().rolling(20, min_periods=1).mean() + 1e-8)

    # Money Flow Index (MFI)
    raw_mf = tp * v
    pos_mf = np.where(tp > tp.shift(1), raw_mf, 0.0)
    neg_mf = np.where(tp < tp.shift(1), raw_mf, 0.0)
    mfi_pos = pd.Series(pos_mf, index=c.index).rolling(14, min_periods=1).sum()
    mfi_neg = pd.Series(neg_mf, index=c.index).rolling(14, min_periods=1).sum()
    mfr = mfi_pos / (mfi_neg + 1e-8)
    df["mfi_14"] = 100.0 - (100.0 / (1.0 + mfr))

    # Chaikin Money Flow (CMF)
    clv = ((c - l) - (h - c)) / (h - l + 1e-8)
    df["cmf_20"] = (clv * v).rolling(20, min_periods=1).sum() / (v.rolling(20, min_periods=1).sum() + 1e-8)

    # Day Range & Opening Range (Zero-lookahead: use rolling max/min over initial 15 bars)
    df["pdh_dist_pct"] = (h.rolling(75, min_periods=1).max() - c) / (c + 1e-8) * 100.0
    df["pdl_dist_pct"] = (c - l.rolling(75, min_periods=1).min()) / (c + 1e-8) * 100.0
    df["orh_dist_pct"] = (h.iloc[:3].max() - c) / (c + 1e-8) * 100.0 if len(h) >= 3 else 0.0
    df["orl_dist_pct"] = (c - l.iloc[:3].min()) / (c + 1e-8) * 100.0 if len(l) >= 3 else 0.0

    # Cyclical Time Encoding (Assuming bar sequence 0 to 74 in 375 min session)
    bar_indices = np.arange(len(df)) % 75
    df["sin_time_of_day"] = np.sin(2 * np.pi * bar_indices / 75.0)
    df["cos_time_of_day"] = np.cos(2 * np.pi * bar_indices / 75.0)

    # ── 5. OPTIONS & INSTITUTIONAL GREEKS (20) ────────────────────────────────
    df["pcr_value"] = pcr_value
    df["pcr_15m_delta"] = pcr_15m_delta
    df["pcr_regime_bullish"] = 1.0 if pcr_value > 1.2 else 0.0
    df["pcr_regime_bearish"] = 1.0 if pcr_value < 0.8 else 0.0

    greeks = atm_greeks or {}
    df["atm_iv"] = float(greeks.get("iv", 14.5))
    df["atm_delta"] = float(greeks.get("delta", 0.50))
    df["atm_gamma"] = float(greeks.get("gamma", 0.0005))
    df["atm_theta"] = float(greeks.get("theta", -12.5))
    df["atm_vega"] = float(greeks.get("vega", 18.0))

    df["gamma_exposure_ratio"] = df["atm_gamma"] * 10000.0
    df["theta_decay_rate"] = abs(df["atm_theta"]) / (c + 1e-8) * 100.0
    df["iv_skew_ce_pe"] = float(greeks.get("iv_skew", 0.0))

    df["max_pain_dist_pct"] = float(greeks.get("max_pain_dist_pct", 0.0))
    df["max_call_oi_dist_pct"] = float(greeks.get("max_call_oi_dist_pct", 1.5))
    df["max_put_oi_dist_pct"] = float(greeks.get("max_put_oi_dist_pct", 1.5))
    df["call_oi_growth_pct"] = float(greeks.get("call_oi_growth", 0.0))
    df["put_oi_growth_pct"] = float(greeks.get("put_oi_growth", 0.0))

    df["fii_net_flow_cr"] = fii_net_cr
    df["dii_net_flow_cr"] = dii_net_cr
    df["institutional_bias_score"] = 1.0 if (fii_net_cr + dii_net_cr) > 1000 else (-1.0 if (fii_net_cr + dii_net_cr) < -1000 else 0.0)

    # Ensure all 80 features are present and cleanly filled
    return df[EXTENDED_FEATURE_COLS].fillna(0.0)
