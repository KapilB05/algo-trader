"""
Historical 5-Minute Data Downloader for Nifty & BankNifty Futures via Angel One SmartAPI.

Features:
- Dynamically resolves front-month Futures contract tokens from data/instruments.json
- Fetches 6 months (180 days) of 5-minute historical candles in 30-day chunks
- Formats columns: timestamp, symbol, open, high, low, close, volume, oi
- Saves to data/nifty_5m.parquet, data/banknifty_5m.parquet, and data/historical_5m.parquet
- Reports comprehensive dataset statistics (date range, total candles, volume %, OI %, ATR stats)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.auth.angel_auth import AngelManager

logger = logging.getLogger(__name__)

INSTRUMENTS_FILE = Path("data/instruments.json")
OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_front_month_future(symbol_name: str) -> Dict[str, str]:
    """
    Dynamically resolve front-month Futures contract from data/instruments.json.
    """
    if INSTRUMENTS_FILE.exists():
        try:
            with open(INSTRUMENTS_FILE, "r", encoding="utf-8") as f:
                instruments = json.load(f)
            df = pd.DataFrame(instruments)

            if "name" in df.columns and "instrument_type" in df.columns:
                sub = df[
                    (df["name"] == symbol_name)
                    & (df["instrument_type"].str.contains("FUT", na=False))
                ].copy()

                if not sub.empty:
                    sub["expiry_dt"] = pd.to_datetime(sub["expiry"], unit="ms", errors="coerce")
                    now = pd.Timestamp.now()
                    valid_fut = sub[sub["expiry_dt"] >= now].sort_values("expiry_dt")

                    if valid_fut.empty:
                        valid_fut = sub.sort_values("expiry_dt", ascending=False)

                    front = valid_fut.iloc[0]
                    return {
                        "token": str(front["exchange_token"]),
                        "symbol": str(front["trading_symbol"]),
                        "name": str(front["name"]),
                        "exchange": "NFO",
                    }
        except Exception as exc:
            logger.warning("Failed to resolve dynamic contract from %s: %s", INSTRUMENTS_FILE, exc)

    # Fallback known tokens
    fallback_tokens = {
        "NIFTY": {"token": "58072", "symbol": "NIFTY FUT 25 AUG 26", "name": "NIFTY", "exchange": "NFO"},
        "BANKNIFTY": {"token": "58067", "symbol": "BANKNIFTY FUT 25 AUG 26", "name": "BANKNIFTY", "exchange": "NFO"},
    }
    return fallback_tokens.get(symbol_name, {"token": "58072", "symbol": f"{symbol_name} FUT", "name": symbol_name, "exchange": "NFO"})


def calculate_wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate standard Wilder's 14-period Average True Range (ATR)."""
    high = df["high"]
    low = df["low"]
    close = df["close"].shift(1)

    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's exponential smoothing
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


def fetch_symbol_historical_5m(
    auth: AngelManager,
    symbol_name: str,
    days: int = 180,
    chunk_days: int = 30,
) -> pd.DataFrame:
    """
    Fetch 5-minute historical candle data for a given symbol over N days in chunk_days requests.
    """
    contract_info = resolve_front_month_future(symbol_name)
    token = contract_info["token"]
    symbol_code = contract_info["symbol"]
    exchange = contract_info["exchange"]

    logger.info("Fetching %d days of 5-min candles for %s (Token: %s)", days, symbol_code, token)
    print(f"\n[Downloader] Starting 5-minute download for {symbol_code} (Token: {token}) over past {days} days...")

    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=days)
    current_start = start_dt

    all_rows: List[list] = []

    while current_start < end_dt:
        current_end = min(current_start + timedelta(days=chunk_days), end_dt)
        f_str = current_start.strftime("%Y-%m-%d 09:15")
        t_str = current_end.strftime("%Y-%m-%d 15:30")

        params = {
            "exchange": exchange,
            "symboltoken": token,
            "interval": "FIVE_MINUTE",
            "fromdate": f_str,
            "todate": t_str,
        }

        try:
            res = auth.smartApi.getCandleData(params)
            if res and isinstance(res, dict) and res.get("status") and res.get("data"):
                chunk_data = res.get("data", [])
                all_rows.extend(chunk_data)
                print(f"  - Downloaded chunk {f_str[:10]} to {t_str[:10]}: {len(chunk_data)} candles")
            else:
                msg = res.get("message") if isinstance(res, dict) else str(res)
                logger.warning("Empty/Failed chunk %s to %s: %s", f_str[:10], t_str[:10], msg)
        except Exception as exc:
            logger.error("Exception fetching chunk %s to %s: %s", f_str[:10], t_str[:10], exc)

        current_start = current_end + timedelta(days=1)
        time.sleep(0.35)

    if not all_rows:
        logger.error("No candle data fetched for %s!", symbol_code)
        return pd.DataFrame()

    # Parse raw rows into DataFrame
    records = []
    for r in all_rows:
        ts = r[0]
        o = float(r[1])
        h = float(r[2])
        l = float(r[3])
        c = float(r[4])
        v = float(r[5])
        oi = float(r[6]) if len(r) > 6 else 0.0
        records.append({
            "timestamp": ts,
            "symbol": symbol_code,
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
            "oi": oi,
        })

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Compute Wilder's ATR(14)
    df["atr_14"] = calculate_wilder_atr(df, period=14)

    return df


def print_dataset_statistics(df: pd.DataFrame, symbol_name: str) -> None:
    """Print comprehensive dataset statistics as required by specifications."""
    if df.empty:
        print(f"\n❌ [Stats] {symbol_name}: Dataset is empty!")
        return

    start_date = df["timestamp"].min().strftime("%Y-%m-%d %H:%M")
    end_date = df["timestamp"].max().strftime("%Y-%m-%d %H:%M")
    total_candles = len(df)
    vol_gt_0 = (df["volume"] > 0).mean() * 100
    oi_gt_0 = (df["oi"] > 0).mean() * 100

    min_close = df["close"].min()
    max_close = df["close"].max()
    mean_close = df["close"].mean()

    valid_atr = df["atr_14"].dropna()
    min_atr = valid_atr.min() if not valid_atr.empty else 0.0
    max_atr = valid_atr.max() if not valid_atr.empty else 0.0
    mean_atr = valid_atr.mean() if not valid_atr.empty else 0.0
    mean_atr_pct = (mean_atr / mean_close) * 100 if mean_close > 0 else 0.0

    print("\n" + "=" * 80)
    print(f"         DATASET STATISTICS REPORT: {symbol_name}")
    print("=" * 80)
    print(f"Date Range                  : {start_date}  -->  {end_date}")
    print(f"Total 5-Minute Candles      : {total_candles:,}")
    print(f"Candles with Volume > 0 (%) : {vol_gt_0:.2f}%")
    print(f"Candles with OI > 0 (%)     : {oi_gt_0:.2f}%")
    print(f"Close Price Range           : Min {min_close:.2f} | Max {max_close:.2f} | Mean {mean_close:.2f}")
    print(f"Wilder ATR(14) Range        : Min {min_atr:.2f} | Max {max_atr:.2f} | Mean {mean_atr:.2f} ({mean_atr_pct:.2f}% of price)")
    print("=" * 80 + "\n")


def run_historical_download(days: int = 180) -> Dict[str, pd.DataFrame]:
    """Execute historical download for NIFTY and BANKNIFTY Futures."""
    auth = AngelManager()
    if not auth.authenticate():
        logger.error("Angel One authentication failed! Unable to download historical data.")
        raise RuntimeError("Angel One Authentication failed. Verify ANGEL_API_KEY, ANGEL_CLIENT_ID, etc. in .env")

    datasets = {}

    for sym in ["NIFTY", "BANKNIFTY"]:
        df_sym = fetch_symbol_historical_5m(auth, symbol_name=sym, days=days)
        if not df_sym.empty:
            out_file = OUTPUT_DIR / f"{sym.lower()}_5m.parquet"
            df_sym.to_parquet(out_file, index=False)
            print(f"Saved {len(df_sym)} rows to {out_file}")
            print_dataset_statistics(df_sym, sym)
            datasets[sym] = df_sym

    if datasets:
        df_combined = pd.concat(datasets.values(), ignore_index=True)
        unified_file = OUTPUT_DIR / "historical_5m.parquet"
        df_combined.to_parquet(unified_file, index=False)
        print(f"[OK] Saved unified dataset with {len(df_combined):,} rows to {unified_file}")

    return datasets


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_historical_download(days=180)
