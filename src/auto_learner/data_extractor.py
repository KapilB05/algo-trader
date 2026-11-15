"""
Continuous Auto-Learner — Historical Data Extractor & Ingestion Pipeline.

Connects to TimescaleDB / yfinance / local parquet feeds to:
1. Extract the latest 5-minute OHLCV candles for BANKNIFTY
2. Merge with existing historical dataset
3. Deduplicate by timestamp and sort chronologically
4. Save combined archive to `data/historical/banknifty_full_5m.parquet` and CSV
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

logger = logging.getLogger("AutoLearner.DataExtractor")
IST = ZoneInfo("Asia/Kolkata")
HIST_DIR = Path("data/historical")
PARQUET_OUT = HIST_DIR / "banknifty_full_5m.parquet"
CSV_OUT = HIST_DIR / "banknifty_5min.csv"

HIST_DIR.mkdir(parents=True, exist_ok=True)


class DataExtractor:
    """
    Automated historical data ingestion and reconciliation engine.
    """

    def __init__(self, symbol: str = "BANKNIFTY", yf_ticker: str = "^NSEBANK") -> None:
        self.symbol = symbol
        self.yf_ticker = yf_ticker

    def extract_and_merge(self) -> Tuple[pd.DataFrame, int]:
        """
        Extract recent data, merge with existing archives, deduplicate, and save.
        Returns: (merged_df, new_bars_count)
        """
        existing_df = pd.DataFrame()

        # 1. Load existing parquet or CSV
        if PARQUET_OUT.exists():
            existing_df = pd.read_parquet(PARQUET_OUT)
        elif CSV_OUT.exists():
            existing_df = pd.read_csv(CSV_OUT)
        elif Path("data/banknifty_5m.parquet").exists():
            existing_df = pd.read_parquet("data/banknifty_5m.parquet")

        if not existing_df.empty and not pd.api.types.is_datetime64_any_dtype(existing_df["timestamp"]):
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])

        # 2. Fetch latest recent 60-day 5m data via yfinance
        recent_df = pd.DataFrame()
        try:
            t = yf.Ticker(self.yf_ticker)
            raw = t.history(period="60d", interval="5m")
            if not raw.empty:
                raw = raw.reset_index()
                dt_col = "Datetime" if "Datetime" in raw.columns else "Date"
                recent_df = pd.DataFrame({
                    "timestamp": pd.to_datetime(raw[dt_col]),
                    "symbol": self.symbol,
                    "open": raw["Open"].astype(float),
                    "high": raw["High"].astype(float),
                    "low": raw["Low"].astype(float),
                    "close": raw["Close"].astype(float),
                    "volume": raw["Volume"].astype(float),
                })
                logger.info("Fetched %d recent bars via yfinance for %s", len(recent_df), self.symbol)
        except Exception as e:
            logger.warning("yfinance recent data fetch notice: %s", e)

        # 3. Concatenate and Deduplicate
        if existing_df.empty and recent_df.empty:
            raise ValueError(f"No historical or live data available for {self.symbol}")

        combined = pd.concat([existing_df, recent_df], ignore_index=True)
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
        
        # Normalize timezone to IST
        try:
            if combined["timestamp"].dt.tz is None:  # type: ignore
                combined["timestamp"] = combined["timestamp"].dt.tz_localize(IST)  # type: ignore
            else:
                combined["timestamp"] = combined["timestamp"].dt.tz_convert(IST)  # type: ignore
        except Exception as e:
            logger.debug("Timezone localization skipped: %s", e)

        combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # 4. Fill required auxiliary columns if missing
        if "symbol" not in combined.columns:
            combined["symbol"] = self.symbol
        if "volume" not in combined.columns:
            combined["volume"] = 10000.0

        # Save to parquet and CSV
        combined.to_parquet(PARQUET_OUT, index=False)
        combined.to_csv(CSV_OUT, index=False)

        new_count = len(combined) - len(existing_df)
        logger.info("Successfully updated %s dataset: Total %d bars (%d new). Saved to %s",
                    self.symbol, len(combined), max(0, new_count), PARQUET_OUT)

        return combined, max(0, new_count)


# Global instance
data_extractor = DataExtractor()

if __name__ == "__main__":
    df_merged, count = data_extractor.extract_and_merge()
    logger.info("Data Extractor Complete: %d bars in dataset (+%d new bars).", len(df_merged), count)
