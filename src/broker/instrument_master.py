"""
Instrument Master — Daily instrument download and option strike resolver.

Downloads the daily instrument master (Upstox format, matches the cached
data/instruments.json schema) and caches it locally. Provides fast
option strike resolution using Polars for filtering.
"""

from __future__ import annotations

import gzip
import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import orjson
import polars as pl

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────
INSTRUMENT_CACHE = Path("data/instruments.json")

# Upstox instrument master (schema: instrument_key, trading_symbol, name,
# segment, expiry [epoch ms], strike_price, instrument_type, lot_size, ...)
IST = timezone(timedelta(hours=5, minutes=30))
INSTRUMENT_URLS = {
    "ALL": "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz",
}


class InstrumentMaster:
    """
    Downloads, caches, and queries the daily instrument master.

    Uses Polars for high-speed filtering of the instrument dataset
    (typically 100k+ rows for NSE + NFO combined).
    """

    def __init__(self) -> None:
        self._df: Optional[pl.DataFrame] = None
        self._last_download: Optional[date] = None

    async def download(self, exchanges: Optional[list[str]] = None) -> pl.DataFrame:
        """
        Download the complete instrument master and cache it locally.

        The Upstox master is a single gzipped JSON covering all exchanges;
        rows are tagged with their own segment (NSE_EQ / NSE_FO / ...).
        """
        all_instruments: list[dict] = []

        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            for name, url in INSTRUMENT_URLS.items():
                logger.info("Downloading instrument master (%s) ...", name)
                try:
                    response = await client.get(url)
                    response.raise_for_status()

                    content = response.content
                    if content[:2] == b"\x1f\x8b":  # gzip magic
                        content = gzip.decompress(content)

                    instruments = orjson.loads(content)
                    if isinstance(instruments, list):
                        for inst in instruments:
                            inst["_exchange"] = inst.get("segment") or inst.get("exchange") or ""
                        all_instruments.extend(instruments)
                        logger.info(
                            "  → %s: %d instruments loaded.", name, len(instruments)
                        )

                except Exception as exc:
                    logger.error("Failed to download %s instruments: %s", name, exc)

        if not all_instruments:
            logger.error("No instruments downloaded!")
            return pl.DataFrame()

        # Build Polars DataFrame for fast querying
        self._df = pl.DataFrame(all_instruments)
        self._last_download = date.today()

        # Persist to disk
        self._persist(all_instruments)

        logger.info(
            "Instrument master loaded: %d total instruments.", len(all_instruments)
        )
        return self._df

    def load_from_cache(self) -> Optional[pl.DataFrame]:
        """Load instruments from the local JSON cache."""
        if not INSTRUMENT_CACHE.exists():
            logger.warning("No cached instrument file found.")
            return None

        try:
            raw = INSTRUMENT_CACHE.read_bytes()
            instruments = orjson.loads(raw)
            self._df = pl.DataFrame(instruments)
            logger.info(
                "Loaded %d instruments from cache.", len(instruments)
            )
            return self._df

        except Exception as exc:
            logger.error("Failed to load instrument cache: %s", exc)
            return None

    def resolve_option_strike(
        self,
        underlying: str,
        expiry: date,
        strike: float,
        option_type: str,  # "CE" or "PE"
    ) -> Optional[str]:
        """
        Resolve a weekly option's instrument_key from the master.

        Args:
            underlying: e.g., "NIFTY", "BANKNIFTY"
            expiry: Expiry date
            strike: Strike price
            option_type: "CE" or "PE"

        Returns:
            Upstox instrument_key (e.g. "NSE_FO|59312") or None if not found.
        """
        if self._df is None:
            logger.error("Instrument master not loaded. Call download() first.")
            return None

        try:
            # Expiry is stored as epoch milliseconds (end of day IST)
            expiry_start = datetime.combine(expiry, datetime.min.time(), tzinfo=IST)
            expiry_end = expiry_start + timedelta(days=1)
            expiry_ms_min = int(expiry_start.timestamp() * 1000)
            expiry_ms_max = int(expiry_end.timestamp() * 1000)

            # Filter using Polars for speed
            result = self._df.filter(
                (pl.col("segment") == "NSE_FO")
                & (pl.col("name") == underlying.upper())
                & (pl.col("expiry").is_not_null())
                & (pl.col("expiry") >= expiry_ms_min)
                & (pl.col("expiry") < expiry_ms_max)
                & (pl.col("strike_price") == float(strike))
                & (pl.col("instrument_type") == option_type.upper())
            )

            if result.height == 0:
                logger.warning(
                    "No instrument found: %s %s %s %s",
                    underlying, expiry.isoformat(), strike, option_type,
                )
                return None

            token = result.row(0, named=True).get("instrument_key", "")
            logger.debug(
                "Resolved %s %s %s %s → %s",
                underlying, expiry.isoformat(), strike, option_type, token,
            )
            return token or None

        except Exception as exc:
            logger.error("Strike resolution failed: %s", exc)
            return None

    def get_next_weekly_expiry(self, underlying: str = "NIFTY") -> Optional[date]:
        """Find the nearest weekly expiry for an underlying from the master."""
        if self._df is None:
            return None

        today = date.today()

        try:
            result = (
                self._df.filter(
                    (pl.col("segment") == "NSE_FO")
                    & (pl.col("name") == underlying.upper())
                    & (pl.col("expiry").is_not_null())
                )
                .select("expiry")
                .unique()
                .sort("expiry")
            )

            for row in result.iter_rows():
                expiry_ms = row[0]
                if isinstance(expiry_ms, (int, float)):
                    expiry_date = datetime.fromtimestamp(
                        expiry_ms / 1000.0, tz=IST
                    ).date()
                    if expiry_date >= today:
                        return expiry_date

            return None

        except Exception as exc:
            logger.error("Failed to find next expiry: %s", exc)
            return None

    def search(self, query: str, exchange: Optional[str] = None) -> pl.DataFrame:
        """Search instruments by name or symbol (case-insensitive)."""
        if self._df is None:
            return pl.DataFrame()

        df = self._df
        if exchange:
            df = df.filter(pl.col("_exchange") == exchange)

        return df.filter(
            pl.col("name").str.to_lowercase().str.contains(query.lower())
            | pl.col("trading_symbol").str.to_lowercase().str.contains(query.lower())
        )

    @property
    def is_loaded(self) -> bool:
        """Check if instrument data is available."""
        return self._df is not None and self._df.height > 0

    @staticmethod
    def _persist(instruments: list[dict]) -> None:
        """Save instruments to local JSON cache."""
        INSTRUMENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        INSTRUMENT_CACHE.write_bytes(orjson.dumps(instruments))
        logger.debug("Instruments cached to %s", INSTRUMENT_CACHE)
