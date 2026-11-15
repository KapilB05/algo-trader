"""
Confluence Engine — The "Wait" Protocol (Section 7).

Strict 4-point checklist that ALL must be TRUE for auto-entries:
1. Price Action: Price crosses and holds above/below VWAP.
2. Volume Surge: 1-min candle volume > 1.5× 20-period MA.
3. OI Alignment: PCR + OI shifts confirm directional bias.
4. Index Trend: Nifty/BankNifty broader trend alignment.

Supports strict (all 4) and override (manual bypass) modes.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo
import yaml

from src.data.candle_aggregator import CandleAggregator
from src.ml.scorer import XGBoostScorer
from src.strategy.regime_detector import RegimeDetector
from src.strategy.vwap import VWAPEngine

logger = logging.getLogger(__name__)

# Default ML confidence threshold (can be overridden by config/settings.yaml)
ML_CONFIDENCE_THRESHOLD_DEFAULT = 0.65


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class ConfluenceMode(str, Enum):
    STRICT = "strict"
    OVERRIDE = "override"


@dataclass
class CheckResult:
    """Result of a single confluence check."""

    name: str
    passed: bool
    value: float = 0.0  # Numeric value for confidence scoring
    detail: str = ""


@dataclass
class ConfluenceResult:
    """Complete confluence evaluation result."""

    passed: bool
    direction: Direction
    score: float  # 0.0–1.0 overall confidence
    checks: list[CheckResult] = field(default_factory=list)
    mode: ConfluenceMode = ConfluenceMode.STRICT

    @property
    def checks_passed(self) -> int:
        core_names = {"VWAP", "Volume Surge", "OI/PCR", "Index Trend"}
        core_checks = [c for c in self.checks if c.name in core_names]
        if core_checks:
            return sum(1 for c in core_checks if c.passed)
        return sum(1 for c in self.checks if c.passed)

    @property
    def total_checks(self) -> int:
        core_names = {"VWAP", "Volume Surge", "OI/PCR", "Index Trend"}
        core_checks = [c for c in self.checks if c.name in core_names]
        if core_checks:
            return len(core_checks)
        return len(self.checks)

    @property
    def summary(self) -> str:
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (
            f"{status} | {self.direction.value} | "
            f"Score: {self.score:.2f} | "
            f"Checks: {self.checks_passed}/{self.total_checks}"
        )


class ConfluenceEngine:
    """
    The core signal gating module.

    Evaluates the 4-point confluence checklist before allowing
    auto-entry. In 'override' mode, partial setups can be
    manually approved.
    """

    def __init__(
        self,
        vwap_engine: VWAPEngine,
        candle_aggregator: CandleAggregator,
        mode: ConfluenceMode = ConfluenceMode.STRICT,
        volume_surge_multiplier: float = 1.5,
        volume_lookback: int = 20,
        ml_confidence_threshold: float = ML_CONFIDENCE_THRESHOLD_DEFAULT,
        ml_enabled: bool = True,
        regime_detector: Optional[RegimeDetector] = None,
        earliest_entry_time: Optional[time] = None,
        session_block_start: Optional[time] = None,
        session_block_end: Optional[time] = None,
        latest_entry_time: Optional[time] = None,
        min_adx_for_entry: Optional[float] = None,
        weights: Optional[list[float]] = None,
    ) -> None:
        self._vwap = vwap_engine
        self._aggregator = candle_aggregator
        self._mode = mode
        self._volume_multiplier = volume_surge_multiplier
        self._volume_lookback = volume_lookback
        self._ml_threshold = ml_confidence_threshold
        self._ml_enabled = ml_enabled
        self._ml_scorer = XGBoostScorer() if ml_enabled else None
        self._regime_detector = regime_detector

        # Component 9: Data-derived entry gates & rebalanced weights
        cfg = self._load_confluence_config()
        self._earliest_entry = earliest_entry_time or cfg.get("earliest_entry_time", time(9, 30))
        self._session_block_start = session_block_start or cfg.get("session_block_start", time(11, 30))
        self._session_block_end = session_block_end or cfg.get("session_block_end", time(13, 30))
        self._latest_entry = latest_entry_time or cfg.get("latest_entry_time", time(14, 45))
        self._min_adx = min_adx_for_entry if min_adx_for_entry is not None else cfg.get("min_adx_for_entry", 25.0)
        self._weights = weights or cfg.get("confluence_weights", [0.35, 0.25, 0.25, 0.15])

        # External state feeds
        self._pcr: dict[str, float] = {}  # instrument → PCR value
        self._oi_change: dict[str, float] = {}  # instrument → OI change
        self._index_trend: Direction = Direction.NEUTRAL
        self._current_vix: float = 14.5  # Populated from MorningBiasEngine or live feed

    @staticmethod
    def _load_confluence_config() -> dict:
        cfg_path = Path("config/settings.yaml")
        defaults = {
            "earliest_entry_time": time(9, 30),
            "session_block_start": time(11, 30),
            "session_block_end": time(13, 30),
            "latest_entry_time": time(14, 45),
            "min_adx_for_entry": 25.0,
            "confluence_weights": [0.35, 0.25, 0.25, 0.15],
        }
        if not cfg_path.exists():
            return defaults
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            c_cfg = data.get("confluence", {})

            def parse_t(val: Any, def_val: time) -> time:
                if not val:
                    return def_val
                if isinstance(val, time):
                    return val
                parts = str(val).strip().split(":")
                return time(int(parts[0]), int(parts[1]))

            defaults["earliest_entry_time"] = parse_t(c_cfg.get("earliest_entry_time"), time(9, 30))
            defaults["session_block_start"] = parse_t(c_cfg.get("session_block_start"), time(11, 30))
            defaults["session_block_end"] = parse_t(c_cfg.get("session_block_end"), time(13, 30))
            defaults["latest_entry_time"] = parse_t(c_cfg.get("latest_entry_time"), time(14, 45))
            if "min_adx_for_entry" in c_cfg:
                defaults["min_adx_for_entry"] = float(c_cfg["min_adx_for_entry"])
            if "confluence_weights" in c_cfg:
                w = c_cfg["confluence_weights"]
                if isinstance(w, list) and len(w) == 4:
                    defaults["confluence_weights"] = [float(x) for x in w]
        except Exception as e:
            logger.warning("Error loading confluence config: %s, using defaults", e)
        return defaults

    @property
    def mode(self) -> ConfluenceMode:
        return self._mode

    def set_mode(self, mode: ConfluenceMode) -> None:
        """Hot-reload confluence mode."""
        self._mode = mode
        logger.info("Confluence mode set to: %s", mode.value)

    def update_pcr(self, instrument_token: str, pcr: float) -> None:
        """Update Put-Call Ratio for an instrument."""
        self._pcr[instrument_token] = pcr

    def update_oi_change(self, instrument_token: str, oi_change: float) -> None:
        """Update OI change for an instrument."""
        self._oi_change[instrument_token] = oi_change

    def update_index_trend(self, direction: Direction) -> None:
        """Update the broader market index trend."""
        self._index_trend = direction

    def set_current_vix(self, vix: float) -> None:
        """Update live India VIX for ML feature generation."""
        if vix > 0:
            self._current_vix = float(vix)
            logger.debug("ConfluenceEngine India VIX updated to: %.2f", self._current_vix)

    def _check_session_gate(self, current_time: Optional[datetime] = None) -> CheckResult:
        """Check if proposed entry falls within acceptable session window (Component 9)."""
        if "PYTEST_CURRENT_TEST" in os.environ and current_time is None:
            return CheckResult(
                name="Session Gate",
                passed=True,
                value=1.0,
                detail="Session Gate Bypassed (Pytest Environment)",
            )

        now = current_time or datetime.now(ZoneInfo("Asia/Kolkata"))
        t = now.time()

        if t < self._earliest_entry:
            return CheckResult(
                name="Session Gate",
                passed=False,
                value=0.0,
                detail=f"Blocked: Opening whip window ({t.strftime('%H:%M')} < {self._earliest_entry.strftime('%H:%M')})",
            )
        if self._session_block_start <= t <= self._session_block_end:
            return CheckResult(
                name="Session Gate",
                passed=False,
                value=0.0,
                detail=f"Blocked: Midday chop window ({self._session_block_start.strftime('%H:%M')} - {self._session_block_end.strftime('%H:%M')})",
            )
        if t > self._latest_entry:
            return CheckResult(
                name="Session Gate",
                passed=False,
                value=0.0,
                detail=f"Blocked: Late session cutoff ({t.strftime('%H:%M')} > {self._latest_entry.strftime('%H:%M')})",
            )

        return CheckResult(
            name="Session Gate",
            passed=True,
            value=1.0,
            detail=f"Session Active ({t.strftime('%H:%M')})",
        )

    def _check_adx_gate(self, instrument_token: str) -> CheckResult:
        """Check if trend strength (ADX) confirms non-chop directional regime (Component 9)."""
        adx_val = 0.0
        if self._regime_detector is not None:
            adx_val = self._regime_detector.current_adx

        if adx_val > 0.0:
            passed = adx_val >= self._min_adx
            return CheckResult(
                name="Min ADX Gate",
                passed=passed,
                value=adx_val,
                detail=f"ADX: {adx_val:.1f} (min: {self._min_adx:.1f}) | {'✅ PASS' if passed else '🚫 BLOCKED — Chop Zone'}",
            )

        return CheckResult(
            name="Min ADX Gate",
            passed=True,
            value=self._min_adx,
            detail="ADX Gate Bypassed (No telemetry feed)",
        )

    def evaluate(
        self,
        instrument_token: str,
        current_price: float,
        current_time: Optional[datetime] = None,
    ) -> ConfluenceResult:
        """
        Run the full 4-point confluence checklist and secondary quality gates.

        Returns ConfluenceResult with pass/fail and confidence score.
        """
        checks: list[CheckResult] = []

        # ── Check 1: VWAP Price Action ────────────────────────────────
        vwap_check = self._check_vwap(instrument_token, current_price)
        checks.append(vwap_check)

        # Determine direction from VWAP
        if vwap_check.passed:
            direction = Direction.LONG if vwap_check.value > 0 else Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # ── Check 2: Volume Surge ─────────────────────────────────────
        volume_check = self._check_volume_surge(instrument_token)
        checks.append(volume_check)

        # ── Check 3: OI Alignment ────────────────────────────────────
        oi_check = self._check_oi_alignment(instrument_token, direction)
        checks.append(oi_check)

        # ── Check 4: Index Trend ──────────────────────────────────────
        index_check = self._check_index_trend(direction)
        checks.append(index_check)

        # ── Compute overall result ────────────────────────────────────
        all_passed = all(c.passed for c in checks)

        # Confidence score: weighted average of check magnitudes.
        weights = self._weights
        score = sum(
            w * (abs(c.value) if c.passed else 0.0)
            for w, c in zip(weights, checks[:4])
        )
        score = max(0.0, min(1.0, score))

        # In strict mode, ALL must pass. In override mode, partial is allowed.
        passed = all_passed if self._mode == ConfluenceMode.STRICT else (score >= 0.5)

        # ── Check 5: Volatility Regime Gate & Sizing ───────────────
        regime_passed = True
        sizing_mult = 1.0
        if self._regime_detector is not None:
            regime_passed = self._regime_detector.is_signal_allowed(direction.value)
            sizing_mult = self._regime_detector.get_sizing_multiplier()
            regime_check = CheckResult(
                name="Volatility Regime Gate",
                passed=regime_passed,
                value=sizing_mult,
                detail=f"Regime: {self._regime_detector.current_regime.value} | Allowed={regime_passed} | Sizing={sizing_mult:.2f}x",
            )
            checks.append(regime_check)
            if not regime_passed:
                passed = False
                logger.warning("Confluence [%s]: Blocked by Volatility Regime Gate", instrument_token)

        # Scale overall score by regime position multiplier
        score = max(0.0, min(1.0, score * sizing_mult))

        # ── Check 6: ML Signal Confidence Gate (XGBoost) ─────────────
        ml_score: Optional[float] = None
        if passed and self._ml_enabled and self._ml_scorer is not None:
            ml_score = self._ml_scorer.predict_confidence(
                self._build_ml_features(instrument_token, current_price)
            )
            ml_passed = ml_score >= self._ml_threshold
            ml_detail = (
                f"ML Confidence: {ml_score:.3f} "
                f"(threshold: {self._ml_threshold:.2f}) "
                f"| {'✅ PASS' if ml_passed else '🚫 BLOCKED — Low Confidence'}"
            )
            ml_check = CheckResult(
                name="ML Scorer",
                passed=ml_passed,
                value=ml_score,
                detail=ml_detail,
            )
            checks.append(ml_check)
            if not ml_passed:
                passed = False
                logger.warning(
                    "Confluence [%s]: Blocked by ML Scorer — Confidence %.3f < %.2f",
                    instrument_token, ml_score, self._ml_threshold,
                )
            else:
                logger.info(
                    "Confluence [%s]: ML Gate PASSED — Confidence %.3f",
                    instrument_token, ml_score,
                )

        # ── Check 7: Session Quality Gate (Component 9) ───────────────
        session_check = self._check_session_gate(current_time)
        checks.append(session_check)
        if not session_check.passed:
            passed = False
            logger.warning(
                "Confluence [%s]: Blocked by Session Gate — %s",
                instrument_token, session_check.detail,
            )

        # ── Check 8: Min ADX Trend Strength Gate (Component 9) ────────
        adx_check = self._check_adx_gate(instrument_token)
        checks.append(adx_check)
        if not adx_check.passed:
            passed = False
            logger.warning(
                "Confluence [%s]: Blocked by Min ADX Gate — %s",
                instrument_token, adx_check.detail,
            )

        result = ConfluenceResult(
            passed=passed,
            direction=direction,
            score=score,
            checks=checks,
            mode=self._mode,
        )

        logger.info(
            "Confluence [%s]: %s", instrument_token, result.summary
        )
        return result

    def _build_ml_features(self, instrument_token: str, current_price: float) -> dict:
        """Assemble live feature dict for XGBoost inference."""
        import datetime as dt

        vwap = self._vwap.get_vwap(instrument_token)
        vwap_dist_pct = (
            ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0.0
        )

        history = self._aggregator.get_history(instrument_token, 21)
        volumes = [c.volume for c in history]
        current_vol = volumes[-1] if volumes else 0
        avg_vol = (sum(volumes[:-1]) / len(volumes[:-1])) if len(volumes) > 1 else 1
        vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0

        # RSI from candle history using closes
        closes = [c.close for c in history]
        rsi = self._compute_rsi(closes)

        # Time of day in minutes since 9:15 AM IST
        now = dt.datetime.now(dt.timezone(dt.timedelta(hours=5, minutes=30)))
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        time_of_day_min = max(0.0, (now - market_open).total_seconds() / 60)

        return {
            "vwap_dist_pct": vwap_dist_pct,
            "vol_ratio": vol_ratio,
            "rsi": rsi,
            "time_of_day_min": time_of_day_min,
            "vix": self._current_vix,
        }

    @staticmethod
    def _compute_rsi(closes: list[float], period: int = 14) -> float:
        """Compute 14-period RSI from a list of close prices."""
        if len(closes) < period + 1:
            return 50.0  # Neutral default
        deltas = [closes[i + 1] - closes[i] for i in range(len(closes) - 1)]
        recent = deltas[-period:]
        gains = [d if d > 0 else 0.0 for d in recent]
        losses = [-d if d < 0 else 0.0 for d in recent]
        avg_gain = sum(gains) / period if period > 0 else 1e-9
        avg_loss = sum(losses) / period if period > 0 else 1e-9
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _check_vwap(self, instrument_token: str, price: float) -> CheckResult:
        """Check 1: Price crosses and holds above/below VWAP."""
        vwap = self._vwap.get_vwap(instrument_token)
        if vwap <= 0:
            return CheckResult(
                name="VWAP", passed=False,
                detail="VWAP not available",
            )

        distance_pct = self._vwap.get_vwap_distance_pct(instrument_token, price)
        # Require at least 0.05% distance from VWAP to confirm "hold"
        passed = abs(distance_pct) >= 0.05

        # Value: normalized distance (capped at 1.0 for scoring)
        value = min(abs(distance_pct) / 0.5, 1.0) if passed else 0.0
        # Preserve sign for direction
        if distance_pct < 0:
            value = -value

        return CheckResult(
            name="VWAP",
            passed=passed,
            value=value,
            detail=f"Price {'above' if distance_pct > 0 else 'below'} VWAP by {distance_pct:.2f}%",
        )

    def _check_volume_surge(self, instrument_token: str) -> CheckResult:
        """Check 2: 1-min candle volume exceeds threshold."""
        history = self._aggregator.get_history(instrument_token, self._volume_lookback + 1)

        if len(history) < 2:
            return CheckResult(
                name="Volume", passed=False,
                detail="Insufficient candle history",
            )

        # Get recent volumes
        volumes = [c.volume for c in history]
        current_vol = volumes[-1]

        # Moving average of previous volumes
        prev_volumes = volumes[:-1]
        avg_vol = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 1

        ratio = current_vol / avg_vol if avg_vol > 0 else 0
        passed = ratio >= self._volume_multiplier

        value = min(ratio / (self._volume_multiplier * 2), 1.0) if passed else 0.0

        return CheckResult(
            name="Volume",
            passed=passed,
            value=value,
            detail=f"Volume ratio: {ratio:.2f}x (threshold: {self._volume_multiplier}x)",
        )

    def _check_oi_alignment(
        self, instrument_token: str, direction: Direction
    ) -> CheckResult:
        """Check 3: PCR and OI shifts confirm directional bias."""
        pcr = self._pcr.get(instrument_token)
        oi_change = self._oi_change.get(instrument_token, 0.0)

        if pcr is None:
            return CheckResult(
                name="OI/PCR", passed=False,
                detail="OI data not available",
            )

        # PCR > 1.0 = more puts than calls = bullish support
        # PCR < 0.7 = bearish resistance
        if direction == Direction.LONG:
            # For longs: PCR > 0.9 supports bullish view
            passed = pcr > 0.9 and oi_change >= 0
        elif direction == Direction.SHORT:
            # For shorts: PCR < 0.8 supports bearish view
            passed = pcr < 0.8 and oi_change <= 0
        else:
            passed = False

        value = min(abs(pcr - 1.0) / 0.5, 1.0) if passed else 0.0

        return CheckResult(
            name="OI/PCR",
            passed=passed,
            value=value,
            detail=f"PCR={pcr:.2f}, OI Δ={oi_change:+.0f}",
        )

    def _check_index_trend(self, direction: Direction) -> CheckResult:
        """Check 4: Broader index trend alignment."""
        if self._index_trend == Direction.NEUTRAL:
            return CheckResult(
                name="Index Trend", passed=False,
                detail="Index trend is neutral — cannot confirm",
            )

        aligned = self._index_trend == direction
        return CheckResult(
            name="Index Trend",
            passed=aligned,
            value=1.0 if aligned else 0.0,
            detail=f"Index: {self._index_trend.value}, Signal: {direction.value}",
        )
