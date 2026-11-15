"""
Signal Confidence Scorer — 0.0 to 1.0 confidence for position sizing.

Computes a continuous confidence score from:
- Confluence checklist strength
- VIX regime alignment
- Volume surge magnitude
- OI support level proximity
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.strategy.confluence_engine import ConfluenceResult
from src.strategy.regime_detector import MarketRegime, RegimeDetector

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBreakdown:
    """Detailed breakdown of the confidence score."""

    confluence_score: float  # From confluence engine (0–1)
    regime_alignment: float  # VIX regime alignment bonus (0–0.2)
    volume_strength: float   # Volume surge strength (0–0.15)
    oi_conviction: float     # OI alignment conviction (0–0.15)
    total: float             # Final score capped at 1.0


class SignalConfidence:
    """
    Computes position-sizing confidence from multiple signal inputs.

    The total score (0.0–1.0) directly scales the position sizer's
    output: higher confidence → larger position (up to 2% risk cap).
    """

    def __init__(self, regime_detector: RegimeDetector) -> None:
        self._regime = regime_detector

    def compute(
        self,
        confluence: ConfluenceResult,
        strategy_direction: str = "LONG",
    ) -> ConfidenceBreakdown:
        """
        Compute confidence score from all available signals.
        """
        # ── Base: Confluence score ────────────────────────────────────
        confluence_score = confluence.score

        # ── Regime alignment bonus ────────────────────────────────────
        regime = self._regime.current_regime
        if strategy_direction == "LONG":
            # High VIX + long momentum = aligned
            regime_alignment = 0.2 if regime == MarketRegime.HIGH_VIX else 0.1
        elif strategy_direction == "SHORT":
            regime_alignment = 0.2 if regime == MarketRegime.HIGH_VIX else 0.1
        else:
            regime_alignment = 0.05

        # Penalty for neutral regime
        if regime == MarketRegime.NEUTRAL:
            regime_alignment = 0.05

        # ── Volume strength ───────────────────────────────────────────
        volume_check = next(
            (c for c in confluence.checks if c.name == "Volume"), None
        )
        volume_strength = (volume_check.value * 0.15) if volume_check and volume_check.passed else 0.0

        # ── OI conviction ─────────────────────────────────────────────
        oi_check = next(
            (c for c in confluence.checks if c.name == "OI/PCR"), None
        )
        oi_conviction = (oi_check.value * 0.15) if oi_check and oi_check.passed else 0.0

        # ── Total (capped at 1.0) ─────────────────────────────────────
        total = min(
            1.0,
            confluence_score * 0.5  # 50% weight from confluence
            + regime_alignment
            + volume_strength
            + oi_conviction,
        )

        breakdown = ConfidenceBreakdown(
            confluence_score=confluence_score,
            regime_alignment=regime_alignment,
            volume_strength=volume_strength,
            oi_conviction=oi_conviction,
            total=total,
        )

        logger.debug(
            "Confidence: %.2f (confluence=%.2f, regime=%.2f, vol=%.2f, oi=%.2f)",
            total, confluence_score, regime_alignment, volume_strength, oi_conviction,
        )

        return breakdown
