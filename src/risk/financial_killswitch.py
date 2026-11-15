"""
Financial Kill Switch — Sharpe ratio and slippage degradation halt.

Halts execution if:
- Live Sharpe falls below 50% of backtest Sharpe.
- Execution slippage exceeds 2× modeled slippage.
"""

from __future__ import annotations

import logging
from math import sqrt  # type: ignore[attr-defined]
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchStatus:
    """Current kill switch status."""

    is_halted: bool
    reason: str
    live_sharpe: float
    backtest_sharpe: float
    sharpe_ratio_pct: float  # live/backtest as %
    live_slippage_bps: float
    backtest_slippage_bps: float
    slippage_ratio: float


class FinancialKillSwitch:
    """
    Alpha health monitor that halts execution when live performance
    degrades below acceptable thresholds.

    Tracks:
    - Rolling Sharpe ratio from recent trade returns
    - Average execution slippage in basis points
    """

    def __init__(
        self,
        backtest_sharpe: float = 1.5,
        backtest_slippage_bps: float = 5.0,
        sharpe_degradation_pct: float = 0.50,
        slippage_multiplier: float = 2.0,
        rolling_window: int = 50,
        enabled: bool = True,
    ) -> None:
        self._backtest_sharpe = backtest_sharpe
        self._backtest_slippage = backtest_slippage_bps
        self._sharpe_threshold_pct = sharpe_degradation_pct
        self._slippage_max_mult = slippage_multiplier
        self._enabled = enabled

        # Rolling return buffer for Sharpe computation
        self._returns: deque[float] = deque(maxlen=rolling_window)

        # Slippage tracking
        self._slippages: deque[float] = deque(maxlen=rolling_window)

        self._is_halted = False
        self._halt_reason = ""

    @property
    def is_halted(self) -> bool:
        if not self._enabled:
            return False
        return self._is_halted

    @property
    def status(self) -> KillSwitchStatus:
        live_sharpe = self._compute_sharpe()
        live_slippage = self._compute_avg_slippage()
        sharpe_pct = (live_sharpe / self._backtest_sharpe * 100) if self._backtest_sharpe > 0 else 0
        slip_ratio = (live_slippage / self._backtest_slippage) if self._backtest_slippage > 0 else 0

        return KillSwitchStatus(
            is_halted=self._is_halted,
            reason=self._halt_reason,
            live_sharpe=live_sharpe,
            backtest_sharpe=self._backtest_sharpe,
            sharpe_ratio_pct=sharpe_pct,
            live_slippage_bps=live_slippage,
            backtest_slippage_bps=self._backtest_slippage,
            slippage_ratio=slip_ratio,
        )

    def record_trade(self, return_pct: float, slippage_bps: float = 0.0) -> None:
        """Record a completed trade's return and slippage."""
        self._returns.append(return_pct)
        if slippage_bps > 0:
            self._slippages.append(slippage_bps)

        # Re-evaluate after each trade
        self._evaluate()

    def _evaluate(self) -> None:
        """Check if kill switch should trigger."""
        if not self._enabled or len(self._returns) < 5:
            return

        live_sharpe = self._compute_sharpe()
        live_slippage = self._compute_avg_slippage()

        # Check 1: Sharpe degradation
        if self._backtest_sharpe > 0:
            sharpe_ratio = live_sharpe / self._backtest_sharpe
            if sharpe_ratio < self._sharpe_threshold_pct:
                self._halt(
                    f"Sharpe degradation: live={live_sharpe:.2f} vs "
                    f"backtest={self._backtest_sharpe:.2f} "
                    f"({sharpe_ratio:.0%} < {self._sharpe_threshold_pct:.0%})"
                )
                return

        # Check 2: Slippage blowout
        if self._backtest_slippage > 0 and len(self._slippages) >= 5:
            slippage_ratio = live_slippage / self._backtest_slippage
            if slippage_ratio > self._slippage_max_mult:
                self._halt(
                    f"Slippage blowout: live={live_slippage:.1f}bps vs "
                    f"modeled={self._backtest_slippage:.1f}bps "
                    f"({slippage_ratio:.1f}x > {self._slippage_max_mult:.1f}x)"
                )
                return

        # If previously halted but now recovered, consider unhalting
        # (conservative: keep halted — manual reset required)

    def _halt(self, reason: str) -> None:
        """Trigger the kill switch."""
        if not self._is_halted:
            self._is_halted = True
            self._halt_reason = reason
            logger.critical("🛑 FINANCIAL KILL SWITCH: %s", reason)

    def reset(self) -> None:
        """Manually reset the kill switch (requires human decision)."""
        self._is_halted = False
        self._halt_reason = ""
        logger.warning("Financial kill switch manually reset.")

    def _compute_sharpe(self) -> float:
        """Compute rolling annualized Sharpe ratio."""
        if len(self._returns) < 2:
            return 0.0

        returns = np.array(self._returns)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)

        if std_ret == 0:
            return float("inf") if mean_ret > 0 else 0.0

        # Annualize assuming ~252 trading days
        daily_sharpe = mean_ret / std_ret
        return daily_sharpe * sqrt(252)

    def _compute_avg_slippage(self) -> float:
        """Compute average slippage in basis points."""
        if not self._slippages:
            return 0.0
        return sum(self._slippages) / len(self._slippages)

    def update_baselines(
        self, backtest_sharpe: float, backtest_slippage: float
    ) -> None:
        """Hot-reload baseline metrics (e.g., after WFO cycle)."""
        self._backtest_sharpe = backtest_sharpe
        self._backtest_slippage = backtest_slippage
        logger.info(
            "Kill switch baselines updated: Sharpe=%.2f, Slippage=%.1f bps",
            backtest_sharpe, backtest_slippage,
        )
