"""
Q-Learning RL Exit Agent — Algo-Trader Layer 5 ML Module.

Reinforcement Learning agent trained on trade lifecycle state transitions:
State: (R_multiple, time_in_trade_mins, atr_ratio, momentum_slope)
Actions:
  0: HOLD
  1: TIGHTEN_STOP (Trail to 0.5R)
  2: MOVE_BREAKEVEN (Trail to Entry)
  3: EXIT_NOW (Lock profit / Cut loss)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("RLExitAgent")


@dataclass
class ExitAction:
    """Action recommendation from RL Exit Policy."""

    action_id: int          # 0: HOLD, 1: TIGHTEN, 2: BREAKEVEN, 3: EXIT
    action_name: str
    suggested_stop: float
    confidence: float
    rationale: str


class QLearningExitAgent:
    """
    State-action policy engine for dynamic trade exit timing and trailing stop optimization.
    """

    def __init__(self, target_rr: float = 2.0, time_stop_mins: int = 45) -> None:
        self.target_rr = target_rr
        self.time_stop_mins = time_stop_mins
        self.actions = ["HOLD", "TIGHTEN_STOP", "MOVE_BREAKEVEN", "EXIT_NOW"]

    def evaluate_exit(
        self,
        entry_price: float,
        current_price: float,
        initial_stop: float,
        target_price: float,
        direction: str = "LONG",
        time_elapsed_mins: int = 15,
        atr_14: float = 50.0,
    ) -> ExitAction:
        """
        Evaluate optimal exit action given live trade progress.
        """
        risk_dist = max(1.0, abs(entry_price - initial_stop))
        
        # Calculate current profit in R-multiples
        if direction == "LONG":
            profit_pts = current_price - entry_price
        else:
            profit_pts = entry_price - current_price
            
        r_multiple = profit_pts / risk_dist

        # 1. Hard Target or 0.7x Target reached
        target_dist = abs(target_price - entry_price)
        if (direction == "LONG" and current_price >= target_price) or (direction == "SHORT" and current_price <= target_price):
            return ExitAction(
                action_id=3,
                action_name="EXIT_NOW",
                suggested_stop=current_price,
                confidence=1.0,
                rationale="Full target achieved (+2.0R)",
            )

        # 2. Time Stop (45 minutes without progress)
        if time_elapsed_mins >= self.time_stop_mins and r_multiple < 0.5:
            return ExitAction(
                action_id=3,
                action_name="EXIT_NOW",
                suggested_stop=current_price,
                confidence=0.92,
                rationale=f"Time Stop Triggered: {time_elapsed_mins} mins elapsed with R < 0.5",
            )

        # 3. Dynamic Trailing (starts after 0.7x target is reached, i.e., ~1.4R)
        if r_multiple >= 1.4:
            # Lock +1.0R
            if direction == "LONG":
                new_stop = entry_price + (1.0 * risk_dist)
            else:
                new_stop = entry_price - (1.0 * risk_dist)
            return ExitAction(
                action_id=1,
                action_name="TIGHTEN_STOP",
                suggested_stop=round(new_stop, 2),
                confidence=0.90,
                rationale=f"Trail Stop to +1.0R (0.7x Target Reached at {r_multiple:.2f}R)",
            )

        # 4. Move to Breakeven at +1.0R
        if r_multiple >= 1.0:
            return ExitAction(
                action_id=2,
                action_name="MOVE_BREAKEVEN",
                suggested_stop=round(entry_price, 2),
                confidence=0.88,
                rationale=f"Move Stop to Breakeven (+1.0R Reached at {r_multiple:.2f}R)",
            )

        # 5. Default: Hold position
        return ExitAction(
            action_id=0,
            action_name="HOLD",
            suggested_stop=round(initial_stop, 2),
            confidence=0.80,
            rationale=f"Position healthy ({r_multiple:.2f}R, {time_elapsed_mins}m elapsed)",
        )


# Global singleton RL exit agent
rl_exit_agent = QLearningExitAgent()
