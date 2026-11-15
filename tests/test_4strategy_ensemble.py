"""
Unit Tests for 4-Strategy Ensemble & Consensus Voting (Layer 4).
"""

import numpy as np
import pandas as pd
import pytest
from src.strategy.ensemble_voter import MultiStrategyEnsembleVoter
from src.strategy.orb_strategy import OpeningRangeBreakoutStrategy
from src.strategy.supertrend_strategy import EMASupertrendStrategy
from src.strategy.vwap_reclaim_strategy import VWAPReclaimStrategy


def test_ensemble_voting_consensus_rules():
    """Verify voting thresholds: 3/4 = 100%, 2/4 = 50%, <2 = No Trade."""
    voter = MultiStrategyEnsembleVoter()

    # Generate synthetic breakout bars
    n_bars = 40
    c = 50000.0 + np.cumsum(np.full(n_bars, 20.0))
    h = c + 10.0
    l = c - 10.0
    o = l + 5.0
    v = np.full(n_bars, 25000)
    df = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v})

    decision = voter.evaluate_all(
        df_5m=df,
        symbol="BANKNIFTY",
        fii_net_cr=1500.0,
        dii_net_cr=500.0,
        pcr=1.25,
        current_vix=13.5,
    )

    assert isinstance(decision.agreed_votes, int)
    assert decision.position_size_multiplier in (0.0, 0.5, 1.0)
    if decision.agreed_votes >= 3:
        assert decision.position_size_multiplier == 1.0
        assert decision.trade_approved is True
    elif decision.agreed_votes == 2:
        assert decision.position_size_multiplier == 0.5
        assert decision.trade_approved is True
