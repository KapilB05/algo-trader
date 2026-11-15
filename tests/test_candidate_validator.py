"""
Unit tests for Stage 5: Backtest / WFO Candidate Validation Engine.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.ml.research_scorer import ResearchCandidate
from src.wfo.candidate_validator import CandidateValidator, ValidatedTradeIdea


def test_validate_strong_candidate_passes():
    validator = CandidateValidator()

    strong_cand = ResearchCandidate(
        symbol="BANKNIFTY FUT",
        direction="BULLISH",
        research_score=85.0,
        confidence_pct=85.0,
        sub_scores={"trend": 28.0, "volume_oi": 25.0, "volatility": 16.0, "confluence": 15.0, "news_modifier": 5.0},
        key_reasons=["Strong trend", "High volume"],
        suitable_classes=["INTRADAY", "FUTURES"],
        ltp=57600.0,
        volume=5000000,
        oi=4000000,
        data_source="ANGEL_ONE",
        timestamp="09:30:00",
    )

    idea = validator.validate_candidate(strong_cand)
    assert idea.validation_status in ("APPROVED", "REJECTED")
    assert idea.win_rate_pct > 0.0
    assert idea.profit_factor > 0.0
    assert idea.max_drawdown_pct > 0.0
    assert idea.risk_reward_ratio >= 1.50
    assert idea.expected_value_pct != 0.0
    assert idea.recommended_entry == 57600.0
    assert idea.stop_loss < idea.recommended_entry
    assert idea.target > idea.recommended_entry


def test_validate_neutral_candidate_rejected():
    validator = CandidateValidator()

    weak_cand = ResearchCandidate(
        symbol="TCS",
        direction="NEUTRAL",
        research_score=45.0,
        confidence_pct=45.0,
        sub_scores={"trend": 10.0, "volume_oi": 5.0, "volatility": 16.0, "confluence": 14.0, "news_modifier": 0.0},
        key_reasons=["Consolidation"],
        suitable_classes=["EQUITY"],
        ltp=3850.0,
        volume=20000,
        oi=0,
        data_source="NSE_BACKUP",
        timestamp="09:30:00",
    )

    idea = validator.validate_candidate(weak_cand)
    assert not idea.passed_validation
    assert idea.validation_status == "REJECTED"
    assert idea.rejection_reason is not None
    assert "Neutral direction" in idea.rejection_reason or "below confidence gate" in idea.rejection_reason


@pytest.mark.asyncio
async def test_validate_all_candidates_sorting():
    validator = CandidateValidator()

    cand1 = ResearchCandidate("RELIANCE", "BULLISH", 88.0, 88.0, {}, [], ["EQUITY"], 2980.0, 5000000, 2000000, "ANGEL_ONE", "09:30:00")
    cand2 = ResearchCandidate("INFY", "NEUTRAL", 42.0, 42.0, {}, [], ["EQUITY"], 1780.0, 20000, 0, "NSE_BACKUP", "09:30:00")

    with patch("src.ml.research_scorer.research_scoring_engine.score_watchlist", new_callable=AsyncMock) as mock_score:
        mock_score.return_value = [cand1, cand2]

        res = await validator.validate_all_candidates()
        assert res["status"] == "SUCCESS"
        assert res["total_candidates"] == 2
        assert res["approved_count"] + res["rejected_count"] == 2
        assert len(res["all_ideas"]) == 2
