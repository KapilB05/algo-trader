"""
Unit tests for Stage 4: ML / Research Layer & Scoring Engine.
"""

from unittest.mock import AsyncMock, patch
import pytest

from src.ml.research_scorer import ResearchCandidate, ResearchScoringEngine


@pytest.mark.asyncio
async def test_evaluate_bullish_instrument():
    engine = ResearchScoringEngine()

    item = {
        "symbol": "RELIANCE",
        "ltp": 2980.0,
        "change_pct": 1.25,
        "volume": 6500000,
        "oi": 3200000,
        "sentiment": "POSITIVE",
        "sentiment_score": 0.5,
        "data_source": "ANGEL_ONE",
        "category": "stocks",
    }

    cand = await engine.evaluate_instrument(item)
    assert cand is not None
    assert cand.symbol == "RELIANCE"
    assert cand.direction == "BULLISH"
    assert cand.research_score >= 70.0
    assert "trend" in cand.sub_scores
    assert "volume_oi" in cand.sub_scores
    assert cand.sub_scores["news_modifier"] > 0.0  # Positive news boost
    assert len(cand.key_reasons) >= 3


@pytest.mark.asyncio
async def test_evaluate_bearish_instrument():
    engine = ResearchScoringEngine()

    item = {
        "symbol": "HDFCBANK",
        "ltp": 1640.0,
        "change_pct": -0.85,
        "volume": 8500000,
        "oi": 4100000,
        "sentiment": "NEGATIVE",
        "sentiment_score": -0.4,
        "data_source": "ANGEL_ONE",
        "category": "stocks",
    }

    cand = await engine.evaluate_instrument(item)
    assert cand is not None
    assert cand.symbol == "HDFCBANK"
    assert cand.direction == "BEARISH"
    assert cand.research_score >= 70.0
    assert cand.sub_scores["news_modifier"] > 0.0  # Negative news confirms short bias


@pytest.mark.asyncio
async def test_news_cannot_create_signal_alone():
    engine = ResearchScoringEngine()

    # Flat price movement (+0.0%) with positive news
    item = {
        "symbol": "TCS",
        "ltp": 3850.0,
        "change_pct": 0.0,
        "volume": 20000,
        "oi": 0,
        "sentiment": "POSITIVE",
        "sentiment_score": 0.9,
        "data_source": "NSE_BACKUP",
        "category": "stocks",
    }

    cand = await engine.evaluate_instrument(item)
    assert cand is not None
    assert cand.direction == "NEUTRAL"
    assert cand.research_score < 50.0  # Fails quality threshold due to flat structure


@pytest.mark.asyncio
async def test_score_watchlist_ranking():
    engine = ResearchScoringEngine()

    mock_wl_data = {
        "categories": {
            "stocks": [
                {"symbol": "RELIANCE", "ltp": 2980.0, "change_pct": 1.5, "volume": 10000000, "oi": 5000000, "sentiment": "POSITIVE", "sentiment_score": 0.6, "data_source": "ANGEL_ONE", "category": "stocks"},
                {"symbol": "INFY", "ltp": 1780.0, "change_pct": 0.3, "volume": 1000000, "oi": 500000, "sentiment": "NEUTRAL", "sentiment_score": 0.0, "data_source": "ANGEL_ONE", "category": "stocks"},
            ],
            "futures": [],
            "options": []
        }
    }

    with patch("src.data.algo_watchlist.algo_watchlist_engine.get_segregated_watchlist", new_callable=AsyncMock) as mock_wl:
        mock_wl.return_value = mock_wl_data

        ranked = await engine.score_watchlist()
        assert len(ranked) >= 1
        assert ranked[0].symbol == "RELIANCE"
        assert ranked[0].research_score > ranked[1].research_score
