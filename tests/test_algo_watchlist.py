"""
Unit tests for Stage 3: Dynamic Algo Watchlist Engine.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.data.algo_watchlist import AlgoWatchlistEngine
from src.data.market_extractor import MarketQuote
from src.data.news_filter import SymbolNewsSummary


@pytest.mark.asyncio
async def test_algo_watchlist_segregation():
    engine = AlgoWatchlistEngine()

    fake_quotes = {
        "NIFTY 50": MarketQuote("NIFTY 50", "INDEX", 24300.0, 0.5, 100000, 200000, 24350.0, 24250.0, "ANGEL_ONE", "09:30:00"),
        "BANKNIFTY FUT": MarketQuote("BANKNIFTY FUT", "FUTURES", 57600.0, 0.8, 50000, 80000, 57700.0, 57500.0, "ANGEL_ONE", "09:30:00"),
        "RELIANCE": MarketQuote("RELIANCE", "STOCK", 2980.0, 1.2, 40000, 60000, 3000.0, 2960.0, "ANGEL_ONE", "09:30:00"),
    }

    fake_news = SymbolNewsSummary(
        symbol="NIFTY",
        overall_sentiment="POSITIVE",
        sentiment_score=0.6,
        article_count=3,
        articles=[],
        confidence_modifier=0.05,
        updated_at="09:30:00",
    )

    fake_chain = {
        "symbol": "BANKNIFTY",
        "atm_strike": 57500,
        "source": "ANGEL_ONE",
        "strikes": [
            {
                "strike": 57500,
                "is_atm": True,
                "ce_ltp": 320.0,
                "ce_volume": 10000,
                "ce_oi": 20000,
                "ce_oi_chg_pct": 5.0,
                "ce_symbol": "BANKNIFTY57500CE",
                "pe_ltp": 150.0,
                "pe_volume": 8000,
                "pe_oi": 15000,
                "pe_oi_chg_pct": -4.0,
                "pe_symbol": "BANKNIFTY57500PE",
            }
        ]
    }

    with patch("src.data.market_extractor.market_extractor.get_multi_quotes", new_callable=AsyncMock) as mock_mq, \
         patch("src.data.news_filter.news_filter_engine.get_symbol_news", new_callable=AsyncMock) as mock_news, \
         patch("src.data.algo_watchlist.fetch_live_option_chain") as mock_chain:

        mock_mq.return_value = fake_quotes
        mock_news.return_value = fake_news
        mock_chain.return_value = fake_chain

        res = await engine.get_segregated_watchlist()

        assert res["status"] == "SUCCESS"
        assert "categories" in res
        cats = res["categories"]

        assert "indices" in cats
        assert "futures" in cats
        assert "stocks" in cats
        assert "options" in cats

        # Check options populated
        assert len(cats["options"]) >= 2
        ce_opt = next((o for o in cats["options"] if "CE" in o["symbol"]), None)
        assert ce_opt is not None
        assert ce_opt["ltp"] == 320.0
        assert ce_opt["category"] == "options"
