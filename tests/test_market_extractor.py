"""
Unit tests for Stage 1: Market Data Extraction Layer.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.data.market_extractor import MarketExtractor, MarketQuote


@pytest.mark.asyncio
async def test_market_extractor_angel_success():
    extractor = MarketExtractor()

    with patch("src.auth.angel_auth.api_manager.smartApi") as mock_api, \
         patch("src.auth.angel_auth.api_manager._token_data") as mock_token:

        mock_token.is_expired = False
        mock_api.getMarketData.return_value = {
            "status": True,
            "data": {
                "fetched": [
                    {
                        "symbolToken": "99926000",
                        "ltp": 24350.50,
                        "close": 24250.00,
                        "tradeVolume": 15000000,
                        "opnInterest": 20000000,
                        "high": 24400.0,
                        "low": 24200.0,
                    }
                ]
            }
        }

        quote = await extractor.get_live_quote("NIFTY 50")
        assert quote.symbol == "NIFTY 50"
        assert quote.ltp == 24350.50
        assert quote.source == "ANGEL_ONE"
        assert quote.volume == 15000000
        assert quote.change_pct > 0


@pytest.mark.asyncio
async def test_market_extractor_nse_fallback():
    extractor = MarketExtractor()

    # Angel One unavailable
    with patch("src.auth.angel_auth.api_manager.smartApi", None), \
         patch("jugaad_data.nse.NSELive") as mock_nse_cls:

        mock_nse = MagicMock()
        mock_nse_cls.return_value = mock_nse
        mock_nse.stock_quote.return_value = {
            "priceInfo": {
                "lastPrice": 2980.0,
                "pChange": 1.25,
                "intraDayHighLow": {"max": 3000.0, "min": 2950.0}
            },
            "securityWiseDP": {"quantityTraded": 4500000}
        }

        quote = await extractor.get_live_quote("RELIANCE")
        assert quote.symbol == "RELIANCE"
        assert quote.ltp == 2980.0
        assert quote.source == "NSE_BACKUP"
        assert quote.change_pct == 1.25


@pytest.mark.asyncio
async def test_market_extractor_fail_closed():
    extractor = MarketExtractor()

    with patch("src.auth.angel_auth.api_manager.smartApi", None), \
         patch("jugaad_data.nse.NSELive", side_effect=Exception("NSE Down")):

        quote = await extractor.get_live_quote("UNKNOWN_SYM")
        assert quote.source == "UNAVAILABLE"
        assert quote.ltp == 0.0
        assert not quote.is_tradable
