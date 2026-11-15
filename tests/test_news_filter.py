"""
Unit tests for Stage 2: News Filter & Sentiment Scoring Engine.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.data.news_filter import NewsFilterEngine


@pytest.mark.asyncio
async def test_news_classification_positive():
    engine = NewsFilterEngine()
    text = "Reliance Q1 profit jumps 25%, shares rally to record high on earnings beat"
    sentiment, score = engine.classify_text(text)
    assert sentiment == "POSITIVE"
    assert score > 0.3


@pytest.mark.asyncio
async def test_news_classification_negative():
    engine = NewsFilterEngine()
    text = "BankNifty plunges 500 points amid heavy selloff and inflation surge"
    sentiment, score = engine.classify_text(text)
    assert sentiment == "NEGATIVE"
    assert score < -0.3


@pytest.mark.asyncio
async def test_news_classification_neutral():
    engine = NewsFilterEngine()
    text = "Markets trade sideways ahead of monetary policy committee outcome"
    sentiment, score = engine.classify_text(text)
    assert sentiment == "NEUTRAL"
    assert score == 0.0


def test_confidence_modifier_logic():
    engine = NewsFilterEngine()
    # Positive news on BUY trade -> confidence boost
    mod_buy_pos = engine.compute_confidence_modifier(0.8, "BUY")
    assert mod_buy_pos > 0.0

    # Negative news on BUY trade -> confidence penalty
    mod_buy_neg = engine.compute_confidence_modifier(-0.8, "BUY")
    assert mod_buy_neg < 0.0

    # Negative news on SELL trade -> confidence boost
    mod_sell_neg = engine.compute_confidence_modifier(-0.8, "SELL")
    assert mod_sell_neg > 0.0


@pytest.mark.asyncio
async def test_google_news_rss_mock():
    engine = NewsFilterEngine()
    sample_rss = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Google News</title>
        <item>
          <title>Nifty hits record high led by banking rally and rate cut hopes</title>
          <link>https://news.google.com/articles/123</link>
          <pubDate>Wed, 19 Aug 2026 06:00:00 GMT</pubDate>
        </item>
      </channel>
    </rss>"""

    with patch("urllib.request.urlopen") as mock_url:
        mock_resp = MagicMock()
        mock_resp.read.return_value = sample_rss
        mock_resp.__enter__.return_value = mock_resp
        mock_url.return_value = mock_resp

        summary = await engine.get_symbol_news("NIFTY")
        assert summary.symbol == "NIFTY"
        assert summary.article_count == 1
        assert summary.overall_sentiment == "POSITIVE"
        assert summary.confidence_modifier > 0
