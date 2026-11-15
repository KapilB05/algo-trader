"""
Unit tests for Index Futures & Options data provider (src/data/index_fo_data.py).
"""

from src.data.index_fo_data import (
    calculate_max_pain,
    fetch_futures_data,
    fetch_historical_candles,
    fetch_live_option_chain,
)


def test_live_option_chain() -> None:
    for sym in ["NIFTY", "BANKNIFTY", "FINNIFTY"]:
        chain = fetch_live_option_chain(sym)
        assert chain["status"] == "SUCCESS"
        assert chain["symbol"] == sym
        assert chain["pcr"] > 0
        assert chain["max_pain"] > 0
        assert len(chain["strikes"]) >= 11
        assert "available_expiries" in chain
        assert len(chain["available_expiries"]) >= 1

        # Check ATM strike is present
        atm_found = any(s["is_atm"] for s in chain["strikes"])
        assert atm_found

    # Test with explicit expiry
    chain_expiry = fetch_live_option_chain("NIFTY", expiry=chain["available_expiries"][0])
    assert chain_expiry["selected_expiry"] == chain["available_expiries"][0]


def test_futures_data() -> None:
    fut = fetch_futures_data("BANKNIFTY")
    assert fut["status"] == "SUCCESS"
    assert fut["symbol"] == "BANKNIFTY FUT"
    assert fut["spot_price"] > 0
    assert fut["futures_price"] > 0
    assert "basis" in fut


def test_historical_candles_indicators() -> None:
    res = fetch_historical_candles(symbol="BANKNIFTY FUT", interval="5m")
    assert res["status"] == "SUCCESS"
    candles = res["candles"]
    assert len(candles) > 0

    first = candles[0]
    assert "open" in first
    assert "high" in first
    assert "low" in first
    assert "close" in first
    assert "volume" in first
    assert "ema9" in first
    assert "ema21" in first
    assert "vwap" in first


def test_max_pain_calculation() -> None:
    strikes = [24200.0, 24300.0, 24400.0]
    call_oi = {24200.0: 1000, 24300.0: 5000, 24400.0: 10000}
    put_oi = {24200.0: 10000, 24300.0: 5000, 24400.0: 1000}

    max_pain = calculate_max_pain(strikes, call_oi, put_oi)
    assert max_pain == 24300.0
