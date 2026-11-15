"""
Portfolio API — Positions, Holdings, and Funds queries.

Returns normalized dataclass objects for internal consumption.
Compatible with Angel One SmartAPI and Upstox API clients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Normalized position record."""

    instrument_token: str
    symbol: str
    exchange: str
    product: str
    quantity: int
    average_price: float
    last_price: float
    pnl: float
    day_buy_qty: int = 0
    day_sell_qty: int = 0
    net_qty: int = 0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Holding:
    """Normalized holding record."""

    instrument_token: str
    symbol: str
    exchange: str
    quantity: int
    authorised_quantity: int
    average_price: float
    last_price: float
    pnl: float
    close_price: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarginSummary:
    """Normalized funds / margin record."""

    available_margin: float
    used_margin: float
    total_balance: float
    raw: dict[str, Any] = field(default_factory=dict)


class PortfolioAPI:
    """Wrapper around broker Portfolio API with normalized output."""

    def __init__(self, portfolio_api: Any) -> None:
        self._api = portfolio_api

    async def get_positions(self) -> List[Position]:
        """Fetch all current positions (intraday + carry-forward)."""
        try:
            positions: List[Position] = []

            # 1. Angel One SmartAPI style
            if hasattr(self._api, "position") and callable(self._api.position):
                res = self._api.position()
                if isinstance(res, dict) and res.get("status") is False:
                    msg = str(res.get("message", ""))
                    code = str(res.get("errorCode", ""))
                    if code in ("AG8001", "AG8003") or "Token" in msg:
                        logger.warning("Token expired/missing in get_positions. Re-authenticating...")
                        from src.auth.angel_auth import api_manager
                        if api_manager.authenticate():
                            res = self._api.position()

                raw_items = []
                if isinstance(res, dict) and res.get("status") and res.get("data"):
                    data = res.get("data")
                    if isinstance(data, list):
                        raw_items = data

                for p in raw_items:
                    qty = int(p.get("netqty", p.get("quantity", 0)) or 0)
                    avg_p = float(p.get("averageprice", p.get("avgprice", 0.0)) or 0.0)
                    ltp = float(p.get("ltp", 0.0) or 0.0)
                    pnl = float(p.get("pnl", p.get("realised", 0.0)) or 0.0)

                    positions.append(
                        Position(
                            instrument_token=str(p.get("symboltoken", "")),
                            symbol=str(p.get("tradingsymbol", "")),
                            exchange=str(p.get("exchange", "")),
                            product=str(p.get("producttype", "")),
                            quantity=qty,
                            average_price=avg_p,
                            last_price=ltp,
                            pnl=pnl,
                            net_qty=qty,
                            raw=p,
                        )
                    )
                logger.info("Fetched %d positions via SmartAPI.", len(positions))
                return positions

            # 2. Generic / Upstox style
            if hasattr(self._api, "get_positions") and callable(self._api.get_positions):
                response = self._api.get_positions()
                if hasattr(response, "data") and response.data:
                    items = []
                    if hasattr(response.data, "day"):
                        items.extend(response.data.day or [])
                    elif isinstance(response.data, list):
                        items = response.data

                    for pos in items:
                        raw = pos.to_dict() if hasattr(pos, "to_dict") else {}
                        positions.append(
                            Position(
                                instrument_token=getattr(pos, "instrument_token", ""),
                                symbol=getattr(pos, "trading_symbol", ""),
                                exchange=getattr(pos, "exchange", ""),
                                product=getattr(pos, "product", ""),
                                quantity=getattr(pos, "quantity", 0),
                                average_price=getattr(pos, "average_price", 0.0),
                                last_price=getattr(pos, "last_price", 0.0),
                                pnl=getattr(pos, "pnl", 0.0),
                                day_buy_qty=getattr(pos, "day_buy_quantity", 0),
                                day_sell_qty=getattr(pos, "day_sell_quantity", 0),
                                net_qty=getattr(pos, "quantity", 0),
                                raw=raw,
                            )
                        )
                logger.info("Fetched %d positions.", len(positions))
                return positions

            logger.warning("No position fetching method available on API client.")
            return []

        except Exception as exc:
            # Re-raise: callers must be able to distinguish "no positions"
            # from "could not query broker" (silently returning [] here has
            # previously caused live positions to be treated as stale/closed).
            logger.error("Failed to fetch positions: %s", exc)
            raise

    async def get_holdings(self) -> List[Holding]:
        """Fetch long-term holdings."""
        try:
            holdings: List[Holding] = []

            # 1. Angel One SmartAPI style
            if hasattr(self._api, "holding") and callable(self._api.holding):
                res = self._api.holding()
                raw_items = []
                if isinstance(res, dict) and res.get("status") and res.get("data"):
                    data = res.get("data")
                    if isinstance(data, list):
                        raw_items = data

                for h in raw_items:
                    holdings.append(
                        Holding(
                            instrument_token=str(h.get("symboltoken", "")),
                            symbol=str(h.get("tradingsymbol", "")),
                            exchange=str(h.get("exchange", "")),
                            quantity=int(h.get("quantity", 0) or 0),
                            authorised_quantity=int(h.get("authorisedquantity", 0) or 0),
                            average_price=float(h.get("averageprice", 0.0) or 0.0),
                            last_price=float(h.get("ltp", 0.0) or 0.0),
                            pnl=float(h.get("profitandloss", h.get("pnl", 0.0)) or 0.0),
                            close_price=float(h.get("close", 0.0) or 0.0),
                            raw=h,
                        )
                    )
                logger.info("Fetched %d holdings via SmartAPI.", len(holdings))
                return holdings

            # 2. Generic / Upstox style
            if hasattr(self._api, "get_holdings") and callable(self._api.get_holdings):
                response = self._api.get_holdings()
                if hasattr(response, "data") and response.data:
                    items = response.data if isinstance(response.data, list) else []
                    for hld in items:
                        raw = hld.to_dict() if hasattr(hld, "to_dict") else {}
                        holdings.append(
                            Holding(
                                instrument_token=getattr(hld, "instrument_token", ""),
                                symbol=getattr(hld, "trading_symbol", ""),
                                exchange=getattr(hld, "exchange", ""),
                                quantity=getattr(hld, "quantity", 0),
                                authorised_quantity=getattr(hld, "authorised_quantity", 0),
                                average_price=getattr(hld, "average_price", 0.0),
                                last_price=getattr(hld, "last_price", 0.0),
                                pnl=getattr(hld, "pnl", 0.0),
                                raw=raw,
                            )
                        )
                logger.info("Fetched %d holdings.", len(holdings))
                return holdings

            logger.warning("No holding fetching method available on API client.")
            return []

        except Exception as exc:
            logger.error("Failed to fetch holdings: %s", exc)
            raise

    async def get_funds(self) -> MarginSummary:
        """Fetch available margin and balance."""
        try:
            # 1. Angel One SmartAPI style
            if hasattr(self._api, "rmsLimit") and callable(self._api.rmsLimit):
                res = self._api.rmsLimit()
                if isinstance(res, dict) and res.get("status") is False:
                    msg = str(res.get("message", ""))
                    code = str(res.get("errorCode", ""))
                    if code in ("AG8001", "AG8003") or "Token" in msg:
                        logger.warning("Token expired/missing in get_funds. Re-authenticating...")
                        from src.auth.angel_auth import api_manager
                        if api_manager.authenticate():
                            res = self._api.rmsLimit()

                data = res.get("data", {}) if (isinstance(res, dict) and res.get("status")) else {}
                avail_cash = float(data.get("availablecash", data.get("net", 0.0)) or 0.0)
                used_margin = float(data.get("utiliseddebits", 0.0) or 0.0)
                return MarginSummary(
                    available_margin=avail_cash,
                    used_margin=used_margin,
                    total_balance=avail_cash + used_margin,
                    raw=data,
                )

            # 2. Generic / Upstox style
            if hasattr(self._api, "get_user_fund_margin") and callable(self._api.get_user_fund_margin):
                response = self._api.get_user_fund_margin()
                data_dict: dict[str, Any] = {}

                if hasattr(response, "data") and response.data:
                    if hasattr(response.data, "to_dict"):
                        data_dict = response.data.to_dict()
                    elif isinstance(response.data, dict):
                        data_dict = response.data

                equity = data_dict.get("equity", {})
                available = float(equity.get("available_margin", 0.0))
                used = float(equity.get("used_margin", 0.0))

                return MarginSummary(
                    available_margin=available,
                    used_margin=used,
                    total_balance=available + used,
                    raw=data_dict,
                )

            return MarginSummary(available_margin=0.0, used_margin=0.0, total_balance=0.0)

        except Exception as exc:
            logger.error("Failed to fetch funds: %s", exc)
            raise

    async def get_total_pnl(self) -> float:
        """Calculate total unrealized P&L across all positions."""
        positions = await self.get_positions()
        return sum(p.pnl for p in positions)

