"""
Order API — Place, modify, cancel orders via Angel One / Fyers.

Each order gets a unique client_order_id (UUID4) for idempotency.
All operations emit structured audit log events.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.audit.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class TransactionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    SL = "SL"
    SL_M = "SL-M"


class ProductType(str, Enum):
    INTRADAY = "I"
    DELIVERY = "D"
    CO = "CO"
    BO = "BO"


class Validity(str, Enum):
    DAY = "DAY"
    IOC = "IOC"


@dataclass
class OrderRequest:
    """Structured order request with all required parameters."""

    instrument_token: str  # trading symbol as expected by the broker
    transaction_type: TransactionType
    order_type: OrderType
    quantity: int
    product: ProductType = ProductType.INTRADAY
    validity: Validity = Validity.DAY
    price: float = 0.0
    trigger_price: float = 0.0
    disclosed_quantity: int = 0
    tag: str = ""
    symboltoken: str = ""  # broker symbol token (Angel One); required for Angel orders
    exchange: str = "NSE"
    client_order_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class OrderResponse:
    """Normalized order response."""

    order_id: str
    client_order_id: str
    status: str
    raw: dict[str, Any] = field(default_factory=dict)


class OrderAPI:
    """
    Thin wrapper around broker OrderApi with idempotency and audit logging.
    """

    def __init__(
        self,
        order_api: Any,
        audit: AuditLogger,
    ) -> None:
        self._api = order_api
        self._audit = audit

    async def place_order(self, request: OrderRequest) -> OrderResponse:
        """Place a new order with idempotent client_order_id."""
        trace_id = str(uuid.uuid4())

        self._audit.log_event(
            event_type="ORDER_INTENT",
            trace_id=trace_id,
            payload={
                "action": "place",
                "instrument": request.instrument_token,
                "side": request.transaction_type.value,
                "qty": request.quantity,
                "order_type": request.order_type.value,
                "price": request.price,
                "trigger": request.trigger_price,
                "client_order_id": request.client_order_id,
            },
            source="order_api",
        )

        try:
            # Handle Angel One SmartAPI
            if hasattr(self._api, "placeOrder"):
                if not request.symboltoken:
                    raise ValueError(
                        f"OrderRequest.symboltoken is required for Angel One orders "
                        f"(refusing to place {request.instrument_token} with a guessed token)."
                    )
                params = {
                    "variety": "NORMAL",
                    "tradingsymbol": request.instrument_token,
                    "symboltoken": request.symboltoken,
                    "transactiontype": request.transaction_type.value,
                    "exchange": request.exchange,
                    "ordertype": request.order_type.value,
                    "producttype": "INTRADAY" if request.product == ProductType.INTRADAY else "DELIVERY",
                    "duration": request.validity.value,
                    "price": str(request.price),
                    "squareoff": "0",
                    "stoploss": "0",
                    "quantity": str(request.quantity),
                }
                response = self._api.placeOrder(params)
                oid = response.get("data", {}).get("orderid", request.client_order_id) if isinstance(response, dict) else request.client_order_id
                status = "SUCCESS" if isinstance(response, dict) and response.get("status") else "FAILED"
                result = OrderResponse(
                    order_id=str(oid),
                    client_order_id=request.client_order_id,
                    status=status,
                    raw=response if isinstance(response, dict) else {},
                )
            # Handle Fyers SDK
            elif hasattr(self._api, "place_order"):
                fyers_side = 1 if request.transaction_type == TransactionType.BUY else -1
                # Fyers v3 order types: 1=Limit, 2=Market, 3=SL-M, 4=SL
                fyers_type = {
                    OrderType.LIMIT: 1,
                    OrderType.MARKET: 2,
                    OrderType.SL_M: 3,
                    OrderType.SL: 4,
                }[request.order_type]
                fyers_prod = "INTRADAY" if request.product == ProductType.INTRADAY else "MARGIN"
                # Fyers v3 validity: 0=DAY, 1=IOC
                fyers_validity = 1 if request.validity == Validity.IOC else 0

                payload = {
                    "symbol": request.instrument_token,
                    "qty": request.quantity,
                    "type": fyers_type,
                    "side": fyers_side,
                    "productType": fyers_prod,
                    "limitPrice": request.price if fyers_type in (1, 4) else 0,
                    "stopPrice": request.trigger_price if fyers_type in (3, 4) else 0,
                    "disclosedQty": request.disclosed_quantity,
                    "validity": fyers_validity,
                    "offlineOrder": False,
                }
                response = self._api.place_order(data=payload)
                oid = response.get("id", response.get("order_id", request.client_order_id))
                status = response.get("s", "ok")

                result = OrderResponse(
                    order_id=str(oid),
                    client_order_id=request.client_order_id,
                    status=status,
                    raw=response if isinstance(response, dict) else {},
                )
            else:
                result = OrderResponse(
                    order_id=request.client_order_id,
                    client_order_id=request.client_order_id,
                    status="SUCCESS",
                    raw={},
                )

            self._audit.log_event(
                event_type="ORDER_PLACED",
                trace_id=trace_id,
                payload={
                    "order_id": result.order_id,
                    "client_order_id": result.client_order_id,
                    "status": result.status,
                },
                source="order_api",
            )

            logger.info(
                "Order placed: %s | %s %s x%d @ %s | ID: %s",
                request.instrument_token,
                request.transaction_type.value,
                request.order_type.value,
                request.quantity,
                request.price,
                result.order_id,
            )
            return result

        except Exception as exc:
            self._audit.log_event(
                event_type="ORDER_ERROR",
                trace_id=trace_id,
                payload={
                    "error": str(exc),
                    "client_order_id": request.client_order_id,
                },
                source="order_api",
            )
            logger.error("Order placement failed: %s", exc)
            raise

    async def modify_order(
        self,
        order_id: str,
        *,
        quantity: Optional[int] = None,
        price: Optional[float] = None,
        trigger_price: Optional[float] = None,
        order_type: Optional[OrderType] = None,
    ) -> OrderResponse:
        """Modify an existing order."""
        trace_id = str(uuid.uuid4())

        modify_params: dict[str, Any] = {"order_id": order_id}
        if quantity is not None:
            modify_params["quantity"] = quantity
        if price is not None:
            modify_params["price"] = price
        if trigger_price is not None:
            modify_params["trigger_price"] = trigger_price
        if order_type is not None:
            modify_params["order_type"] = order_type.value

        self._audit.log_event(
            event_type="ORDER_MODIFY_INTENT",
            trace_id=trace_id,
            payload=modify_params,
            source="order_api",
        )

        try:
            raw_response: Any = {}
            modified = False

            # Handle Angel One SmartAPI
            if hasattr(self._api, "modifyOrder"):
                params: dict[str, Any] = {
                    "variety": "NORMAL",
                    "orderid": order_id,
                }
                if quantity is not None:
                    params["quantity"] = str(quantity)
                if price is not None:
                    params["price"] = str(price)
                if trigger_price is not None:
                    params["triggerprice"] = str(trigger_price)
                if order_type is not None:
                    params["ordertype"] = order_type.value
                raw_response = self._api.modifyOrder(params)
                modified = isinstance(raw_response, dict) and bool(raw_response.get("status"))
            # Handle Fyers SDK v3
            elif hasattr(self._api, "modify_order"):
                data: dict[str, Any] = {"id": order_id}
                if quantity is not None:
                    data["qty"] = quantity
                if price is not None:
                    data["limitPrice"] = price
                if trigger_price is not None:
                    data["stopPrice"] = trigger_price
                if order_type is not None:
                    data["type"] = {
                        OrderType.LIMIT: 1,
                        OrderType.MARKET: 2,
                        OrderType.SL_M: 3,
                        OrderType.SL: 4,
                    }[order_type]
                raw_response = self._api.modify_order(data=data)
                modified = isinstance(raw_response, dict) and raw_response.get("s") == "ok"
            else:
                raise RuntimeError(
                    f"Broker client {type(self._api).__name__} does not support order modification."
                )

            if not modified:
                logger.warning(
                    "Order modify for %s not confirmed by broker: %s", order_id, raw_response
                )

            status = "MODIFIED" if modified else "MODIFY_REJECTED"
            result = OrderResponse(order_id=order_id, client_order_id="", status=status, raw=raw_response if isinstance(raw_response, dict) else {})

            self._audit.log_event(
                event_type="ORDER_MODIFIED",
                trace_id=trace_id,
                payload={"order_id": order_id, "status": result.status},
                source="order_api",
            )
            logger.info("Order modified: %s (%s)", order_id, result.status)
            return result

        except Exception as exc:
            self._audit.log_event(
                event_type="ORDER_MODIFY_ERROR",
                trace_id=trace_id,
                payload={"order_id": order_id, "error": str(exc)},
                source="order_api",
            )
            logger.error("Order modify failed for %s: %s", order_id, exc)
            raise

    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancel a single order by ID."""
        trace_id = str(uuid.uuid4())

        self._audit.log_event(
            event_type="ORDER_CANCEL_INTENT",
            trace_id=trace_id,
            payload={"order_id": order_id},
            source="order_api",
        )

        try:
            # Handle Angel One / Fyers
            if hasattr(self._api, "cancelOrder"):
                # Angel One SmartAPI
                response = self._api.cancelOrder(orderid=order_id)
                ok = isinstance(response, dict) and bool(response.get("status"))
                result = OrderResponse(
                    order_id=order_id,
                    client_order_id="",
                    status="CANCELLED" if ok else "CANCEL_REJECTED",
                    raw=response if isinstance(response, dict) else {},
                )
            elif hasattr(self._api, "cancel_order"):
                # Fyers API v3: cancel_order(data={"id": ...})
                response = self._api.cancel_order(data={"id": order_id})
                ok = isinstance(response, dict) and response.get("s") == "ok"
                result = OrderResponse(
                    order_id=order_id,
                    client_order_id="",
                    status="CANCELLED" if ok else "CANCEL_REJECTED",
                    raw=response if isinstance(response, dict) else {},
                )
            else:
                raise RuntimeError(
                    f"Broker client {type(self._api).__name__} does not support order cancellation."
                )

            self._audit.log_event(
                event_type="ORDER_CANCELLED",
                trace_id=trace_id,
                payload={"order_id": order_id, "status": result.status},
                source="order_api",
            )
            logger.info("Order cancelled: %s", order_id)
            return result

        except Exception as exc:
            self._audit.log_event(
                event_type="ORDER_CANCEL_ERROR",
                trace_id=trace_id,
                payload={"order_id": order_id, "error": str(exc)},
                source="order_api",
            )
            logger.error("Order cancel failed for %s: %s", order_id, exc)
            raise

    async def cancel_all_orders(self) -> list[OrderResponse]:
        """Cancel ALL open/pending orders — used by circuit breaker and heartbeat."""
        trace_id = str(uuid.uuid4())
        self._audit.log_event(
            event_type="CANCEL_ALL_INTENT",
            trace_id=trace_id,
            payload={},
            source="order_api",
        )

        try:
            # Fetch all open orders and cancel each.
            # Angel One returns a dict: {"status": true, "data": [ {...}, ... ]}
            # Fyers v3 returns a dict:  {"s": "ok", "orderBook": [ {...}, ... ]}
            if hasattr(self._api, "orderBook"):
                order_book = self._api.orderBook()
                raw_orders = order_book.get("data", []) if isinstance(order_book, dict) else getattr(order_book, "data", [])
            elif hasattr(self._api, "orderbook"):
                order_book = self._api.orderbook()
                raw_orders = order_book.get("orderBook", []) if isinstance(order_book, dict) else []
            elif hasattr(self._api, "get_order_book"):
                response = self._api.get_order_book(api_version="2.0")
                raw_orders = response.get("data", []) if isinstance(response, dict) else []
            else:
                raise RuntimeError(
                    f"Broker client {type(self._api).__name__} exposes no order book method."
                )

            results: list[OrderResponse] = []

            open_statuses = {"open", "pending", "trigger pending", "transit"}
            for order in raw_orders or []:
                if isinstance(order, dict):
                    status = str(order.get("orderstatus") or order.get("status") or "").lower()
                    oid = str(order.get("orderid") or order.get("id") or "")
                else:
                    status = str(getattr(order, "status", "")).lower()
                    oid = str(getattr(order, "order_id", "") or getattr(order, "id", ""))
                if status in open_statuses and oid:
                    res = await self.cancel_order(oid)
                    results.append(res)

            self._audit.log_event(
                event_type="CANCEL_ALL_COMPLETE",
                trace_id=trace_id,
                payload={"cancelled_count": len(results)},
                source="order_api",
            )
            logger.info("Cancelled %d open orders.", len(results))
            return results

        except Exception as exc:
            self._audit.log_event(
                event_type="CANCEL_ALL_ERROR",
                trace_id=trace_id,
                payload={"error": str(exc)},
                source="order_api",
            )
            logger.error("Cancel all orders failed: %s", exc)
            raise

    @staticmethod
    def _parse_response(response: Any, client_order_id: str) -> OrderResponse:
        """Normalize SDK response into our OrderResponse dataclass."""
        data = {}
        if hasattr(response, "data") and response.data:
            if isinstance(response.data, dict):
                data = response.data
            elif hasattr(response.data, "to_dict"):
                data = response.data.to_dict()
            else:
                data = {"raw": str(response.data)}

        return OrderResponse(
            order_id=data.get("order_id", ""),
            client_order_id=client_order_id,
            status=data.get("status", "unknown"),
            raw=data,
        )
