"""
Binance Demo Trading adapter.

Binance Demo Trading (https://demo-trading.binance.com) is fundamentally
different from Binance Testnet:

  Testnet        - synthetic order book, fake prices, zero market depth
  Demo Trading   - mirrors real Binance mainnet order book in real time,
                   but fills are simulated against that real book
                   → realistic spreads, real price discovery, virtual funds

IMPORTANT: Binance Demo Trading does NOT use the standard CCXT sandbox flag
and does NOT expose a standard /api/time endpoint. It requires direct HTTP
requests to https://demo-trading.binance.com/api/v3/... endpoints.

Environment variables
---------------------
  BINANCE_DEMO_API_KEY    API key from demo-trading.binance.com
  BINANCE_DEMO_SECRET     Secret from demo-trading.binance.com

  BINANCE_DEMO_REST_URL   (default https://demo-trading.binance.com)
  BINANCE_DEMO_WS_URL     (default wss://demo-trading.binance.com/ws)
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any

import requests
import hmac
import hashlib
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BINANCE_DEMO_REST_URL: str = os.getenv(
    "BINANCE_DEMO_REST_URL", "https://demo-trading.binance.com"
).rstrip("/")

BINANCE_DEMO_WS_URL: str = os.getenv(
    "BINANCE_DEMO_WS_URL", "wss://demo-trading.binance.com/ws"
)


def _to_dec(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


class BinanceDemoClient:
    """
    Binance Demo Trading REST client using direct HTTP requests.

    Binance Demo Trading uses the same API structure as mainnet
    but at a different base URL. CCXT's sandbox mode does NOT work
    with Demo Trading - we use requests directly.

    Demo Trading characteristics:
      ✓ Real-time mainnet order book mirrored exactly
      ✓ Free virtual balance (created on account registration)
      ✓ Supports mainnet spot pairs including ARB/USDC
      ✗ Fills are simulated (no real counterparty)
      ✗ Cannot withdraw
    """

    def __init__(self, api_key: str, secret: str) -> None:
        self._api_key = api_key
        self._secret = secret
        self._base_url = BINANCE_DEMO_REST_URL
        self._session = requests.Session()
        self._session.headers.update({
            "X-MBX-APIKEY": self._api_key,
            "Content-Type": "application/json"
        })

        # Verify connectivity
        try:
            resp = self._public_get("/api/v3/time")
            server_time = resp.get("serverTime", 0)
            log.info(
                "BinanceDemoClient connected | url=%s serverTime=%d",
                self._base_url, server_time
            )
        except Exception as exc:
            log.warning(
                "BinanceDemoClient connectivity check failed: %s "
                "(ensure BINANCE_DEMO_API_KEY / BINANCE_DEMO_SECRET are set "
                "and the account exists at demo-trading.binance.com)",
                exc
            )

    @classmethod
    def from_env(cls) -> "BinanceDemoClient":
        """Create from BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET env vars."""
        api_key = os.getenv("BINANCE_DEMO_API_KEY", "")
        secret = os.getenv("BINANCE_DEMO_SECRET", "")
        if not api_key or not secret:
            raise ValueError(
                "BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET must be set. "
                "Register at https://demo-trading.binance.com to get keys."
            )
        return cls(api_key, secret)

    # ------------------------------------------------------------------
    # Public API (compatible with BinanceClient)
    # ------------------------------------------------------------------

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """Return a normalized L2 order book snapshot."""
        # Convert pair format: "ARB/USDC" -> "ARBUSDC"
        raw_symbol = symbol.replace("/", "").upper()
        raw = self._public_get("/api/v3/depth", {"symbol": raw_symbol, "limit": limit})

        bids = sorted(
            [(Decimal(str(p)), Decimal(str(q))) for p, q in raw.get("bids", [])],
            key=lambda x: x[0], reverse=True
        )
        asks = sorted(
            [(Decimal(str(p)), Decimal(str(q))) for p, q in raw.get("asks", [])],
            key=lambda x: x[0]
        )
        best_bid = bids[0] if bids else (Decimal("0"), Decimal("0"))
        best_ask = asks[0] if asks else (Decimal("0"), Decimal("0"))
        mid = (
            (best_bid[0] + best_ask[0]) / Decimal("2")
            if best_bid[0] and best_ask[0]
            else Decimal("0")
        )
        spread_bps = (
            (best_ask[0] - best_bid[0]) / mid * Decimal("10000")
            if mid > 0 else Decimal("0")
        )
        return {
            "symbol": symbol,
            "timestamp": raw.get("lastUpdateId", int(time.time() * 1000)),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps
        }

    def fetch_balance(self) -> dict[str, dict]:
        """Return non-zero balances keyed by asset symbol."""
        raw = self._signed_get("/api/v3/account")
        result: dict[str, dict] = {}
        for item in raw.get("balances", []):
            free = _to_dec(item.get("free", "0"))
            locked = _to_dec(item.get("locked", "0"))
            total = free + locked
            if total == Decimal("0"):
                continue
            result[item["asset"]] = {
                "free": free,
                "locked": locked,
                "total": total
            }
        return result

    def create_limit_ioc_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> dict:
        """Submit a LIMIT IOC order and return the normalized result."""
        raw_symbol = symbol.replace("/", "").upper()
        params = {
            "symbol": raw_symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "IOC",
            "quantity": f"{amount:.8f}",
            "price": f"{price:.8f}"
        }
        raw = self._signed_post("/api/v3/order", params)
        return self._normalize_order(raw)

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Submit a market order and return the normalized result."""
        raw_symbol = symbol.replace("/", "").upper()
        params = {
            "symbol": raw_symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{amount:.8f}"
        }
        raw = self._signed_post("/api/v3/order", params)
        return self._normalize_order(raw)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        raw_symbol = symbol.replace("/", "").upper()
        params = {"symbol": raw_symbol, "orderId": order_id}
        raw = self._signed_delete("/api/v3/order", params)
        return self._normalize_order(raw)

    def fetch_order_status(self, order_id: str, symbol: str) -> dict:
        raw_symbol = symbol.replace("/", "").upper()
        params = {"symbol": raw_symbol, "orderId": order_id}
        raw = self._signed_get("/api/v3/order", params)
        return self._normalize_order(raw)

    def get_trading_fees(self, symbol: str) -> dict:
        """Return maker/taker fees. Demo Trading uses standard 0.1% fees."""
        return {
            "maker": Decimal("0.001"),
            "taker": Decimal("0.001")
        }

    # ------------------------------------------------------------------
    # WebSocket endpoint helper (for LiveOrderBook)
    # ------------------------------------------------------------------

    @staticmethod
    def ws_base_url() -> str:
        return BINANCE_DEMO_WS_URL

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _public_get(self, path: str, params: dict | None = None) -> dict:
        """Unauthenticated GET request."""
        url = self._base_url + path
        resp = self._session.get(url, params=params or {}, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _signed_get(self, path: str, params: dict | None = None) -> dict:
        """Authenticated GET request with HMAC signature."""
        params = dict(params or {})
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            self._secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        url = self._base_url + path
        resp = self._session.get(url, params=params, timeout=10)
        self._raise_for_binance_error(resp)
        return resp.json()

    def _signed_post(self, path: str, params: dict) -> dict:
        """Authenticated POST request with HMAC signature."""
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            self._secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        url = self._base_url + path
        resp = self._session.post(url, params=params, timeout=10)
        self._raise_for_binance_error(resp)
        return resp.json()

    def _signed_delete(self, path: str, params: dict) -> dict:
        """Authenticated DELETE request with HMAC signature."""
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urlencode(params)
        signature = hmac.new(
            self._secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        url = self._base_url + path
        resp = self._session.delete(url, params=params, timeout=10)
        self._raise_for_binance_error(resp)
        return resp.json()

    def _raise_for_binance_error(self, resp: requests.Response) -> None:
        """Raise a descriptive exception for Binance API errors."""
        if resp.status_code == 200:
            return
        try:
            body = resp.json()
            code = body.get("code", resp.status_code)
            msg = body.get("msg", resp.text)
            raise Exception(f"Binance Demo API error {code}: {msg}")
        except (ValueError, KeyError):
            resp.raise_for_status()

    def _normalize_order(self, raw: dict) -> dict:
        """Convert a Binance order response into a consistent format."""
        filled = _to_dec(raw.get("executedQty", 0))
        requested = _to_dec(raw.get("origQty", 0))

        # Compute average fill price
        cumulative_quote = _to_dec(raw.get("cummulativeQuoteQty", 0))
        if filled > 0 and cumulative_quote > 0:
            avg_price = cumulative_quote / filled
        else:
            avg_price = _to_dec(raw.get("price", 0))

        status_raw = (raw.get("status") or "").upper()
        if status_raw == "FILLED":
            status = "filled"
        elif status_raw == "PARTIALLY_FILLED":
            status = "partially_filled"
        elif status_raw in ("CANCELED", "CANCELLED", "EXPIRED", "REJECTED"):
            status = "expired"
        else:
            status = status_raw.lower() or "unknown"

        return {
            "id": str(raw.get("orderId", "")),
            "symbol": raw.get("symbol", ""),
            "side": (raw.get("side") or "").lower(),
            "type": (raw.get("type") or "").lower(),
            "time_in_force": raw.get("timeInForce", ""),
            "amount_requested": requested,
            "amount_filled": filled,
            "avg_fill_price": avg_price,
            "fee": Decimal("0"),   # Demo Trading does not return fee detail
            "fee_asset": "",
            "status": status,
            "timestamp": raw.get("transactTime") or raw.get("time") or int(time.time() * 1000)
        }
