"""
Binance Demo Trading adapter.

Binance Demo Trading (https://demo-trading.binance.com) is fundamentally
different from Binance Testnet:

  Testnet        - synthetic order book, fake prices, zero market depth
  Demo Trading   - mirrors real Binance mainnet order book in real time,
                   but fills are simulated against that real book
                   → realistic spreads, real price discovery, virtual funds

This makes Demo Trading the only safe environment where you can validate
CEX-DEX arbitrage logic with *realistic* CEX behavior before real risking.

API compatibility
-----------------
Demo Trading uses the same REST and WebSocket API as Binance mainnet.
The only differences:
  - Base URL:  https://demo-trading.binance.com
  - WS URL:    wss://demo-trading.binance.com/ws
  - Separate API keys (generated on demo-trading.binance.com)

Environment variables
---------------------
  BINANCE_DEMO_API_KEY    API key from demo-trading.binance.com
  BINANCE_DEMO_SECRET     Secret from demo-trading.binance.com

  BINANCE_DEMO_REST_URL   (default https://demo-trading.binance.com)
  BINANCE_DEMO_WS_URL     (default wss://demo-trading.binance.com/ws)

Usage
-----
    from src.exchange.demo_client import BinanceDemoClient
    client = BinanceDemoClient.from_env()
    book = client.fetch_order_book("ARB/USDC")
"""

from __future__ import annotations

import logging
import os
import time
from decimal import Decimal, InvalidOperation
from typing import Any

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BINANCE_DEMO_REST_URL: str = os.getenv(
    "BINANCE_DEMO_REST_URL", "https://demo-trading.binance.com"
)
BINANCE_DEMO_WS_URL: str = os.getenv(
    "BINANCE_DEMO_WS_URL", "wss://demo-trading.binance.com/ws"
)

_REQUEST_WEIGHTS: dict[str, int] = {
    "fetch_order_book": 5, "fetch_balance": 10, "create_order": 1,
    "cancel_order": 1, "fetch_order": 2, "fetch_trading_fee": 20,
    "fetch_time": 1
}
_RATE_LIMIT_MAX = 1200
_RATE_LIMIT_THRESHOLD = int(_RATE_LIMIT_MAX * 0.9)


def _to_dec(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


class BinanceDemoClient:
    """
    Binance Demo Trading REST client.

    Drop-in replacement for BinanceClient - identical public API,
    routes all requests to demo-trading.binance.com instead of Binance mainnet.

    Demo Trading characteristics:
      ✓ Real-time mainnet order book mirrored exactly
      ✓ Free virtual balance ($1,000,000 USDT on account creation)
      ✓ Supports all mainnet spot pairs including ARB/USDC
      ✗ Fills are simulated (no real counterparty)
      ✗ Cannot withdraw
    """

    def __init__(self, api_key: str, secret: str) -> None:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("ccxt is required: pip install ccxt") from exc

        import ccxt

        # ccxt binance supports custom URLs via options
        self._exchange = ccxt.binance({
            "apiKey": api_key,
            "secret": secret,
            "options": {
                "defaultType": "spot",
                # Override both REST and WS base URLs to Demo Trading
                "baseUrl": BINANCE_DEMO_REST_URL
            },
            "enableRateLimit": True,
            # Manually set URLs since ccxt does not have a demo-trading preset
            "urls": {
                "api": {
                    "public": f"{BINANCE_DEMO_REST_URL}/api",
                    "private": f"{BINANCE_DEMO_REST_URL}/api",
                    "v3": f"{BINANCE_DEMO_REST_URL}/api/v3"
                },
                "ws": {
                    "public": BINANCE_DEMO_WS_URL,
                    "private": BINANCE_DEMO_WS_URL
                },
            },
        })

        self._weight_consumed: int = 0
        self._window_resets_at: float = time.monotonic() + 60.0

        # Verify connectivity
        try:
            self._call("fetch_time")
            log.info(
                "BinanceDemoClient connected | url=%s", BINANCE_DEMO_REST_URL
            )
        except Exception as exc:
            log.warning(
                "BinanceDemoClient connectivity check failed: %s "
                "(ensure BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET are set)",
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
    # Public API (identical to BinanceClient)
    # ------------------------------------------------------------------

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        raw = self._call("fetch_order_book", symbol, limit)
        bids = sorted(
            [(Decimal(str(p)), Decimal(str(q))) for p, q in raw["bids"]],
            key=lambda x: x[0], reverse=True
        )
        asks = sorted(
            [(Decimal(str(p)), Decimal(str(q))) for p, q in raw["asks"]],
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
            "timestamp": raw.get("timestamp") or int(time.time() * 1000),
            "bids": bids, "asks": asks,
            "best_bid": best_bid, "best_ask": best_ask,
            "mid_price": mid, "spread_bps": spread_bps
        }

    def fetch_balance(self) -> dict[str, dict]:
        raw = self._call("fetch_balance")
        result: dict[str, dict] = {}
        for asset, info in raw.items():
            if not isinstance(info, dict):
                continue
            free = _to_dec(info.get("free", 0))
            locked = _to_dec(info.get("used", 0))
            total = _to_dec(info.get("total", 0))
            if total == Decimal("0"):
                continue
            result[asset] = {"free": free, "locked": locked, "total": total}
        return result

    def create_limit_ioc_order(
        self, symbol: str, side: str, amount: float, price: float
    ) -> dict:
        raw = self._call(
            "create_order", symbol, "limit", side, amount, price,
            {"timeInForce": "IOC"}
        )
        return self._normalize_order(raw)

    def create_market_order(
        self, symbol: str, side: str, amount: float
    ) -> dict:
        raw = self._call("create_order", symbol, "market", side, amount)
        return self._normalize_order(raw)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self._normalize_order(self._call("cancel_order", order_id, symbol))

    def fetch_order_status(self, order_id: str, symbol: str) -> dict:
        return self._normalize_order(self._call("fetch_order", order_id, symbol))

    def get_trading_fees(self, symbol: str) -> dict:
        raw = self._call("fetch_trading_fee", symbol)
        return {
            "maker": _to_dec(raw.get("maker", "0.001")),
            "taker": _to_dec(raw.get("taker", "0.001"))
        }

    # ------------------------------------------------------------------
    # WebSocket endpoint helper (for LiveOrderBook)
    # ------------------------------------------------------------------

    @staticmethod
    def ws_base_url() -> str:
        return BINANCE_DEMO_WS_URL

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        import ccxt
        self._consume_weight(method)
        try:
            return getattr(self._exchange, method)(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            log.warning("Rate limit exceeded on %s - sleeping 60s", method)
            time.sleep(60)
            raise
        except ccxt.AuthenticationError as exc:
            log.error("Authentication failed on %s: %s", method, exc)
            raise
        except ccxt.NetworkError as exc:
            log.error("Network error on %s: %s", method, exc)
            raise
        except ccxt.BaseError as exc:
            log.error("Exchange error on %s: %s", method, exc)
            raise

    def _consume_weight(self, method: str) -> None:
        now = time.monotonic()
        if now >= self._window_resets_at:
            self._weight_consumed = 0
            self._window_resets_at = now + 60.0
        weight = _REQUEST_WEIGHTS.get(method, 1)
        if self._weight_consumed + weight >= _RATE_LIMIT_THRESHOLD:
            pause = self._window_resets_at - now
            if pause > 0:
                time.sleep(pause)
            self._weight_consumed = 0
            self._window_resets_at = time.monotonic() + 60.0
        self._weight_consumed += weight

    def _normalize_order(self, raw: dict) -> dict:
        filled = _to_dec(raw.get("filled", 0))
        requested = _to_dec(raw.get("amount", 0))
        avg_price = _to_dec(raw.get("average") or raw.get("price") or 0)
        status_raw = (raw.get("status") or "").lower()
        if status_raw == "closed" and filled >= requested:
            status = "filled"
        elif status_raw == "closed" and filled < requested:
            status = "partially_filled"
        elif status_raw in ("canceled", "cancelled", "expired"):
            status = "expired"
        else:
            status = status_raw or "unknown"
        return {
            "id": str(raw.get("id", "")),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", ""),
            "type": raw.get("type", ""),
            "time_in_force": raw.get("timeInForce", ""),
            "amount_requested": requested,
            "amount_filled": filled,
            "avg_fill_price": avg_price,
            "fee": _to_dec((raw.get("fee") or {}).get("cost", 0)),
            "fee_asset": (raw.get("fee") or {}).get("currency", ""),
            "status": status,
            "timestamp": raw.get("timestamp") or int(time.time() * 1000)
        }
