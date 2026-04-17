"""
Binance testnet interface via ccxt.

Provides a rate-limited, type-safe wrapper around ccxt's Binance adapter.
All monetary values are returned as Decimal to avoid float imprecision.
A sliding-window weight tracker prevents hitting exchange rate limits.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal, InvalidOperation
from typing import Any

log = logging.getLogger(__name__)

# Binance spot API allows 1200 weight per minute; we stop at 90%
_RATE_LIMIT_MAX = 1200
_RATE_LIMIT_THRESHOLD = int(_RATE_LIMIT_MAX * 0.9)

# Approximate request weights per endpoint type
_REQUEST_WEIGHTS: dict[str, int] = {
    "fetch_order_book": 5,
    "fetch_balance": 10,
    "create_order": 1,
    "cancel_order": 1,
    "fetch_order": 2,
    "fetch_trading_fee": 20,
    "fetch_time": 1,
    "fetch_status": 1
}


def _to_dec(value: Any) -> Decimal:
    """Safely convert any value to Decimal, returning zero on failure."""
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return Decimal("0")


class BinanceClient:
    """
    Testnet Binance adapter built on top of ccxt.

    Handles:
    - Automatic rate-limit budgeting with sleep-before-ban
    - Normalisation of all monetary values to Decimal
    - Structured logging for every outbound API call
    - Connection health check on construction
    """

    def __init__(self, config: dict) -> None:
        try:
            import ccxt
        except ImportError as exc:
            raise ImportError("ccxt is required: pip install ccxt") from exc

        self._exchange = ccxt.binance(config)

        # Sliding window state
        self._weight_consumed: int = 0
        self._window_resets_at: float = time.monotonic() + 60.0

        # Verify connectivity immediately
        self._call("fetch_time")
        log.info("BinanceClient ready - sandbox=%s", config.get("sandbox", False))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        """
        Return a normalized L2 order book snapshot.

        Keys: symbol, timestamp, bids, asks, best_bid, best_ask,
              mid_price, spread_bps - all monetary values as Decimal.
        """
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

        mid = (best_bid[0] + best_ask[0]) / Decimal("2") if best_bid[0] and best_ask[0] else Decimal("0")
        spread_bps = (best_ask[0] - best_bid[0]) / mid * Decimal("10000") if mid > 0 else Decimal("0")

        return {
            "symbol": symbol,
            "timestamp": raw.get("timestamp") or int(time.time() * 1000),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps
        }

    def fetch_balance(self) -> dict[str, dict]:
        """
        Return non-zero balances keyed by asset symbol.
        Each entry: {free: Decimal, locked: Decimal, total: Decimal}.
        """
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

        log.debug("fetch_balance: %d assets with non-zero balance", len(result))
        return result

    def create_limit_ioc_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float
    ) -> dict:
        """Submit a LIMIT IOC order and return the normalized result."""
        raw = self._call(
            "create_order", symbol, "limit", side, amount, price,
            {"timeInForce": "IOC"}
        )
        return self._normalize_order(raw)

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        """Submit a market order and return the normalized result."""
        raw = self._call("create_order", symbol, "market", side, amount)
        return self._normalize_order(raw)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order."""
        raw = self._call("cancel_order", order_id, symbol)
        return self._normalize_order(raw)

    def fetch_order_status(self, order_id: str, symbol: str) -> dict:
        """Retrieve the current status of any order."""
        raw = self._call("fetch_order", order_id, symbol)
        return self._normalize_order(raw)

    def get_trading_fees(self, symbol: str) -> dict:
        """Return maker/taker fees as Decimals."""
        raw = self._call("fetch_trading_fee", symbol)
        return {
            "maker": _to_dec(raw.get("maker", "0.001")),
            "taker": _to_dec(raw.get("taker", "0.001"))
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a ccxt method with rate-limit tracking and error logging."""
        import ccxt

        self._consume_weight(method)
        log.debug("→ %s(%s)", method, args[:2] if args else "")
        t0 = time.monotonic()

        try:
            result = getattr(self._exchange, method)(*args, **kwargs)
            log.debug("← %s OK (%.3fs)", method, time.monotonic() - t0)
            return result

        except ccxt.RateLimitExceeded:
            log.warning("Rate limit exceeded on %s — sleeping 60s", method)
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
        """Track request weight and pause if the safety budget is reached."""
        now = time.monotonic()
        if now >= self._window_resets_at:
            self._weight_consumed = 0
            self._window_resets_at = now + 60.0

        weight = _REQUEST_WEIGHTS.get(method, 1)
        if self._weight_consumed + weight >= _RATE_LIMIT_THRESHOLD:
            pause = self._window_resets_at - now
            if pause > 0:
                log.warning(
                    "Weight budget at %d/%d — pausing %.1fs",
                    self._weight_consumed, _RATE_LIMIT_THRESHOLD, pause
                )
                time.sleep(pause)
            self._weight_consumed = 0
            self._window_resets_at = time.monotonic() + 60.0

        self._weight_consumed += weight

    def _normalize_order(self, raw: dict) -> dict:
        """Convert a ccxt order dict into a consistent, Decimal-typed format."""
        filled = _to_dec(raw.get("filled", 0))
        requested = _to_dec(raw.get("amount", 0))
        avg_price = _to_dec(raw.get("average") or raw.get("price") or 0)

        fee_info = raw.get("fee") or {}
        fee_amount = _to_dec(fee_info.get("cost", 0))
        fee_currency = fee_info.get("currency", "")

        status_raw = (raw.get("status") or "").lower()
        if status_raw == "closed" and filled >= requested:
            status = "filled"
        elif status_raw == "closed" and filled < requested:
            status = "partially_filled"
        elif status_raw in ("canceled", "cancelled", "expired"):
            status = "expired"
        else:
            status = status_raw or "unknown"

        tif = raw.get("timeInForce") or raw.get("info", {}).get("timeInForce", "")

        return {
            "id": str(raw.get("id", "")),
            "symbol": raw.get("symbol", ""),
            "side": raw.get("side", ""),
            "type": raw.get("type", ""),
            "time_in_force": tif,
            "amount_requested": requested,
            "amount_filled": filled,
            "avg_fill_price": avg_price,
            "fee": fee_amount,
            "fee_asset": fee_currency,
            "status": status,
            "timestamp": raw.get("timestamp") or int(time.time() * 1000)
        }
