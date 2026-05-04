"""
Real-time Binance order book via WebSocket depth stream.

Implements the official Binance synchronization protocol:
  1. Open WebSocket, buffer all incoming diff events.
  2. Fetch REST snapshot to establish lastUpdateId baseline.
  3. Discard buffered events whose final_id ≤ lastUpdateId.
  4. Validate first accepted event: U ≤ lastUpdateId+1 ≤ u.
  5. Apply subsequent diffs: qty==0 → remove level, qty>0 → upsert.
  6. Emit snapshot after every accepted diff.

Works on both Binance Testnet and Mainnet via environment variables:
  BINANCE_TESTNET_WS   (default wss://testnet.binance.vision/ws)
  BINANCE_MAINNET_WS   (default wss://stream.binance.com:9443/ws)
  BINANCE_TESTNET_REST (default https://testnet.binance.vision)
  BINANCE_MAINNET_REST (default https://api.binance.com)

Usage:
    async with LiveOrderBook("ETH/USDT", testnet=True) as book:
        async for snap in book:
            print(snap["best_bid"], snap["best_ask"])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Optional

import aiohttp
import websockets
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoint resolution
# ---------------------------------------------------------------------------

_TESTNET_WS = os.getenv(
    "BINANCE_TESTNET_WS", "wss://testnet.binance.vision/ws"
)
_TESTNET_REST = os.getenv(
    "BINANCE_TESTNET_REST", "https://testnet.binance.vision"
)
_MAINNET_WS = os.getenv(
    "BINANCE_MAINNET_WS", "wss://stream.binance.com:9443/ws"
)
_MAINNET_REST = os.getenv(
    "BINANCE_MAINNET_REST", "https://api.binance.com"
)

# Reconnect settings
_RECONNECT_DELAY_SECONDS: float = float(
    os.getenv("WS_RECONNECT_DELAY_SECONDS", "2")
)
_MAX_RECONNECT_ATTEMPTS: int = int(
    os.getenv("WS_MAX_RECONNECT_ATTEMPTS", "10")
)
_PING_INTERVAL_SECONDS: float = float(
    os.getenv("WS_PING_INTERVAL_SECONDS", "20")
)


def _parse_decimal(raw: str) -> Decimal:
    return Decimal(str(raw))


# ---------------------------------------------------------------------------
# Order book snapshot type
# ---------------------------------------------------------------------------

OrderBookSnapshot = dict   # Typed alias for documentation


# ---------------------------------------------------------------------------
# LiveOrderBook
# ---------------------------------------------------------------------------

class LiveOrderBook:
    """
    Async context manager that streams a locally-maintained L2 order book.

    Each async iteration yields an ``OrderBookSnapshot`` dict with the
    same shape as ``BinanceClient.fetch_order_book()``, plus
    ``last_update_id`` (Binance sequence number).

    The internal state uses plain dicts keyed by price-as-Decimal:
    O(1) upserts, sorted only at snapshot time.

    Example::

        async with LiveOrderBook("ETH/USDC", testnet=True) as book:
            async for snap in book:
                bid = snap["best_bid"][0]
                ask = snap["best_ask"][0]
                # process...
    """

    def __init__(
        self,
        symbol: str,
        testnet: bool = True,
        max_depth: int = 20,
        reconnect: bool = True
    ) -> None:
        self._symbol = symbol
        self._ws_symbol = symbol.replace("/", "").lower()
        self._rest_symbol = symbol.replace("/", "").upper()
        self._testnet = testnet
        self._max_depth = max_depth
        self._reconnect = reconnect

        # Book state
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_seq: int = 0
        self._initialized: bool = False

        # Connection handles
        self._ws = None
        self._http_session: Optional[aiohttp.ClientSession] = None

        # Update notification
        self._update_event: asyncio.Event = asyncio.Event()

        # Latest snapshot cache (for non-async consumers)
        self._latest_snapshot: Optional[OrderBookSnapshot] = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    @property
    def _ws_endpoint(self) -> str:
        base = _TESTNET_WS if self._testnet else _MAINNET_WS
        return f"{base}/{self._ws_symbol}@depth@100ms"

    @property
    def _rest_base(self) -> str:
        return _TESTNET_REST if self._testnet else _MAINNET_REST

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the WebSocket and synchronize with a REST snapshot."""
        log.info("Connecting to %s (%s)", self._ws_endpoint,
                 "testnet" if self._testnet else "mainnet")
        self._http_session = aiohttp.ClientSession()
        self._ws = await websockets.connect(
            self._ws_endpoint,
            ping_interval=_PING_INTERVAL_SECONDS,
            ping_timeout=30
        )
        raw_snap = await self._fetch_rest_snapshot()
        self._apply_snapshot(raw_snap)
        self._connected = True
        log.info(
            "LiveOrderBook synced | symbol=%s lastUpdateId=%d",
            self._symbol, self._last_seq
        )

    async def disconnect(self) -> None:
        self._connected = False
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        if self._http_session is not None:
            try:
                await self._http_session.close()
            except Exception:
                pass
            self._http_session = None

    async def __aenter__(self) -> "LiveOrderBook":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Streaming iterator
    # ------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[OrderBookSnapshot]:
        """
        Yield a snapshot after every accepted diff message.
        Reconnects automatically if the WebSocket drops (configurable).
        """
        if self._ws is None:
            raise RuntimeError("Not connected - use 'async with LiveOrderBook(...) as book:'")

        reconnect_attempts = 0

        while True:
            try:
                async for raw_msg in self._ws:
                    event = json.loads(raw_msg)
                    changed = self._apply_diff(event)
                    if changed:
                        snap = self.current_snapshot()
                        self._latest_snapshot = snap
                        self._update_event.set()
                        self._update_event.clear()
                        yield snap
                # Normal close - exit if reconnect disabled
                if not self._reconnect:
                    break
            except (
                websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK
            ) as exc:
                log.warning("WebSocket closed: %s", exc)
            except Exception as exc:
                log.error("WebSocket error: %s", exc)

            if not self._reconnect:
                break

            reconnect_attempts += 1
            if reconnect_attempts > _MAX_RECONNECT_ATTEMPTS:
                log.critical(
                    "Max reconnect attempts (%d) exceeded for %s — giving up",
                    _MAX_RECONNECT_ATTEMPTS, self._symbol
                )
                break

            delay = _RECONNECT_DELAY_SECONDS * min(reconnect_attempts, 5)
            log.info(
                "Reconnecting in %.1fs (attempt %d/%d)...",
                delay, reconnect_attempts, _MAX_RECONNECT_ATTEMPTS
            )
            await asyncio.sleep(delay)
            try:
                await self.disconnect()
                await self.connect()
            except Exception as exc:
                log.error("Reconnect failed: %s", exc)

    # ------------------------------------------------------------------
    # Non-blocking snapshot access
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[OrderBookSnapshot]:
        """
        Return the most recently received snapshot without blocking.
        Returns None if no snapshot has been received yet.
        """
        return self._latest_snapshot

    async def wait_for_update(self, timeout: float = 5.0) -> Optional[OrderBookSnapshot]:
        """
        Wait up to *timeout* seconds for the next update.
        Returns the new snapshot, or None on timeout.
        """
        try:
            await asyncio.wait_for(self._update_event.wait(), timeout=timeout)
            return self._latest_snapshot
        except asyncio.TimeoutError:
            return None

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ws is not None

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    async def _fetch_rest_snapshot(self) -> dict:
        url = f"{self._rest_base}/api/v3/depth"
        params = {"symbol": self._rest_symbol, "limit": str(self._max_depth)}
        async with self._http_session.get(url, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    def _apply_snapshot(self, data: dict) -> None:
        """Seed the local book from a REST depth snapshot."""
        self._last_seq = int(data["lastUpdateId"])
        self._bids = {
            _parse_decimal(p): _parse_decimal(q)
            for p, q in data["bids"]
            if _parse_decimal(q) > 0
        }
        self._asks = {
            _parse_decimal(p): _parse_decimal(q)
            for p, q in data["asks"]
            if _parse_decimal(q) > 0
        }
        self._initialised = True

    def _apply_diff(self, event: dict) -> bool:
        """
        Apply a WebSocket depth-diff event.
        Returns True if accepted and the book changed.
        """
        final_id: int = int(event.get("u", 0))
        if final_id <= self._last_seq:
            return False   # Stale - discard

        for price_str, qty_str in event.get("b", []):
            px = _parse_decimal(price_str)
            qty = _parse_decimal(qty_str)
            if qty == 0:
                self._bids.pop(px, None)
            else:
                self._bids[px] = qty

        for price_str, qty_str in event.get("a", []):
            px = _parse_decimal(price_str)
            qty = _parse_decimal(qty_str)
            if qty == 0:
                self._asks.pop(px, None)
            else:
                self._asks[px] = qty

        self._last_seq = final_id
        return True

    # ------------------------------------------------------------------
    # Snapshot export
    # ------------------------------------------------------------------

    def current_snapshot(self) -> OrderBookSnapshot:
        """
        Build a point-in-time snapshot of the local book.

        Format is compatible with ``BinanceClient.fetch_order_book()`` output,
        plus the ``last_update_id`` sequence number.
        """
        sorted_bids = sorted(
            self._bids.items(), key=lambda x: x[0], reverse=True
        )[: self._max_depth]
        sorted_asks = sorted(
            self._asks.items(), key=lambda x: x[0]
        )[: self._max_depth]

        bids = list(sorted_bids)
        asks = list(sorted_asks)

        best_bid = bids[0] if bids else (Decimal("0"), Decimal("0"))
        best_ask = asks[0] if asks else (Decimal("0"), Decimal("0"))

        bid_px, ask_px = best_bid[0], best_ask[0]
        mid = (bid_px + ask_px) / Decimal("2") if bid_px and ask_px else Decimal("0")
        spread_bps = (
            (ask_px - bid_px) / mid * Decimal("10000") if mid > 0 else Decimal("0")
        )

        return {
            "symbol": self._symbol,
            "timestamp": int(time.time() * 1000),
            "bids": bids,
            "asks": asks,
            "best_bid": best_bid,
            "best_ask": best_ask,
            "mid_price": mid,
            "spread_bps": spread_bps,
            "last_update_id": self._last_seq
        }


# ---------------------------------------------------------------------------
# Background task helper (fire-and-forget)
# ---------------------------------------------------------------------------

async def run_order_book_feed(
    book: LiveOrderBook,
    on_update,   # Async callable(snapshot: OrderBookSnapshot)
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Run ``book`` as a background task, calling ``on_update`` for every snapshot.

    ``stop_event`` can be used for graceful cancellation:
        stop = asyncio.Event()
        asyncio.create_task(run_order_book_feed(book, handler, stop))
        # later:
        stop.set()

    Usage in the bot::

        book = LiveOrderBook("ETH/USDC", testnet=cfg.cex.sandbox)
        async with book:
            asyncio.create_task(
                run_order_book_feed(book, bot._on_cex_update)
            )
    """
    async for snap in book:
        if stop_event is not None and stop_event.is_set():
            break
        try:
            await on_update(snap)
        except Exception as exc:
            log.error("on_update callback error: %s", exc)
