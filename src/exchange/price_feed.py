"""
PriceFeedManager: unified real-time price hub.

Merges:
  - CEX prices from Binance WebSocket depth stream (LiveOrderBook)
  - DEX prices from Uniswap V2 reserve polling (DEXPriceFeed)

into a single in-memory ``PriceState`` that the signal generator
reads without any blocking I/O.

Architecture
------------

    ┌──────────────────────────────────────────────────────────┐
    │                   PriceFeedManager                       │
    │                                                          │
    │  LiveOrderBook ──ws──► on_cex_update()                   │
    │  (Binance)                     │                         │
    │                                ▼                         │
    │                        PriceState (shared)               │
    │                                ▲                         │
    │  DEXPriceFeed ──poll──► on_dex_update()                  │
    │  (Uniswap V2)                                            │
    │                                                          │
    │  on_price_update callback ◄──── fires when BOTH sides    │
    │                                 have fresh data          │
    └──────────────────────────────────────────────────────────┘

The callback ``on_price_update(pair, state)`` is fired after each
CEX or DEX update, provided both sides have a recent snapshot.
The signal generator is wired as this callback → it runs purely
reactively, not on a polling timer.

Example:

    feed_mgr = PriceFeedManager(
        cex_book=LiveOrderBook("ETH/USDC", testnet=True),
        dex_feed=dex_feed,
        pairs=["ETH/USDC"],
        on_price_update=bot._on_price_update,
        stale_threshold_seconds=5.0
    )
    asyncio.create_task(feed_mgr.run())
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Coroutine, Optional

from src.exchange.feed import LiveOrderBook, OrderBookSnapshot, run_order_book_feed
from src.pricing.dex_feed import DEXPriceFeed, DEXPriceSnapshot

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared price state
# ---------------------------------------------------------------------------

@dataclass
class PriceState:
    """
    Latest merged price from both CEX and DEX.

    Both sides must be fresh (within stale_threshold_seconds) before
    the state is considered valid for trading decisions.
    """
    pair: str

    # CEX (Binance)
    cex_bid: Decimal = Decimal("0")
    cex_ask: Decimal = Decimal("0")
    cex_mid: Decimal = Decimal("0")
    cex_spread_bps: Decimal = Decimal("0")
    cex_timestamp: float = 0.0
    cex_last_update_id: int = 0

    # DEX (Uniswap V2)
    dex_price: Decimal = Decimal("0")
    dex_reserve_base: int = 0
    dex_reserve_quote: int = 0
    dex_fee_bps: int = 30
    dex_timestamp: float = 0.0
    dex_poll_latency_ms: float = 0.0

    # Derived
    spread_bps: Decimal = Decimal("0")   # CEX/DEX spread
    is_valid: bool = False   # Both sides fresh

    def update_cex(self, snap: OrderBookSnapshot) -> None:
        self.cex_bid = snap["best_bid"][0]
        self.cex_ask = snap["best_ask"][0]
        self.cex_mid = snap["mid_price"]
        self.cex_spread_bps = snap["spread_bps"]
        self.cex_timestamp = snap["timestamp"] / 1000.0
        self.cex_last_update_id = snap.get("last_update_id", 0)

    def update_dex(self, snap: DEXPriceSnapshot) -> None:
        self.dex_price = snap.price
        self.dex_reserve_base = snap.reserve_base
        self.dex_reserve_quote = snap.reserve_quote
        self.dex_fee_bps = snap.fee_bps
        self.dex_timestamp = snap.timestamp
        self.dex_poll_latency_ms = snap.block_time_ms

    def recompute(self, stale_threshold: float) -> None:
        """Recompute derived fields and validity."""
        now = time.time()
        cex_fresh = (self.cex_bid > 0 and
                     now - self.cex_timestamp < stale_threshold)
        dex_fresh = (self.dex_price > 0 and
                     now - self.dex_timestamp < stale_threshold)
        self.is_valid = cex_fresh and dex_fresh

        if self.cex_mid > 0 and self.dex_price > 0:
            # Buy CEX sell DEX spread: dex_sell - cex_ask
            # Use cex_mid as a proxy for dex comparison baseline
            self.spread_bps = (
                abs(self.dex_price - self.cex_mid)
                / self.cex_mid
                * Decimal("10000")
            )


# ---------------------------------------------------------------------------
# PriceFeedManager
# ---------------------------------------------------------------------------

PriceUpdateCallback = Callable[
    [str, PriceState], Coroutine   # Async (pair, state) -> None
]


class PriceFeedManager:
    """
    Orchestrates CEX WebSocket and DEX polling into a unified real-time
    price hub for all tracked pairs.

    Fires ``on_price_update(pair, state)`` after every incoming update,
    provided the state is valid (both sides have fresh data).
    """

    def __init__(
        self,
        cex_book: LiveOrderBook,
        dex_feed: DEXPriceFeed,
        pairs: list[str],
        on_price_update: Optional[PriceUpdateCallback] = None,
        stale_threshold_seconds: float = 5.0
    ) -> None:
        self._cex_book = cex_book
        self._dex_feed = dex_feed
        self._pairs = pairs
        self._on_update = on_price_update
        self._stale = stale_threshold_seconds

        # One PriceState per pair
        self._states: dict[str, PriceState] = {
            pair: PriceState(pair=pair) for pair in pairs
        }

        self._running = False
        self._stop_event = asyncio.Event()

        # Wire DEX callback
        self._dex_feed.on_update = self._on_dex_update

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start both feeds as concurrent background tasks.
        This coroutine runs until stop() is called.
        """
        self._running = True
        self._stop_event.clear()

        log.info(
            "PriceFeedManager starting | pairs=%s cex=%s dex_pools=%d",
            self._pairs,
            "testnet" if self._cex_book._testnet else "mainnet",
            len(self._dex_feed._pools)
        )

        # CEX WebSocket feed task
        cex_task = asyncio.create_task(
            run_order_book_feed(
                self._cex_book,
                self._on_cex_update,
                stop_event=self._stop_event
            ),
            name="cex_ws_feed"
        )

        # DEX polling task
        dex_task = asyncio.create_task(
            self._dex_feed.run(),
            name="dex_poll_feed"
        )

        try:
            await self._stop_event.wait()
        finally:
            cex_task.cancel()
            await self._dex_feed.stop()
            for task in (cex_task, dex_task):
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        self._running = False
        log.info("PriceFeedManager stopped")

    async def stop(self) -> None:
        self._stop_event.set()

    def get_state(self, pair: str) -> Optional[PriceState]:
        """Return current merged price state for *pair* (non-blocking)."""
        return self._states.get(pair)

    def get_all_states(self) -> dict[str, PriceState]:
        return dict(self._states)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Callbacks from feeds
    # ------------------------------------------------------------------

    async def _on_cex_update(self, snap: OrderBookSnapshot) -> None:
        """Called by LiveOrderBook on every depth diff."""
        # Match snapshot symbol to our pairs
        symbol_raw = snap.get("symbol", "").upper()   # E.g. "ETH/USDT" or "ETHUSDT"
        matched_pair = self._match_pair(symbol_raw)
        if matched_pair is None:
            return

        state = self._states[matched_pair]
        state.update_cex(snap)
        state.recompute(self._stale)

        log.info(
            "CEX update | %s bid=%.4f ask=%.4f spread=%.2fbps seq=%d",
            matched_pair,
            float(state.cex_bid), float(state.cex_ask),
            float(state.cex_spread_bps),
            state.cex_last_update_id
        )

        if state.is_valid and self._on_update is not None:
            try:
                await self._on_update(matched_pair, state)
            except Exception as exc:
                log.error("Callback on_price_update error: %s", exc)

    async def _on_dex_update(self, pair: str, snap: DEXPriceSnapshot) -> None:
        """Called by DEXPriceFeed on every price change."""
        if pair not in self._states:
            return

        state = self._states[pair]
        state.update_dex(snap)
        state.recompute(self._stale)

        log.info(
            "DEX update | %s price=%.4f latency=%.0fms",
            pair, float(snap.price), snap.block_time_ms
        )

        if state.is_valid and self._on_update is not None:
            try:
                await self._on_update(pair, state)
            except Exception as exc:
                log.error("Callback on_price_update error: %s", exc)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _match_pair(self, raw_symbol: str) -> Optional[str]:
        """Map a raw symbol like 'ETHUSDT' or 'ETH/USDT' to a tracked pair."""
        for pair in self._pairs:
            canonical = pair.upper()
            compressed = pair.replace("/", "").upper()
            if raw_symbol in (canonical, compressed):
                return pair
        return None
