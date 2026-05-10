"""
DEX price feed: event-driven Uniswap V2 reserve polling.

On-chain data has no push model - a node cannot push reserve changes.
However, Arbitrum has ~0.25 s block times, so polling every 1–2 s
(configurable via ``DEX_POLL_INTERVAL_SECONDS``) gives latency
comparable to a WebSocket feed for low-frequency arbitrage.

Architecture
------------
DEXPriceFeed polls ``getReserves()`` on every registered pool via
eth_call (free, no gas cost).  When reserves change it:
  1. Recomputes the AMM price.
  2. Updates the in-memory snapshot.
  3. Fires ``on_update(pair, snapshot)`` asynchronously.

Subscribers (e.g. the signal generator) consume prices from the
shared snapshot dict - no blocking REST calls per tick.

Integration
-----------
    feed = DEXPriceFeed(
        chain_client=chain_client,
        pools={"ETH/USDC": pool_state},
        poll_interval=1.0
    )
    feed.on_update = my_callback
    asyncio.create_task(feed.run())   # Later: await feed.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Coroutine, Optional

log = logging.getLogger(__name__)

# Configurable via environment
_DEX_POLL_INTERVAL: float = float(
    os.getenv("DEX_POLL_INTERVAL_SECONDS", "1.0")
)
_DEX_MIN_CHANGE_BPS: float = float(
    os.getenv("DEX_MIN_CHANGE_BPS", "0.1")   # Only notify on ≥0.1 bps price change
)


@dataclass
class DEXPriceSnapshot:
    """Current AMM price snapshot for one pair."""
    pair: str
    pool_address: str
    reserve_base: int   # Raw units
    reserve_quote: int   # Raw units
    price: Decimal   # Quote tokens per base token (human units)
    mid_price: Decimal   # Same as price for V2 AMM
    fee_bps: int
    block_time_ms: float   # Time taken to fetch (ms)
    timestamp: float   # Unix seconds


UpdateCallback = Callable[[str, DEXPriceSnapshot], Coroutine]


class DEXPriceFeed:
    """
    Polls Uniswap V2 pool reserves on a tight async loop and fires
    ``on_update(pair, snapshot)`` whenever price changes by ≥ threshold.

    Parameters
    ----------
    chain_client   : ChainClient - connected to Arbitrum or Ethereum
    pools          : dict[pair_str, PoolState] - preloaded pool objects
    poll_interval  : seconds between reserve polls (default 1.0 s)
    min_change_bps : minimum price change to trigger on_update (default 0.1 bps)
    on_update      : async callback - called with (pair, snapshot) on price change
    """

    def __init__(
        self,
        chain_client,
        pools: dict,
        poll_interval: float = _DEX_POLL_INTERVAL,
        min_change_bps: float = _DEX_MIN_CHANGE_BPS,
        on_update: Optional[UpdateCallback] = None
    ) -> None:
        self._client = chain_client
        self._pools = pools   # {pair: PoolState}
        self._interval = poll_interval
        self._min_change_bps = Decimal(str(min_change_bps))
        self.on_update = on_update

        # In-memory price cache: {pair: DEXPriceSnapshot}
        self._snapshots: dict[str, DEXPriceSnapshot] = {}
        self._prev_prices: dict[str, Decimal] = {}

        self._running = False
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """
        Start the polling loop. Run as a background asyncio task:
            asyncio.create_task(dex_feed.run())
        """
        self._running = True
        self._stop_event.clear()
        log.info(
            "DEXPriceFeed starting | pools=%d interval=%.1fs",
            len(self._pools), self._interval
        )

        # Prime the snapshots immediately (do not wait for first interval)
        await self._poll_all()

        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._stop_event.wait()),
                    timeout=self._interval
                )
                break   # Stop_event was set
            except asyncio.TimeoutError:
                pass   # Normal - interval elapsed

            await self._poll_all()

        self._running = False
        log.info("DEXPriceFeed stopped")

    async def stop(self) -> None:
        """Gracefully stop the polling loop."""
        self._stop_event.set()

    def get_snapshot(self, pair: str) -> Optional[DEXPriceSnapshot]:
        """
        Get the latest cached price snapshot for *pair*.
        Returns None if no snapshot has been fetched yet.
        Thread-safe for single-threaded asyncio usage.
        """
        return self._snapshots.get(pair)

    def get_all_snapshots(self) -> dict[str, DEXPriceSnapshot]:
        """Return all current price snapshots."""
        return dict(self._snapshots)

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal polling
    # ------------------------------------------------------------------

    async def _poll_all(self) -> None:
        """Poll all registered pools concurrently."""
        tasks = [
            self._poll_pool(pair, pool)
            for pair, pool in self._pools.items()
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for pair, result in zip(self._pools.keys(), results):
            if isinstance(result, Exception):
                log.info("DEX poll error for %s: %s", pair, result)

    async def _poll_pool(self, pair: str, pool) -> None:
        """Fetch reserves for one pool and fire callback if price changed."""
        t0 = time.monotonic()

        loop = asyncio.get_event_loop()
        try:
            r0, r1 = await loop.run_in_executor(
                None, lambda: self._fetch_reserves(pool)
            )
        except Exception as exc:
            log.info("Reserve fetch failed for %s: %s", pair, exc)
            return

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Compute price from reserves
        try:
            price = self._reserves_to_price(pool, r0, r1)
        except Exception as exc:
            log.info("Price computation failed for %s: %s", pair, exc)
            return

        snap = DEXPriceSnapshot(
            pair=pair,
            pool_address=pool.contract.checksum,
            reserve_base=r0,
            reserve_quote=r1,
            price=price,
            mid_price=price,
            fee_bps=pool.fee_bps,
            block_time_ms=elapsed_ms,
            timestamp=time.time()
        )
        self._snapshots[pair] = snap

        # Check if price moved enough to notify
        prev = self._prev_prices.get(pair)
        if prev is None or prev == 0:
            changed = True
        else:
            change_bps = abs(price - prev) / prev * Decimal("10000")
            changed = change_bps >= self._min_change_bps

        if changed:
            self._prev_prices[pair] = price
            log.info(
                "DEX price update | %s price=%.4f elapsed=%.0fms",
                pair, float(price), elapsed_ms
            )
            if self.on_update is not None:
                try:
                    await self.on_update(pair, snap)
                except Exception as exc:
                    log.error("DEX on_update callback error: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _fetch_reserves(self, pool) -> tuple[int, int]:
        """Read on-chain reserves via eth_call (synchronous, run in executor)."""
        from eth_abi import decode as abi_decode

        _SEL_RESERVES = bytes.fromhex("0902f1ac")   # Method getReserves()
        w3 = self._client._get_w3()
        raw = w3.eth.call({
            "to": pool.contract.checksum,
            "data": "0x" + _SEL_RESERVES.hex()
        })
        r0, r1, _ = abi_decode(["uint112", "uint112", "uint32"], raw)
        return r0, r1

    def _reserves_to_price(self, pool, r0: int, r1: int) -> Decimal:
        """
        Compute the marginal price from reserves.

        Returns quote_human / base_human so the price is always expressed
        as "how many quote tokens per 1 base token" in human units.
        """
        if r0 == 0 or r1 == 0:
            return Decimal("0")

        # Determine which reserve is base (WETH) and which is quote (USDC)
        # by checking the token ordering in the PoolState
        left_is_base = True   # Base token by convention: pool.left

        if left_is_base:
            base_raw = r0
            quote_raw = r1
            base_dec = pool.left.decimals
            quote_dec = pool.right.decimals
        else:
            base_raw = r1
            quote_raw = r0
            base_dec = pool.right.decimals
            quote_dec = pool.left.decimals

        base_human = Decimal(base_raw) / Decimal(10 ** base_dec)
        quote_human = Decimal(quote_raw) / Decimal(10 ** quote_dec)

        if base_human == 0:
            return Decimal("0")

        return quote_human / base_human
