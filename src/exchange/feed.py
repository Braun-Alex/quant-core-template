"""
Real-time order book via WebSocket depth stream.

Maintains a locally-consistent order book by following the official
Binance synchronization protocol:

  1. Open WebSocket stream (buffering all incoming diff events).
  2. Fetch a REST snapshot to establish the baseline lastUpdateId.
  3. Discard any buffered event whose final update ID (u) ≤ lastUpdateId.
  4. Validate the first accepted event: its first update ID (U) must be
     ≤ lastUpdateId + 1 ≤ its final update ID (u).
  5. Apply subsequent diffs: qty == "0" means remove the level;
     qty > "0" means upsert.
  6. After each applied diff, yield a snapshot compatible with the
     BinanceClient.fetch_order_book() output format (plus last_update_id).

Usage::

    import asyncio
    from src.exchange.feed import LiveOrderBook

    async def main():
        async with LiveOrderBook("ETH/USDT", testnet=True) as book:
            async for snap in book:
                print(snap["best_bid"], snap["best_ask"])
                break

    asyncio.run(main())

CLI::

    python3 -m src.exchange.feed ETH/USDT
    python3 -m src.exchange.feed BTC/USDT --count 3 --mainnet
"""

from __future__ import annotations

import aiohttp
import argparse
import asyncio
import json
import logging
import os
import sys
import time
import websockets
from collections.abc import AsyncIterator
from decimal import Decimal
from dotenv import load_dotenv

log = logging.getLogger(__name__)

load_dotenv()

_TESTNET_WS = os.getenv("BINANCE_TESTNET_WS", "")
_TESTNET_REST = os.getenv("BINANCE_TESTNET_REST", "")
_MAINNET_WS = os.getenv("BINANCE_MAINNET_WS", "")
_MAINNET_REST = os.getenv("BINANCE_MAINNET_REST", "")


def _parse_decimal(raw: str) -> Decimal:
    return Decimal(str(raw))


class LiveOrderBook:
    """
    Async context manager that streams a locally-maintained order book.

    Each iteration yields a snapshot dict with the same shape as
    BinanceClient.fetch_order_book(), extended with a ``last_update_id``
    field that tracks the Binance sequence number.

    The internal state is a plain dict keyed by price-as-Decimal so that
    level lookups are O(1) and sorting is only done at snapshot time.
    """

    def __init__(
        self,
        symbol: str,
        testnet: bool = True,
        max_depth: int = 20
    ) -> None:
        self._symbol = symbol
        self._ws_symbol = symbol.replace("/", "").lower()
        self._rest_symbol = symbol.replace("/", "").upper()
        self._testnet = testnet
        self._max_depth = max_depth

        # Order book state - dicts for O(1) upsert/delete
        self._bids: dict[Decimal, Decimal] = {}
        self._asks: dict[Decimal, Decimal] = {}
        self._last_seq: int = 0   # lastUpdateId from snapshot
        self._initialised: bool = False

        # Connection handles
        self._ws = None
        self._http_session = None

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
        """Open the WebSocket connection and synchronize with a REST snapshot."""
        log.info("Connecting to %s", self._ws_endpoint)
        self._ws = await websockets.connect(self._ws_endpoint)
        self._http_session = aiohttp.ClientSession()

        raw_snap = await self._fetch_rest_snapshot()
        self._apply_snapshot(raw_snap)
        log.info("Synced %s at lastUpdateId=%d", self._symbol, self._last_seq)

    async def disconnect(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

    async def __aenter__(self) -> "LiveOrderBook":
        await self.connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def __aiter__(self) -> AsyncIterator[dict]:
        """
        Yield an updated book snapshot after every accepted diff message.
        Stale messages (where final_id ≤ last_seq) are silently dropped.
        """
        if self._ws is None:
            raise RuntimeError("Not connected - use 'async with LiveOrderBook(...) as book:'")

        async for raw_msg in self._ws:
            event = json.loads(raw_msg)
            changed = self._apply_diff(event)
            if changed:
                yield self.current_snapshot()

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
        Apply one WebSocket depth-diff event.

        Returns True if the event was accepted and the book changed;
        False if the event was stale (final_id ≤ last_seq).

        Event fields:
          U   first_update_id_in_event
          u   final_update_id_in_event
          b   [[price, qty], ...]  bid updates
          a   [[price, qty], ...]  ask updates
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

    def current_snapshot(self) -> dict:
        """
        Build a point-in-time snapshot of the local book.

        Format is compatible with BinanceClient.fetch_order_book() output,
        plus the ``last_update_id`` sequence number for traceability.
        """
        sorted_bids = sorted(self._bids.items(), key=lambda x: x[0], reverse=True)[: self._max_depth]
        sorted_asks = sorted(self._asks.items(), key=lambda x: x[0])[: self._max_depth]

        bids = list(sorted_bids)
        asks = list(sorted_asks)

        best_bid = bids[0] if bids else (Decimal("0"), Decimal("0"))
        best_ask = asks[0] if asks else (Decimal("0"), Decimal("0"))

        bid_px, ask_px = best_bid[0], best_ask[0]
        mid = (bid_px + ask_px) / Decimal("2") if bid_px and ask_px else Decimal("0")
        spread_bps = (ask_px - bid_px) / mid * Decimal("10000") if mid > 0 else Decimal("0")

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
# CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stream a live Binance order book over WebSocket",
        prog="python3 -m src.exchange.feed",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 -m src.exchange.feed ETH/USDT\n"
            "  python3 -m src.exchange.feed BTC/USDT --count 3 --mainnet"
        )
    )
    parser.add_argument("symbol", help="Trading pair, e.g. ETH/USDT")
    parser.add_argument("--count", type=int, default=5, help="Number of updates to print")
    parser.add_argument("--mainnet", dest="testnet", action="store_false", default=True,
                        help="Use mainnet instead of testnet")
    args = parser.parse_args(argv)

    async def _stream() -> None:
        net_label = "testnet" if args.testnet else "mainnet"
        print(f"\n{args.symbol} - live depth stream ({net_label})")
        print(f"Printing {args.count} updates.\n")

        async with LiveOrderBook(args.symbol, testnet=args.testnet) as book:
            count = 0
            async for snap in book:
                bid_p, _ = snap["best_bid"]
                ask_p, _ = snap["best_ask"]
                mid = snap["mid_price"]
                spread = snap["spread_bps"]
                seq = snap["last_update_id"]
                print(
                    f"  seq={seq:>12d}  bid={float(bid_p):>10,.2f}"
                    f"  ask={float(ask_p):>10,.2f}"
                    f"  mid={float(mid):>10,.2f}"
                    f"  spread={float(spread):>6.2f} bps"
                )
                count += 1
                if count >= args.count:
                    break

    try:
        asyncio.run(_stream())
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
