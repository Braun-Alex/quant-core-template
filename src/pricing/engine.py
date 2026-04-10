"""
Top-level pricing facade for the arbitrage engine.

PricingEngine is the single public entry point that orchestrates:
  - live pool state from chain (via ChainClient)
  - multi-hop routing (PathFinder)
  - fork-based verification (TradeSimulator)
  - mempool surveillance (MempoolWatcher)

Callers need only import PricingEngine and call .get_quote().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.fork_simulator import TradeSimulator
from src.pricing.mempool import MempoolWatcher, PendingSwap
from src.pricing.router import PathFinder, SwapPath

log = logging.getLogger(__name__)

# Address used as *from* in read-only eth_call simulations
_SIMULATION_SENDER = Address("0x0000000000000000000000000000000000000003")


class PricingError(Exception):
    """Raised whenever the engine cannot produce a valid quote."""


@dataclass
class PriceQuote:
    """
    A verified price quote for a swap.

    Attributes
    ----------
    path          : the recommended execution path
    qty_in        : raw input amount
    expected_net  : net output after subtracting estimated gas cost
    verified_out  : raw output as confirmed by fork simulation
    gas_used      : gas estimate from simulation
    created_at    : wall-clock timestamp (time.time())
    """

    path: SwapPath
    qty_in: int
    expected_net: int
    verified_out: int
    gas_used: int
    created_at: float

    @property
    def trustworthy(self) -> bool:
        """
        True when the fork-verified output is within 0.1 % of the AMM estimate.
        A larger gap means the cached reserves are stale and the quote should
        be refreshed before acting on it.
        """
        if self.expected_net == 0:
            return self.verified_out == 0
        discrepancy = abs(self.expected_net - self.verified_out)
        return Decimal(discrepancy) / Decimal(self.expected_net) < Decimal("0.001")


class PricingEngine:
    """
    Orchestrates the full pricing pipeline.

    Parameters
    ----------
    chain_client  : connected ChainClient (from src.chain)
    simulator     : TradeSimulator backed by a local Anvil fork
    ws_url        : WebSocket endpoint for mempool monitoring
    """

    def __init__(
        self,
        chain_client,
        simulator: TradeSimulator,
        ws_url: str
    ) -> None:
        self._client = chain_client
        self._simulator = simulator
        self._watcher = MempoolWatcher(ws_url, self._handle_pending)
        self._pools: dict[Address, PoolState] = {}
        self._finder: PathFinder | None = None
        self.seen_swaps: list[PendingSwap] = []

    # ------------------------------------------------------------------
    # Pool registry
    # ------------------------------------------------------------------

    def register_pools(self, addresses: list[Address]) -> None:
        """
        Fetch current on-chain state for every address and rebuild the
        path graph. Call this once at startup and after large reserve shifts.
        """
        for addr in addresses:
            state = PoolState.load(addr, self._client)
            self._pools[addr] = state
            log.debug("Registered pool %s  %s/%s",
                      addr, state.left.symbol, state.right.symbol)
        self._finder = PathFinder(list(self._pools.values()))
        log.info("Pool registry ready (%d pools).", len(self._pools))

    def refresh(self, addr: Address) -> None:
        """
        Re-fetch reserves for one pool without rebuilding the full graph.
        The PathFinder's pool references are updated in place.
        """
        if addr not in self._pools:
            raise KeyError(f"Pool {addr} not registered. Call register_pools first.")
        fresh = PoolState.load(addr, self._client)
        stored = self._pools[addr]
        stored.qty_left = fresh.qty_left
        stored.qty_right = fresh.qty_right
        stored.fee_bps = fresh.fee_bps
        log.debug("Refreshed %s: L=%d R=%d", addr, stored.qty_left, stored.qty_right)

    # ------------------------------------------------------------------
    # Quote
    # ------------------------------------------------------------------

    def get_quote(
        self,
        selling: Token,
        buying: Token,
        qty_in: int,
        gas_gwei: int
    ) -> PriceQuote:
        """
        Find the gas-optimal swap path and verify it against the fork.

        Raises PricingError on failure (no pools, no path, simulation error).
        """
        if self._finder is None:
            raise PricingError("No pools registered. Call register_pools() first.")

        try:
            path, net = self._finder.find_optimal(selling, buying, qty_in, gas_gwei)
        except ValueError as exc:
            raise PricingError(str(exc)) from exc

        receipt = self._simulator.verify_path(path, qty_in, _SIMULATION_SENDER)
        if not receipt.ok:
            raise PricingError(f"Fork verification failed: {receipt.error}")

        return PriceQuote(
            path=path,
            qty_in=qty_in,
            expected_net=net,
            verified_out=receipt.qty_out,
            gas_used=receipt.gas_used,
            created_at=time.time()
        )

    # ------------------------------------------------------------------
    # Mempool integration
    # ------------------------------------------------------------------

    def _handle_pending(self, swap: PendingSwap) -> None:
        """
        Invoked by MempoolWatcher for each decoded pending swap.
        Queues swaps that overlap with registered pools.
        """
        overlap = self._overlapping_pools(swap)
        if not overlap:
            return
        log.info("Pending %s.%s touches %d registered pool(s).",
                 swap.protocol, swap.fn_name, len(overlap))
        self.seen_swaps.append(swap)

    def _overlapping_pools(self, swap: PendingSwap) -> list[PoolState]:
        """Return pools whose token set intersects the swap's token pair."""
        involved: set[Address] = set()
        if swap.token_sold:
            involved.add(swap.token_sold)
        if swap.token_bought:
            involved.add(swap.token_bought)
        if not involved:
            return []
        return [
            ps for ps in self._pools.values()
            if {ps.left.address, ps.right.address} & involved
        ]
