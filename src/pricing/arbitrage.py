"""
Arbitrage opportunity detection.

Two strategies are implemented:

1. Triangular (circular) arbitrage
   Starting from a chosen token, the detector walks the pool graph looking
   for a cycle that returns more tokens than it started with. Example:
       WETH → DAI → USDC → WETH
   If the final WETH amount exceeds the initial amount, a profitable
   opportunity exists.

2. Cross-pool arbitrage
   When the same token pair is listed on more than one pool at different
   prices, you can buy on the cheaper pool and sell on the more expensive
   one. This scanner tries both directions for every pair of compatible
   pools.

Both strategies return Opportunity records sorted by gross profit (highest
first). Gas cost is included so callers can filter by net profitability.

Design notes
------------
- All math stays in raw integer units - no floats.
- Pools are treated as duck-typed objects: anything with .out_for_in(),
  .left, and .right attributes works (V2 PoolState, V3Pool, etc.).
- The DFS for circular arb is cycle-free: each pool is visited at most once,
  and non-origin tokens already on the current path are skipped.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from src.core.types import Token
from src.pricing.router import SwapPath, _BASE_GAS, _HOP_GAS

_WEI_PER_GWEI = 10 ** 9


# ---------------------------------------------------------------------------
# Result record
# ---------------------------------------------------------------------------

@dataclass
class Opportunity:
    """
    A detected arbitrage opportunity.

    Attributes
    ----------
    path        : execution path (may be circular)
    qty_in      : raw input amount in *token* units
    gross_profit: raw profit before gas (qty_out − qty_in)
    gas_cost    : estimated gas cost in the same raw units as *token*
    token       : the token in which profit is denominated
    strategy    : "triangular" or "cross_pool"
    """

    path: SwapPath
    qty_in: int
    gross_profit: int
    gas_cost: int
    token: Token
    strategy: str

    @property
    def net_profit(self) -> int:
        """Profit after deducting gas."""
        return self.gross_profit - self.gas_cost

    @property
    def is_profitable(self) -> bool:
        """True when gross profit is positive (before gas)."""
        return self.gross_profit > 0

    @property
    def is_net_profitable(self) -> bool:
        """True when net profit (after gas) is positive."""
        return self.net_profit > 0

    def __repr__(self) -> str:
        return (
            f"Opportunity({self.strategy}, gross={self.gross_profit}, "
            f"net={self.net_profit}, path={self.path!r})"
        )


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ArbitrageDetector:
    """
    Scans a set of liquidity pools for profitable arbitrage opportunities.

    Parameters
    ----------
    pools : list of pool objects (PoolState, V3Pool, or any duck-typed pool)
    """

    def __init__(self, pools: list[Any]) -> None:
        self.pools = pools
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------

    def _build_graph(self) -> dict[Token, list[tuple[Any, Token]]]:
        """adjacency[token] = [(pool, neighbour_token), ...]"""
        g: dict[Token, list[tuple[Any, Token]]] = defaultdict(list)
        for p in self.pools:
            g[p.left].append((p, p.right))
            g[p.right].append((p, p.left))
        return g

    # ------------------------------------------------------------------
    # Triangular (circular) arbitrage
    # ------------------------------------------------------------------

    def find_triangular(
        self,
        token: Token,
        qty_in: int,
        gas_gwei: int,
        max_hops: int = 3
    ) -> list[Opportunity]:
        """
        Find every cycle starting and ending at *token* with gross_profit > 0.

        The search is a depth-limited DFS. A valid cycle is recorded when
        the DFS returns to *token* having traversed at least one pool.

        Returns opportunities sorted by gross_profit descending.
        """
        results: list[Opportunity] = []
        self._dfs_cycle(
            origin=token, current=token,
            qty=qty_in, running=qty_in,
            gas_gwei=gas_gwei * _WEI_PER_GWEI,
            max_hops=max_hops,
            used_pools=set(),
            pools_so_far=[], tokens_so_far=[token],
            results=results
        )
        results.sort(key=lambda o: o.gross_profit, reverse=True)
        return results

    def _dfs_cycle(
        self,
        origin: Token,
        current: Token,
        qty: int,
        running: int,
        gas_gwei: int,
        max_hops: int,
        used_pools: set[int],
        pools_so_far: list[Any],
        tokens_so_far: list[Token],
        results: list[Opportunity]
    ) -> None:
        for pool, neighbour in self._graph.get(current, []):
            pid = id(pool)
            if pid in used_pools:
                continue

            try:
                received = pool.out_for_in(running, current)
            except Exception:
                continue   # Skip pools that cannot execute this trade

            # Cycle closed - record if profitable
            if neighbour == origin and pools_so_far:
                profit = received - qty
                if profit > 0:
                    full_path = pools_so_far + [pool]
                    gas_cost = gas_gwei * (_BASE_GAS + _HOP_GAS * len(full_path))
                    results.append(Opportunity(
                        path=SwapPath(full_path, tokens_so_far + [neighbour]),
                        qty_in=qty,
                        gross_profit=profit,
                        gas_cost=gas_cost,
                        token=origin,
                        strategy="triangular"
                    ))
                continue   # Do not recurse further after closing a cycle

            # Avoid revisiting non-origin tokens
            if neighbour in tokens_so_far:
                continue

            if len(pools_so_far) + 1 >= max_hops:
                continue

            used_pools.add(pid)
            pools_so_far.append(pool)
            tokens_so_far.append(neighbour)

            self._dfs_cycle(
                origin=origin, current=neighbour,
                qty=qty, running=received,
                gas_gwei=gas_gwei, max_hops=max_hops,
                used_pools=used_pools,
                pools_so_far=pools_so_far, tokens_so_far=tokens_so_far,
                results=results
            )

            pools_so_far.pop()
            tokens_so_far.pop()
            used_pools.discard(pid)

    # ------------------------------------------------------------------
    # Cross-pool arbitrage
    # ------------------------------------------------------------------

    def find_cross_pool(
        self,
        token_a: Token,
        token_b: Token,
        qty_in: int,
        gas_gwei: int
    ) -> list[Opportunity]:
        """
        Scan every pair of pools that both trade (token_a, token_b) and
        look for a buy-on-cheap / sell-on-dear price discrepancy.

        Both directions are tested for each pool pair.
        Returns opportunities with gross_profit > 0, sorted descending.
        """
        # Find all pools that contain both tokens
        candidates = [
            p for p in self.pools
            if ({p.left, p.right} == {token_a, token_b})
        ]
        # Deduplicate by object identity
        seen: set[int] = set()
        unique = [p for p in candidates if not (id(p) in seen or seen.add(id(p)))]

        gas_wei = gas_gwei * _WEI_PER_GWEI
        gas_cost = gas_wei * (_BASE_GAS + _HOP_GAS * 2)

        results: list[Opportunity] = []
        for i, pool_x in enumerate(unique):
            for pool_y in unique[i + 1:]:
                for buy_pool, sell_pool in [(pool_x, pool_y), (pool_y, pool_x)]:
                    opp = self._cross_pair(
                        buy_pool, sell_pool, token_a, token_b, qty_in, gas_cost
                    )
                    if opp is not None:
                        results.append(opp)

        results.sort(key=lambda o: o.gross_profit, reverse=True)
        return results

    def _cross_pair(
        self,
        buy_pool: Any,
        sell_pool: Any,
        token_in: Token,
        token_out: Token,
        qty_in: int,
        gas_cost: int
    ) -> Opportunity | None:
        """
        Buy token_out on buy_pool, then sell it back on sell_pool.
        Return an Opportunity if gross_profit > 0, otherwise None.
        """
        try:
            mid = buy_pool.out_for_in(qty_in, token_in)
            final = sell_pool.out_for_in(mid, token_out)
        except Exception:
            return None

        profit = final - qty_in
        if profit <= 0:
            return None

        path = SwapPath(
            [buy_pool, sell_pool],
            [token_in, token_out, token_in]
        )
        return Opportunity(
            path=path,
            qty_in=qty_in,
            gross_profit=profit,
            gas_cost=gas_cost,
            token=token_in,
            strategy="cross_pool"
        )

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def best_triangular(
        self, token: Token, qty_in: int, gas_gwei: int, max_hops: int = 3
    ) -> Opportunity | None:
        """Return the single most profitable triangular opportunity, or None."""
        opps = self.find_triangular(token, qty_in, gas_gwei, max_hops)
        return opps[0] if opps else None

    def best_cross_pool(
        self, token_a: Token, token_b: Token, qty_in: int, gas_gwei: int
    ) -> Opportunity | None:
        """Return the single most profitable cross-pool opportunity, or None."""
        opps = self.find_cross_pool(token_a, token_b, qty_in, gas_gwei)
        return opps[0] if opps else None
