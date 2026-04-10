"""
Multi-hop swap routing across a set of liquidity pools.

Architecture
------------
SwapPath   - an immutable ordered sequence of pools linking token A to token B.
             Exposes simulation helpers and a gas estimate.
PathFinder - builds a token adjacency graph from a pool set and enumerates
             all simple (cycle-free) paths up to a hop limit via DFS.
             find_optimal() returns the path that maximizes *net* output
             (gross AMM output minus the cost of the required gas).

Gas model
---------
The gas estimate is deliberately conservative:
    total_gas = BASE + PER_HOP × num_hops
    BASE    = 150 000  (router overhead + one transfer)
    PER_HOP = 100 000  (one pair swap + two transfers)
This matches observed Uniswap V2 router costs closely enough for
route comparison purposes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.core.types import Token

_BASE_GAS = 150_000
_HOP_GAS = 100_000
_WEI_PER_GWEI = 10 ** 9


class SwapPath:
    """
    An ordered chain of pools that converts *tokens[0]* into *tokens[-1]*.

    Invariant: len(tokens) == len(pools) + 1
    Each consecutive (pool, token_sold) pair defines one AMM swap.
    """

    __slots__ = ("pools", "tokens")

    def __init__(self, pools: list[Any], tokens: list[Token]) -> None:
        if not pools:
            raise ValueError("SwapPath requires at least one pool.")
        if len(tokens) != len(pools) + 1:
            raise ValueError(
                f"Expected len(pools)+1 tokens, "
                f"got {len(pools)} pools and {len(tokens)} tokens."
            )
        self.pools = pools
        self.tokens = tokens

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hops(self) -> int:
        return len(self.pools)

    @property
    def token_in(self) -> Token:
        return self.tokens[0]

    @property
    def token_out(self) -> Token:
        return self.tokens[-1]

    def gas_estimate(self) -> int:
        return _BASE_GAS + _HOP_GAS * self.hops

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(self, qty_in: int) -> int:
        """Simulate the full path and return the final raw output."""
        current = qty_in
        for pool, sold in zip(self.pools, self.tokens):
            current = pool.out_for_in(current, sold)
        return current

    def simulate_steps(self, qty_in: int) -> list[int]:
        """Return amounts at every step including the input: [in, step1, ..., out]."""
        trace = [qty_in]
        current = qty_in
        for pool, sold in zip(self.pools, self.tokens):
            current = pool.out_for_in(current, sold)
            trace.append(current)
        return trace

    def net_output(self, qty_in: int, gas_gwei: int) -> int:
        """Gross output minus the estimated gas cost (in the same wei units)."""
        gross = self.simulate(qty_in)
        gas_cost = gas_gwei * _WEI_PER_GWEI * self.gas_estimate()
        return max(0, gross - gas_cost)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        chain = " → ".join(t.symbol for t in self.tokens)
        return f"SwapPath[{self.hops}h]({chain})"


class PathFinder:
    """
    Discovers all simple swap paths between two tokens and selects the best one.

    Usage
    -----
        pf = PathFinder(pool_list)
        path, net = pf.find_optimal(USDC, WETH, amount_in=1_000e6, gas_gwei=20)
    """

    def __init__(self, pools: list[Any]) -> None:
        self.pools = pools
        self._graph = self._build_adjacency()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_adjacency(self) -> dict[Token, list[tuple[Any, Token]]]:
        """
        adjacency[token] = [(pool, neighbor_token), ...]
        Both directions are registered for each pool.
        """
        graph: dict[Token, list[tuple[Any, Token]]] = defaultdict(list)
        for pool in self.pools:
            graph[pool.left].append((pool, pool.right))
            graph[pool.right].append((pool, pool.left))
        return graph

    # ------------------------------------------------------------------
    # Path enumeration (depth-limited DFS)
    # ------------------------------------------------------------------

    def enumerate(
        self,
        src: Token,
        dst: Token,
        max_hops: int = 3,
    ) -> list[SwapPath]:
        """Return every simple path from *src* to *dst* up to *max_hops*."""
        results: list[SwapPath] = []
        self._dfs(
            current = src,
            destination = dst,
            max_hops = max_hops,
            seen_tokens = {src},
            seen_pools = set(),
            path_pools = [],
            path_tokens = [src],
            results = results
        )
        return results

    def _dfs(
        self,
        current: Token,
        destination: Token,
        max_hops: int,
        seen_tokens: set[Token],
        seen_pools: set[int],
        path_pools: list[Any],
        path_tokens: list[Token],
        results: list[SwapPath]
    ) -> None:
        if current == destination:
            results.append(SwapPath(list(path_pools), list(path_tokens)))
            return
        if len(path_pools) >= max_hops:
            return
        for pool, neighbour in self._graph.get(current, []):
            pid = id(pool)
            if pid in seen_pools:
                continue
            if neighbour in seen_tokens and neighbour != destination:
                continue

            path_pools.append(pool)
            path_tokens.append(neighbour)
            seen_pools.add(pid)
            seen_tokens.add(neighbour)

            self._dfs(current=neighbour, destination=destination,
                      max_hops=max_hops, seen_tokens=seen_tokens,
                      seen_pools=seen_pools, path_pools=path_pools,
                      path_tokens=path_tokens, results=results)

            path_pools.pop()
            path_tokens.pop()
            seen_pools.discard(pid)
            seen_tokens.discard(neighbour)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def find_optimal(
        self,
        src: Token,
        dst: Token,
        qty_in: int,
        gas_gwei: int,
        max_hops: int = 3
    ) -> tuple[SwapPath, int]:
        """
        Return (best_path, net_output) where best_path maximizes net output
        after deducting the gas cost of executing that path.

        Raises ValueError when no path exists.
        """
        candidates = self.enumerate(src, dst, max_hops)
        if not candidates:
            raise ValueError(
                f"No route found between {src.symbol} and {dst.symbol} "
                f"within {max_hops} hop(s)."
            )
        scored = [(p, p.net_output(qty_in, gas_gwei)) for p in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0]

    def rank_all(
        self,
        src: Token,
        dst: Token,
        qty_in: int,
        gas_gwei: int
    ) -> list[dict]:
        """
        Evaluate every discovered path and return a ranked list of dicts:
            path, gross_out, gas_units, gas_cost_wei, net_out
        Sorted best-first by net_out.
        """
        rows = []
        for path in self.enumerate(src, dst):
            gross = path.simulate(qty_in)
            gas_u = path.gas_estimate()
            gas_cost = gas_gwei * _WEI_PER_GWEI * gas_u
            net = max(0, gross - gas_cost)
            rows.append(dict(path=path, gross_out=gross,
                             gas_units=gas_u, gas_cost_wei=gas_cost,
                             net_out=net))
        rows.sort(key=lambda r: r["net_out"], reverse=True)
        return rows
