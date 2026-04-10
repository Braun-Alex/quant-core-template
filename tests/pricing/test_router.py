"""
Tests for SwapPath and PathFinder.

Strategy: build small pool networks and assert that enumeration counts,
output values, and gas-adjusted selections are correct.
"""

from __future__ import annotations

import pytest

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.router import SwapPath, PathFinder, _BASE_GAS, _HOP_GAS

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)
DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)
LINK = Token(Address("0x514910771AF9Ca656af840dff83E8264EcF986CA"), "LINK", 18)


def _ps(n: str, left: Token, right: Token, ql: int, qr: int, fee: int = 30) -> PoolState:
    return PoolState(Address("0x" + n.zfill(40)), left, right, ql, qr, fee_bps=fee)


# Convenient pool factories
def liquid() -> PoolState:
    """Deep DAI/WETH pool — 30 bps fee."""
    return _ps("3", DAI, WETH, 2_000_000 * 10**18, 1_000 * 10**18)


def stable() -> PoolState:
    """Deep USDC/DAI stable pool — 5 bps fee."""
    return _ps("2", USDC, DAI, 10_000_000 * 10**6, 10_000_000 * 10**18, fee=5)


def direct() -> PoolState:
    """Shallow USDC/WETH pool — lower liquidity, more slippage."""
    return _ps("1", USDC, WETH, 100_000 * 10**6, 50 * 10**18)


# ===========================================================================
# SwapPath
# ===========================================================================

class TestSwapPath:
    def test_single_hop_properties(self):
        sp = SwapPath([direct()], [USDC, WETH])
        assert sp.hops == 1
        assert sp.token_in == USDC
        assert sp.token_out == WETH

    def test_two_hop_properties(self):
        sp = SwapPath([stable(), liquid()], [USDC, DAI, WETH])
        assert sp.hops == 2

    def test_token_count_mismatch_raises(self):
        with pytest.raises(ValueError):
            SwapPath([direct()], [USDC, DAI, WETH])

    def test_empty_pools_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SwapPath([], [USDC])

    def test_gas_single_hop(self):
        sp = SwapPath([direct()], [USDC, WETH])
        assert sp.gas_estimate() == _BASE_GAS + _HOP_GAS

    def test_gas_two_hops(self):
        sp = SwapPath([stable(), liquid()], [USDC, DAI, WETH])
        assert sp.gas_estimate() == _BASE_GAS + 2 * _HOP_GAS

    def test_simulate_two_hops_matches_sequential(self):
        s, liq = stable(), liquid()
        sp = SwapPath([s, liq], [USDC, DAI, WETH])
        qty = 1_000 * 10**6
        mid = s.out_for_in(qty, USDC)
        expected = liq.out_for_in(mid, DAI)
        assert sp.simulate(qty) == expected

    def test_simulate_steps_last_matches_simulate(self):
        sp = SwapPath([stable(), liquid()], [USDC, DAI, WETH])
        qty = 1_000 * 10**6
        assert sp.simulate_steps(qty)[-1] == sp.simulate(qty)

    def test_simulate_single_hop_matches_pool(self):
        pool = direct()
        sp = SwapPath([pool], [USDC, WETH])
        qty = 1_000 * 10**6
        assert sp.simulate(qty) == pool.out_for_in(qty, USDC)

    def test_net_output_less_than_gross(self):
        sp = SwapPath([direct()], [USDC, WETH])
        qty = 1_000 * 10**6
        gross = sp.simulate(qty)
        net = sp.net_output(qty, gas_gwei=10)
        assert net <= gross

    def test_net_output_non_negative(self):
        sp = SwapPath([direct()], [USDC, WETH])
        assert sp.net_output(1, gas_gwei=10_000) >= 0


# ===========================================================================
# PathFinder
# ===========================================================================

class TestPathFinder:
    def _finder(self, *pools) -> PathFinder:
        return PathFinder(list(pools))

    def test_direct_path_found(self):
        paths = self._finder(direct()).enumerate(USDC, WETH)
        assert len(paths) == 1
        assert paths[0].hops == 1

    def test_two_hop_path_found(self):
        paths = self._finder(stable(), liquid()).enumerate(USDC, WETH)
        assert len(paths) == 1
        assert paths[0].hops == 2

    def test_no_path_returns_empty(self):
        assert self._finder(direct()).enumerate(USDC, LINK) == []

    def test_find_optimal_no_path_raises(self):
        with pytest.raises(ValueError, match="No route found"):
            self._finder(direct()).find_optimal(USDC, LINK, 10**6, gas_gwei=1)

    def test_low_gas_prefers_multihop(self):
        finder = self._finder(direct(), stable(), liquid())
        best, _ = finder.find_optimal(USDC, WETH, 1_000 * 10**6, gas_gwei=1)
        assert best.hops == 2

    def test_high_gas_prefers_direct(self):
        finder = self._finder(direct(), stable(), liquid())
        best, _ = finder.find_optimal(USDC, WETH, 1_000 * 10**6, gas_gwei=200)
        assert best.hops == 1

    def test_no_pool_used_twice(self):
        for path in self._finder(direct(), stable(), liquid()).enumerate(USDC, WETH):
            ids = [id(p) for p in path.pools]
            assert len(ids) == len(set(ids))

    def test_rank_all_net_equals_gross_minus_gas(self):
        finder = self._finder(direct(), stable(), liquid())
        for row in finder.rank_all(USDC, WETH, 1_000 * 10**6, gas_gwei=5):
            expected = max(0, row["gross_out"] - row["gas_cost_wei"])
            assert row["net_out"] == expected

    def test_simulate_matches_sequential_chain(self):
        s, liq = stable(), liquid()
        sp = SwapPath([s, liq], [USDC, DAI, WETH])
        qty = 2_000 * 10**6
        mid = s.out_for_in(qty, USDC)
        assert sp.simulate(qty) == liq.out_for_in(mid, DAI)
