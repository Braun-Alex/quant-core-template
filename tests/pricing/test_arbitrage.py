"""Tests for ArbitrageDetector.
Tests cover: Opportunity properties, triangular cycle detection, cross-pool spread detection."""

from __future__ import annotations

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.arbitrage import ArbitrageDetector, Opportunity

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)
USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)


def _ps(n, left, right, ql, qr, fee=30):
    return PoolState(Address("0x" + n.zfill(40)), left, right, ql, qr, fee_bps=fee)


class TestOpportunity:
    def _opp(self, gross=1000, gas=200):
        pool = _ps("1", WETH, DAI, 10**18, 2_000 * 10**18)
        from src.pricing.router import SwapPath
        return Opportunity(SwapPath([pool], [WETH, DAI]), 10**18,
                           gross, gas, WETH, "triangular")

    def test_net_profit(self):
        assert self._opp(1000, 200).net_profit == 800

    def test_net_profitable(self):
        assert self._opp(1000, 200).is_net_profitable
        assert not self._opp(100, 500).is_net_profitable

    def test_is_profitable_when_gross_positive(self):
        assert self._opp(1).is_profitable

    def test_not_profitable_when_gross_zero(self):
        assert not self._opp(0).is_profitable


class TestTriangular:
    def _imbalanced_triangle(self):
        """WETH→DAI→USDC→WETH where USDC/WETH pool has cheap WETH."""
        return ArbitrageDetector([
            _ps("1", WETH, DAI, 10**18, 2_000 * 10**18),
            _ps("2", DAI, USDC, 2_000 * 10**18, 2_000 * 10**18),
            _ps("3", USDC, WETH, 10**18, 100 * 10**18)
        ])

    def test_finds_opportunity_in_imbalanced_triangle(self):
        det = self._imbalanced_triangle()
        opps = det.find_triangular(WETH, 10**17, gas_gwei=0)
        assert len(opps) > 0
        assert all(o.gross_profit > 0 for o in opps)

    def test_best_triangular_none_when_no_opportunity(self):
        det = ArbitrageDetector([_ps("1", WETH, DAI, 10**18, 2_000 * 10**18)])
        assert det.best_triangular(WETH, 10**17, gas_gwei=0) is None

    def test_sorted_descending(self):
        det = self._imbalanced_triangle()
        opps = det.find_triangular(WETH, 10**17, gas_gwei=0)
        profits = [o.gross_profit for o in opps]
        assert profits == sorted(profits, reverse=True)

    def test_balanced_pools_no_opportunity(self):
        det = ArbitrageDetector([
            _ps("1", WETH, DAI, 10**18, 2_000 * 10**18),
            _ps("2", DAI, USDC, 2_000 * 10**18, 2_000 * 10**18),
            _ps("3", USDC, WETH, 2_000 * 10**18, 10**18)
        ])
        assert det.find_triangular(WETH, 10**17, gas_gwei=0) == []

    def test_strategy_label(self):
        det = self._imbalanced_triangle()
        opps = det.find_triangular(WETH, 10**17, gas_gwei=0)
        assert all(o.strategy == "triangular" for o in opps)


class TestCrossPool:
    def _imbalanced_pair(self):
        """Same WETH/DAI pair — one pool much more expensive."""
        return ArbitrageDetector([
            _ps("1", WETH, DAI, 10 * 10**18, 10_000 * 10**18),
            _ps("2", WETH, DAI, 10 * 10**18, 30_000 * 10**18)
        ])

    def test_finds_cross_pool_opportunity(self):
        opps = self._imbalanced_pair().find_cross_pool(WETH, DAI, 10**18, gas_gwei=0)
        assert len(opps) > 0

    def test_identical_pools_no_opportunity(self):
        det = ArbitrageDetector([
            _ps("1", WETH, DAI, 10 * 10**18, 20_000 * 10**18),
            _ps("2", WETH, DAI, 10 * 10**18, 20_000 * 10**18)
        ])
        assert det.find_cross_pool(WETH, DAI, 10**18, gas_gwei=0) == []
