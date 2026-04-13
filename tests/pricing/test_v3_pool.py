"""Tests for V3Pool."""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.core.types import Address, Token
from src.pricing.v3_pool import Q96, V3Pool

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)
ADDR = Address("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")

# sqrt(2000) × Q96 - represents WETH/USDC price ≈ 2 000 in raw units
_SQRT_2000 = int(Decimal(2000).sqrt() * Decimal(Q96))


def _pool(sqp=_SQRT_2000, liq=10**22, fee=3_000) -> V3Pool:
    return V3Pool(ADDR, WETH, USDC, sqp, liq, fee)


class TestConstruction:
    def test_valid(self):
        p = _pool()
        assert p.fee_ppm == 3_000

    def test_invalid_fee_raises(self):
        with pytest.raises(ValueError, match="fee_ppm"):
            V3Pool(ADDR, WETH, USDC, Q96, 10**18, 999)

    def test_zero_sqrt_price_raises(self):
        with pytest.raises(ValueError, match="positive"):
            V3Pool(ADDR, WETH, USDC, 0, 10**18, 3_000)

    def test_same_token_raises(self):
        with pytest.raises(ValueError, match="different"):
            V3Pool(ADDR, WETH, WETH, Q96, 10**18, 3_000)


class TestOutForIn:
    def test_zero_for_one_returns_positive_int(self):
        out = _pool().out_for_in(10**18, WETH)
        assert isinstance(out, int) and out > 0

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _pool().out_for_in(0, WETH)

    def test_larger_input_more_output(self):
        p = _pool()
        assert p.out_for_in(10**18, WETH) < p.out_for_in(2 * 10**18, WETH)

    def test_one_for_zero_returns_positive_int(self):
        out = _pool().out_for_in(2_000 * 10**18, USDC)
        assert isinstance(out, int) and out > 0

    def test_higher_fee_lower_output(self):
        lo = _pool(fee=500).out_for_in(10**18, WETH)
        hi = _pool(fee=10_000).out_for_in(10**18, WETH)
        assert lo > hi

    def test_known_manual_vector(self):
        """Verify against a hand-computed Q96 reference (fee=0 ppm not valid; use 100)."""
        # With sqrtP = Q96 (price=1), L = Q96, fee=3000, qty=Q96//100
        p = V3Pool(ADDR, WETH, USDC, Q96, Q96, 3_000)
        qty = Q96 // 100
        net = qty * 997_000 // 1_000_000
        extra = (net * Q96 + Q96 - 1) // Q96
        new_sp = (Q96 * Q96) // (Q96 + extra)
        expected = Q96 * (Q96 - new_sp) // Q96
        assert p.out_for_in(qty, WETH) == expected

    def test_unknown_token_raises(self):
        DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)
        with pytest.raises(ValueError):
            _pool().out_for_in(10**18, DAI)


class TestPriceHelpers:
    def test_price_impact_positive(self):
        assert _pool().price_impact(10**18, WETH) > 0

    def test_spot_near_2000(self):
        sp = float(_pool().spot_price(WETH))
        assert 1990 < sp < 2010

    def test_inverse_prices_product_one(self):
        p = _pool()
        p0 = p.spot_price(WETH)
        p1 = p.spot_price(USDC)
        assert abs(float(p0 * p1) - 1.0) < 1e-6
