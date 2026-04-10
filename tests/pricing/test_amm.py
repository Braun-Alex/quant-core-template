"""
Unit tests for src.pricing.amm.PoolState.

Fixtures use realistic Ethereum mainnet reserves where possible.
All chain-dependent methods are exercised via mocked ChainClient objects.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.types import Address, Token
from src.pricing.amm import PoolState, _FEE_SCALE

# ---------------------------------------------------------------------------
# Token definitions
# ---------------------------------------------------------------------------

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)
DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)
SHIB = Token(Address("0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE"), "SHIB", 18)

_PAIR = Address("0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc")


def _pool(left, right, q_left, q_right, fee=30) -> PoolState:
    return PoolState(contract=_PAIR, left=left, right=right,
                     qty_left=q_left, qty_right=q_right, fee_bps=fee)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

class TestPoolStateConstruction:
    def test_happy_path(self):
        ps = _pool(WETH, USDC, 10**18, 2_000 * 10**6)
        assert ps.qty_left == 10**18
        assert ps.qty_right == 2_000 * 10**6
        assert ps.fee_bps == 30

    def test_zero_left_reserve_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _pool(WETH, USDC, 0, 10**6)

    def test_negative_right_reserve_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _pool(WETH, USDC, 10**18, -1)

    def test_same_token_both_sides_raises(self):
        with pytest.raises(ValueError, match="same token"):
            _pool(WETH, WETH, 10**18, 10**18)

    def test_fee_at_scale_raises(self):
        with pytest.raises(ValueError, match="fee_bps"):
            _pool(WETH, USDC, 10**18, 10**6, fee=_FEE_SCALE)

    def test_fee_zero_allowed(self):
        ps = _pool(WETH, USDC, 10**18, 10**6, fee=0)
        assert ps.fee_bps == 0

    def test_fee_9999_allowed(self):
        ps = _pool(WETH, USDC, 10**18, 10**6, fee=9_999)
        assert ps.fee_bps == 9_999


# ---------------------------------------------------------------------------
# out_for_in
# ---------------------------------------------------------------------------

class TestOutForIn:
    @pytest.fixture
    def weth_usdc(self):
        # 1 000 WETH / 2 000 000 USDC  →  spot ≈ $2 000
        return _pool(WETH, USDC, 1_000 * 10**18, 2_000_000 * 10**6)

    def test_result_is_int(self, weth_usdc):
        assert isinstance(weth_usdc.out_for_in(10**18, WETH), int)

    def test_selling_weth_receives_less_than_spot(self, weth_usdc):
        # Spot ≈ 2 000 USDC per WETH; fee + impact → receive slightly less
        usdc_out = weth_usdc.out_for_in(10**18, WETH)
        assert usdc_out < 2_000 * 10**6
        assert usdc_out > int(1_990 * 10**6)

    def test_selling_usdc_receives_less_than_spot(self, weth_usdc):
        eth_out = weth_usdc.out_for_in(2_000 * 10**6, USDC)
        assert eth_out < 10**18
        assert eth_out > int(0.99 * 10**18)

    def test_larger_sell_more_received(self, weth_usdc):
        small = weth_usdc.out_for_in(100 * 10**6, USDC)
        large = weth_usdc.out_for_in(1_000 * 10**6, USDC)
        assert large > small

    def test_float_input_raises(self, weth_usdc):
        with pytest.raises(TypeError):
            weth_usdc.out_for_in(1.0, WETH)   # Type: ignore

    def test_zero_input_raises(self, weth_usdc):
        with pytest.raises(ValueError, match="positive"):
            weth_usdc.out_for_in(0, WETH)

    def test_unknown_token_raises(self, weth_usdc):
        with pytest.raises(ValueError):
            weth_usdc.out_for_in(10**18, DAI)

    def test_uniswap_v2_core_test_vector(self):
        """
        Official test vector from the Uniswap V2 core Hardhat suite.
        Pool: 5e18 token0 / 10e18 token1
        Sell: 1e18 token0
        Expected output: 1 662 497 915 624 478 906
        """
        ps = _pool(WETH, DAI, 5 * 10**18, 10 * 10**18)
        assert ps.out_for_in(10**18, WETH) == 1_662_497_915_624_478_906

    def test_fee_formula_matches_997_1000(self):
        """
        9970/10000 must be integer-division equivalent to 997/1000
        for all inputs (scales differ by factor of 10; both cancel).
        """
        liq_in, liq_out = 10**24, 5 * 10**23
        for qty in [1, 10**15, 10**18, 10**21, 123_456_789_012]:
            v1 = (qty * 9_970 * liq_out) // (liq_in * 10_000 + qty * 9_970)
            v2 = (qty * 997 * liq_out) // (liq_in * 1_000 + qty * 997)
            assert v1 == v2, f"Mismatch at qty={qty}: {v1} != {v2}"

    def test_integer_math_at_uint112_boundary(self):
        uint112_max = 2**112 - 1
        ps = _pool(WETH, DAI, uint112_max // 2, uint112_max // 2)
        qty = uint112_max // 1_000
        result = ps.out_for_in(qty, WETH)
        assert isinstance(result, int)
        assert result > 0

    def test_output_bounded_by_pool_liquidity(self):
        ps = _pool(WETH, USDC, 10**18, 2_000 * 10**6)
        out = ps.out_for_in(ps.qty_right * 999, USDC)   # Absurdly large sell
        assert out < ps.qty_left

    def test_zero_fee_is_pure_constant_product(self):
        r_l, r_r = 10**18, 2_000 * 10**6
        ps = _pool(WETH, USDC, r_l, r_r, fee=0)
        qty = 10**16
        expected = (qty * r_r) // (r_l + qty)   # Pure k = x*y
        assert ps.out_for_in(qty, WETH) == expected


# ---------------------------------------------------------------------------
# in_for_out
# ---------------------------------------------------------------------------

class TestInForOut:
    @pytest.fixture
    def pool(self):
        return _pool(WETH, USDC, 1_000 * 10**18, 2_000_000 * 10**6)

    def test_result_is_int(self, pool):
        assert isinstance(pool.in_for_out(10**17, WETH), int)

    def test_buying_weth_costs_more_than_spot(self, pool):
        cost = pool.in_for_out(10**17, WETH)   # Buy 0.1 WETH
        assert cost > 200 * 10**6   # Spot ~$200
        assert cost < 210 * 10**6

    def test_round_trip_ceiling(self, pool):
        """in_for_out uses ceiling division → out_for_in(result) >= desired."""
        for desired in (1, 1_000, 10**15, 5 * 10**17):
            needed = pool.in_for_out(desired, WETH)
            received = pool.out_for_in(needed, USDC)
            assert received >= desired, f"Round-trip failed for {desired}"

    def test_buying_entire_reserve_raises(self, pool):
        with pytest.raises(ValueError, match="pool only holds"):
            pool.in_for_out(pool.qty_left, WETH)

    def test_float_input_raises(self, pool):
        with pytest.raises(TypeError):
            pool.in_for_out(0.5, WETH)   # Type: ignore

    def test_unknown_token_raises(self, pool):
        with pytest.raises(ValueError):
            pool.in_for_out(10**18, DAI)

    def test_buying_right_token(self):
        """Exercises the token_out == right branch."""
        ps = _pool(WETH, USDC, 10**18, 2_000 * 10**6)
        needed = ps.in_for_out(500 * 10**6, USDC)
        received = ps.out_for_in(needed, WETH)
        assert received >= 500 * 10**6


# ---------------------------------------------------------------------------
# marginal_price / fill_price / slippage
# ---------------------------------------------------------------------------

class TestPriceHelpers:
    @pytest.fixture
    def ps(self):
        return _pool(WETH, USDC, 1_000 * 10**18, 2_000_000 * 10**6)

    def test_marginal_price_is_decimal(self, ps):
        assert isinstance(ps.marginal_price(WETH), Decimal)

    def test_marginal_price_balanced_pool(self):
        eq = _pool(WETH, DAI, 10**18, 10**18)
        assert eq.marginal_price(WETH) == Decimal(1)

    def test_inverse_prices_product_one(self, ps):
        p1 = ps.marginal_price(WETH)
        p2 = ps.marginal_price(USDC)
        assert abs(p1 * p2 - Decimal(1)) < Decimal("1e-28")

    def test_fill_price_worse_than_marginal(self, ps):
        mp = ps.marginal_price(USDC)
        fp = ps.fill_price(2_000 * 10**6, USDC)
        assert fp < mp

    def test_slippage_positive(self, ps):
        assert ps.slippage(2_000 * 10**6, USDC) > 0

    def test_slippage_under_one(self, ps):
        assert ps.slippage(2_000 * 10**6, USDC) < 1

    def test_slippage_grows_with_size(self, ps):
        s_small = ps.slippage(1_000 * 10**6, USDC)
        s_large = ps.slippage(100_000 * 10**6, USDC)
        assert s_large > s_small

    def test_marginal_zero_returns_zero_slippage(self, ps):
        with patch.object(ps, "marginal_price", return_value=Decimal(0)):
            assert ps.slippage(10**18, WETH) == Decimal(0)

    def test_unknown_token_marginal_raises(self, ps):
        with pytest.raises(ValueError):
            ps.marginal_price(DAI)


# ---------------------------------------------------------------------------
# after_sell
# ---------------------------------------------------------------------------

class TestAfterSell:
    @pytest.fixture
    def ps(self):
        return _pool(WETH, USDC, 1_000 * 10**18, 2_000_000 * 10**6)

    def test_returns_new_object(self, ps):
        new = ps.after_sell(10**18, WETH)
        assert new is not ps

    def test_original_unmodified(self, ps):
        orig_l = ps.qty_left
        ps.after_sell(10**18, WETH)
        assert ps.qty_left == orig_l

    def test_left_increases_when_selling_left(self, ps):
        new = ps.after_sell(10**18, WETH)
        assert new.qty_left > ps.qty_left

    def test_right_decreases_when_selling_left(self, ps):
        new = ps.after_sell(10**18, WETH)
        assert new.qty_right < ps.qty_right

    def test_deltas_match_formula(self, ps):
        qty = 10**18
        out = ps.out_for_in(qty, WETH)
        new = ps.after_sell(qty, WETH)
        assert new.qty_left == ps.qty_left + qty
        assert new.qty_right == ps.qty_right - out

    def test_selling_right_token(self, ps):
        qty = 2_000 * 10**6
        out = ps.out_for_in(qty, USDC)
        new = ps.after_sell(qty, USDC)
        assert new.qty_right == ps.qty_right + qty
        assert new.qty_left == ps.qty_left - out

    def test_metadata_preserved(self, ps):
        new = ps.after_sell(10**18, WETH)
        assert new.contract == ps.contract
        assert new.fee_bps == ps.fee_bps
        assert new.left == WETH
        assert new.right == USDC
