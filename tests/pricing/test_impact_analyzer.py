"""
Tests for ImpactAnalyser and the CLI entry point.
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import patch

import pytest

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.impact_analyzer import ImpactAnalyzer, render, main

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)
DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)
PAIR = Address("0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc")


def _pool() -> PoolState:
    return PoolState(PAIR, WETH, USDC, 1_000 * 10**18, 2_000_000 * 10**6, fee_bps=30)


@pytest.fixture
def analyzer() -> ImpactAnalyzer:
    return ImpactAnalyzer(_pool())


class TestCLI:
    def test_valid_args_exit_0(self):
        with (
            patch("src.pricing.impact_analyzer.ChainClient"),
            patch("src.pricing.impact_analyzer.PoolState") as MockPS,
        ):
            MockPS.load.return_value = _pool()
            code = main([PAIR.checksum, "--sell", "USDC", "--amounts", "1000"])
        assert code == 0

    def test_unknown_token_exit_1(self):
        with (
            patch("src.pricing.impact_analyzer.ChainClient"),
            patch("src.pricing.impact_analyzer.PoolState") as MockPS,
        ):
            MockPS.load.return_value = _pool()
            code = main([PAIR.checksum, "--sell", "SHIB", "--amounts", "1000"])
        assert code == 1

    def test_bad_amounts_exit_1(self):
        with (
            patch("src.pricing.impact_analyzer.ChainClient"),
            patch("src.pricing.impact_analyzer.PoolState") as MockPS,
        ):
            MockPS.load.return_value = _pool()
            code = main([PAIR.checksum, "--sell", "USDC", "--amounts", "x,y"])
        assert code == 1


class TestTable:
    def test_empty_amounts(self, analyzer):
        assert analyzer.table(USDC, []) == []

    def test_fill_worse_than_spot(self, analyzer):
        row = analyzer.table(USDC, [1_000 * 10 ** 6])[0]
        assert row["fill"] > row["spot"]   # Fill price (sold/bought) > spot → pay more

    def test_one_row_per_amount(self, analyzer):
        rows = analyzer.table(USDC, [100 * 10 ** 6, 1_000 * 10 ** 6])
        assert len(rows) == 2

    def test_raw_out_positive(self, analyzer):
        row = analyzer.table(USDC, [1_000 * 10 ** 6])[0]
        assert isinstance(row["raw_out"], int) and row["raw_out"] > 0

    def test_impact_grows_with_size(self, analyzer):
        rows = analyzer.table(USDC, [1_000 * 10 ** 6, 100_000 * 10 ** 6])
        assert rows[1]["impact_pct"] > rows[0]["impact_pct"]


class TestRender:
    def test_contains_token_names(self, analyzer):
        rows = analyzer.table(USDC, [1_000 * 10 ** 6])
        ps = _pool()
        text = render(rows, USDC, WETH, ps, 0, Decimal("1"))
        assert "USDC" in text and "WETH" in text

    def test_is_string(self, analyzer):
        rows = analyzer.table(USDC, [1_000 * 10 ** 6])
        ps = _pool()
        assert isinstance(render(rows, USDC, WETH, ps, 0, Decimal("1")), str)

    def test_contains_max_trade_line(self, analyzer):
        rows = analyzer.table(USDC, [1_000 * 10 ** 6])
        ps = _pool()
        text = render(rows, USDC, WETH, ps, 19_000 * 10**6, Decimal("1"))
        assert "Largest trade" in text


class TestMaxTradeBelow:
    def test_zero_threshold_raises(self, analyzer):
        with pytest.raises(ValueError, match="positive"):
            analyzer.max_trade_below(USDC, Decimal("0"))

    def test_result_within_threshold(self, analyzer):
        mx = analyzer.max_trade_below(USDC, Decimal("1"))
        assert analyzer.pool.slippage(mx, USDC) <= Decimal("0.01")

    def test_stricter_threshold_smaller_result(self, analyzer):
        m1 = analyzer.max_trade_below(USDC, Decimal("1"))
        m5 = analyzer.max_trade_below(USDC, Decimal("5"))
        assert m5 > m1


class TestCostBreakdown:
    def test_gross_matches_amm(self, analyzer):
        qty = 1_000 * 10**6
        r = analyzer.cost_breakdown(qty, USDC, 50)
        assert r["gross_out"] == analyzer.pool.out_for_in(qty, USDC)

    def test_net_never_negative(self, analyzer):
        r = analyzer.cost_breakdown(1, USDC, 100_000, gas_units=500_000)
        assert r["net_out"] >= 0

    def test_expected_keys(self, analyzer):
        r = analyzer.cost_breakdown(1_000 * 10 ** 6, USDC, 50)
        assert set(r) >= {"gross_out", "gas_wei", "gas_in_output", "net_out", "effective_price"}

    def test_effective_price_is_decimal(self, analyzer):
        r = analyzer.cost_breakdown(1_000 * 10 ** 6, USDC, 50)
        assert isinstance(r["effective_price"], Decimal)
