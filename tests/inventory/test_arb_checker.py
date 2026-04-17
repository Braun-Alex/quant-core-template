"""
Tests for ArbChecker.

All external dependencies are mocked.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.inventory.arb_checker import ArbChecker, StaticDexSource
from src.inventory.pnl import PnLTracker
from src.inventory.tracker import VenueTracker, Venue


def _book(bid="2010.00", ask="2010.50", qty="10"):
    bp, ap, q = Decimal(bid), Decimal(ask), Decimal(qty)
    return {
        "symbol": "ETH/USDT",
        "timestamp": 1700000000000,
        "bids": [(bp, q), (bp - Decimal("0.5"), q)],
        "asks": [(ap, q), (ap + Decimal("0.5"), q)],
        "best_bid": (bp, q),
        "best_ask": (ap, q),
        "mid_price": (bp + ap) / 2,
        "spread_bps": (ap - bp) / ((bp + ap) / 2) * Decimal("10000")
    }


def _dex(price="2000.00", impact="1.2", fee="30"):
    return StaticDexSource(
        price=Decimal(price),
        impact_bps=Decimal(impact),
        fee_bps=Decimal(fee)
    )


def _cex_client(bid="2010.00", ask="2010.50", fee="0.001"):
    client = MagicMock()
    client.fetch_order_book.return_value = _book(bid, ask)
    client.get_trading_fees.return_value = {"maker": Decimal("0.001"), "taker": Decimal(fee)}
    return client


def _inventory(eth_binance="10", usdt_binance="50000", eth_wallet="10", usdt_wallet="50000"):
    t = VenueTracker([Venue.BINANCE, Venue.WALLET])
    t.update_from_cex(Venue.BINANCE, {
        "ETH": {"free": eth_binance, "locked": "0"},
        "USDT": {"free": usdt_binance, "locked": "0"}
    })
    t.update_from_wallet(Venue.WALLET, {"ETH": eth_wallet, "USDT": usdt_wallet})
    return t


def _checker(**kw) -> ArbChecker:
    return ArbChecker(
        dex_source=_dex(kw.get("dex_price", "2000")),
        cex_client=_cex_client(kw.get("bid", "2010"), kw.get("ask", "2010.50")),
        inventory=_inventory(
            kw.get("eth_binance", "10"),
            kw.get("usdt_binance", "50000"),
            kw.get("eth_wallet", "10"),
            kw.get("usdt_wallet", "50000")
        ),
        pnl_tracker=PnLTracker()
    )


class TestDirection:
    def test_buy_dex_sell_cex_when_dex_below_bid(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015")
        r = c.assess("ETH/USDT")
        assert r["direction"] == "buy_dex_sell_cex"

    def test_buy_cex_sell_dex_when_dex_above_ask(self):
        c = _checker(dex_price="2020", bid="2009", ask="2010")
        r = c.assess("ETH/USDT")
        assert r["direction"] == "buy_cex_sell_dex"

    def test_no_direction_when_equal_to_bid(self):
        c = _checker(dex_price="2010.00", bid="2010.00", ask="2010.50")
        r = c.assess("ETH/USDT")
        assert r["direction"] is None

    def test_no_direction_when_inside_spread(self):
        c = _checker(dex_price="2010.25", bid="2010", ask="2010.50")
        r = c.assess("ETH/USDT")
        assert r["direction"] is None


class TestGapAndCosts:
    def test_gap_bps_calculated(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015")
        r = c.assess("ETH/USDT")
        assert r["gap_bps"] == pytest.approx(Decimal("50"), rel=Decimal("0.01"))

    def test_net_pnl_equals_gap_minus_costs(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015")
        r = c.assess("ETH/USDT")
        assert r["estimated_net_pnl_bps"] == pytest.approx(
            r["gap_bps"] - r["estimated_costs_bps"], rel=Decimal("0.001")
        )

    def test_gap_zero_when_no_direction(self):
        c = _checker(dex_price="2010.25", bid="2010", ask="2010.50")
        r = c.assess("ETH/USDT")
        assert r["gap_bps"] == Decimal("0")


class TestInventoryCheck:
    def test_ok_when_no_direction(self):
        c = _checker(dex_price="2010.25", bid="2010", ask="2010.50")
        r = c.assess("ETH/USDT")
        assert r["inventory_ok"] is True

    def test_ok_when_sufficient(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015")
        r = c.assess("ETH/USDT")
        assert r["inventory_ok"] is True

    def test_fails_when_no_quote_on_wallet(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015", usdt_wallet="0")
        r = c.assess("ETH/USDT")
        assert r["inventory_ok"] is False

    def test_fails_when_no_base_on_cex(self):
        c = _checker(dex_price="2000", bid="2010", ask="2015", eth_binance="0")
        r = c.assess("ETH/USDT")
        assert r["inventory_ok"] is False


class TestExecutableFlag:
    def test_large_gap_is_executable(self):
        c = _checker(dex_price="2000", bid="2100", ask="2110")
        r = c.assess("ETH/USDT")
        assert r["executable"] is True

    def test_no_inventory_not_executable(self):
        c = _checker(dex_price="2000", bid="2100", ask="2110", usdt_wallet="0", eth_binance="0")
        r = c.assess("ETH/USDT")
        assert r["executable"] is False

    def test_no_direction_not_executable(self):
        c = _checker(dex_price="2010.25", bid="2010", ask="2010.50")
        r = c.assess("ETH/USDT")
        assert r["executable"] is False


class TestResultSchema:
    def setup_method(self):
        self.r = _checker().assess("ETH/USDT")

    def test_details_keys(self):
        detail_keys = {"dex_price_impact_bps", "cex_slippage_bps", "cex_fee_bps", "dex_fee_bps", "gas_cost_usd"}
        assert detail_keys <= set(self.r["details"].keys())

    def test_timestamp_is_datetime(self):
        assert isinstance(self.r["timestamp"], datetime)

    def test_pair_preserved(self):
        assert self.r["pair"] == "ETH/USDT"

    def test_required_keys(self):
        required = {
            "pair", "timestamp", "dex_price", "cex_bid", "cex_ask",
            "gap_bps", "direction", "estimated_costs_bps",
            "estimated_net_pnl_bps", "inventory_ok", "executable", "details"
        }
        assert required <= set(self.r.keys())

    def test_direction_is_str_or_none(self):
        d = self.r["direction"]
        assert d is None or isinstance(d, str)


class TestStaticDexSource:
    def test_returns_fixed_price(self):
        src = StaticDexSource(price=Decimal("2000"))
        r = src.get_dex_quote("ETH", "USDT", Decimal("1"))
        assert r["price"] == Decimal("2000")

    def test_default_fee_bps(self):
        src = StaticDexSource(price=Decimal("2000"))
        r = src.get_dex_quote("ETH", "USDT", Decimal("1"))
        assert r["fee_bps"] == Decimal("30")

    def test_custom_fee_bps(self):
        src = StaticDexSource(price=Decimal("2000"), fee_bps=Decimal("5"))
        r = src.get_dex_quote("ETH", "USDT", Decimal("1"))
        assert r["fee_bps"] == Decimal("5")
