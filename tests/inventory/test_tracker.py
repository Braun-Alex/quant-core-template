"""
Tests for VenueTracker and FillLedger.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.inventory.tracker import FillLedger, VenueTracker, Venue


def _tracker() -> VenueTracker:
    return VenueTracker([Venue.BINANCE, Venue.WALLET])


def _cex_data() -> dict:
    return {
        "ETH": {"free": Decimal("10"), "locked": Decimal("2"), "total": Decimal("12")},
        "USDT": {"free": Decimal("20000"), "locked": Decimal("0"), "total": Decimal("20000")}
    }


class TestFillLedger:
    def test_weighted_average_cost(self):
        ledger = FillLedger()
        ledger.record("ETH", "buy", Decimal("1"), Decimal("2000"))
        ledger.record("ETH", "buy", Decimal("1"), Decimal("3000"))
        pos = ledger.position("ETH")
        assert pos.avg_cost == Decimal("2500")

    def test_sell_books_realized_pnl(self):
        ledger = FillLedger()
        ledger.record("ETH", "buy", Decimal("2"), Decimal("2000"))
        ledger.record("ETH", "sell", Decimal("1"), Decimal("2200"))
        pos = ledger.position("ETH")
        assert pos.realized_pnl == Decimal("200")

    def test_unrealized_pnl(self):
        ledger = FillLedger()
        ledger.record("ETH", "buy", Decimal("1"), Decimal("2000"))
        assert ledger.unrealized_pnl("ETH", Decimal("2200")) == Decimal("200")

    def test_zero_qty_raises(self):
        with pytest.raises(ValueError, match="qty"):
            FillLedger().record("ETH", "buy", Decimal("0"), Decimal("2000"))

    def test_negative_fee_raises(self):
        with pytest.raises(ValueError, match="fee"):
            FillLedger().record("ETH", "buy", Decimal("1"), Decimal("2000"), fee=Decimal("-1"))

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side"):
            FillLedger().record("ETH", "long", Decimal("1"), Decimal("2000"))


class TestCanExecute:
    def setup_method(self):
        self.t = _tracker()
        self.t.update_from_cex(Venue.BINANCE, _cex_data())
        self.t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("8")})

    def test_fails_insufficient_buy(self):
        result = self.t.can_execute(
            Venue.BINANCE, "USDT", Decimal("99999"),
            Venue.WALLET, "ETH", Decimal("1")
        )
        assert result["can_execute"] is False
        assert "USDT" in result["reason"]

    def test_fails_insufficient_sell(self):
        result = self.t.can_execute(
            Venue.BINANCE, "USDT", Decimal("100"),
            Venue.WALLET, "ETH", Decimal("100")
        )
        assert result["can_execute"] is False
        assert "ETH" in result["reason"]

    def test_passes_when_sufficient(self):
        result = self.t.can_execute(
            Venue.BINANCE, "USDT", Decimal("4000"),
            Venue.WALLET, "ETH", Decimal("2")
        )
        assert result["can_execute"] is True
        assert result["reason"] is None

    def test_returns_available_amounts(self):
        result = self.t.can_execute(
            Venue.BINANCE, "USDT", Decimal("100"),
            Venue.WALLET, "ETH", Decimal("1")
        )
        assert result["buy_venue_available"] == Decimal("20000")
        assert result["sell_venue_available"] == Decimal("8")


class TestVenueTrackerUpdate:
    def test_update_replaces_previous(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, _cex_data())
        t.update_from_cex(Venue.BINANCE, {"BTC": {"free": "1", "locked": "0", "total": "1"}})
        assert t.available(Venue.BINANCE, "ETH") == Decimal("0")
        assert t.available(Venue.BINANCE, "BTC") == Decimal("1")

    def test_update_from_cex_stores_balances(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, _cex_data())
        assert t.available(Venue.BINANCE, "ETH") == Decimal("10")

    def test_non_dict_cex_entries_skipped(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "5", "locked": "0", "total": "5"}, "info": "meta"})
        assert "info" not in t.snapshot()["venues"]["binance"]


class TestSnapshot:
    def test_empty_tracker(self):
        t = VenueTracker([Venue.BINANCE])
        snap = t.snapshot()
        assert snap["totals"] == {}

    def test_aggregates_across_venues(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, _cex_data())
        t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("8")})
        snap = t.snapshot()
        assert snap["totals"]["ETH"] == Decimal("20")


class TestApplyTrade:
    def setup_method(self):
        self.t = _tracker()
        self.t.update_from_cex(Venue.BINANCE, {
            "ETH": {"free": "10", "locked": "0", "total": "10"},
            "USDT": {"free": "20000", "locked": "0", "total": "20000"}
        })

    def test_sell_decreases_base_increases_quote(self):
        self.t.apply_trade(
            Venue.BINANCE, "sell", "ETH", "USDT",
            Decimal("2"), Decimal("4000"), Decimal("4"), "USDT"
        )
        assert self.t.available(Venue.BINANCE, "ETH") == Decimal("8")
        assert self.t.available(Venue.BINANCE, "USDT") == Decimal("23996")

    def test_buy_increases_base_decreases_quote(self):
        self.t.apply_trade(
            Venue.BINANCE, "buy", "ETH", "USDT",
            Decimal("1"), Decimal("2000"), Decimal("2"), "USDT"
        )
        assert self.t.available(Venue.BINANCE, "ETH") == Decimal("11")
        assert self.t.available(Venue.BINANCE, "USDT") == Decimal("17998")


class TestSkewReport:
    def test_detects_imbalance(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "9", "locked": "0", "total": "9"}})
        t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("1")})
        skew = t.skew_report("ETH")
        assert skew["needs_rebalance"] is True
        assert skew["max_deviation_pct"] > 30.0

    def test_balanced_not_flagged(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "5", "locked": "0", "total": "5"}})
        t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("5")})
        skew = t.skew_report("ETH")
        assert skew["needs_rebalance"] is False

    def test_all_skews_sorted_alphabetically(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, {
            "USDT": {"free": "5", "locked": "0", "total": "5"},
            "ETH": {"free": "5", "locked": "0", "total": "5"},
        })
        t.update_from_wallet(Venue.WALLET, {})
        skews = t.all_skews()
        names = [s["asset"] for s in skews]
        assert names == sorted(names)

    def test_all_skews_returns_all_assets(self):
        t = _tracker()
        t.update_from_cex(Venue.BINANCE, {
            "ETH": {"free": "9", "locked": "0", "total": "9"},
            "USDT": {"free": "9000", "locked": "0", "total": "9000"}
        })
        t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("1")})
        skews = t.all_skews()
        assets = {s["asset"] for s in skews}
        assert "ETH" in assets
        assert "USDT" in assets
