"""
Tests for RebalancePlanner.
"""

from __future__ import annotations

from decimal import Decimal

from src.inventory.rebalancer import (
    RebalancePlanner,
    TransferPlan
)
from src.inventory.tracker import VenueTracker, Venue


def _balanced() -> VenueTracker:
    t = VenueTracker([Venue.BINANCE, Venue.WALLET])
    t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "5", "locked": "0", "total": "5"}})
    t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("5")})
    return t


def _multi_asset() -> VenueTracker:
    t = VenueTracker([Venue.BINANCE, Venue.WALLET])
    t.update_from_cex(Venue.BINANCE, {
        "ETH": {"free": "9", "locked": "0", "total": "9"},
        "USDT": {"free": "9000", "locked": "0", "total": "9000"},
    })
    t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("1"), "USDT": Decimal("1000")})
    return t


def _skewed() -> VenueTracker:
    t = VenueTracker([Venue.BINANCE, Venue.WALLET])
    t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "9", "locked": "0", "total": "9"}})
    t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("1")})
    return t


class TestTransferPlan:
    def test_zero_fee(self):
        p = TransferPlan(Venue.BINANCE, Venue.WALLET, "ETH", Decimal("2"), Decimal("0"), 15)
        assert p.net_received == Decimal("2")

    def test_net_received(self):
        p = TransferPlan(Venue.BINANCE, Venue.WALLET, "ETH", Decimal("4"), Decimal("0.005"), 15)
        assert p.net_received == Decimal("3.995")


class TestCheckAll:
    def test_skewed_flagged(self):
        planner = RebalancePlanner(_skewed())
        skews = planner.check_all()
        eth = next(s for s in skews if s["asset"] == "ETH")
        assert eth["needs_rebalance"] is True

    def test_balanced_not_flagged(self):
        planner = RebalancePlanner(_balanced())
        skews = planner.check_all()
        eth = next(s for s in skews if s["asset"] == "ETH")
        assert eth["needs_rebalance"] is False


class TestPlan:
    def test_plan_direction_binance_to_wallet(self):
        planner = RebalancePlanner(_skewed())
        plans = planner.plan("ETH")
        assert plans[0].source == Venue.BINANCE
        assert plans[0].destination == Venue.WALLET

    def test_balanced_returns_empty(self):
        planner = RebalancePlanner(_balanced())
        assert planner.plan("ETH") == []

    def test_high_threshold_suppresses_plan(self):
        planner = RebalancePlanner(_skewed(), threshold_pct=99.0)
        assert planner.plan("ETH") == []

    def test_plan_respects_min_operating_balance(self):
        t = VenueTracker([Venue.BINANCE, Venue.WALLET])
        t.update_from_cex(Venue.BINANCE, {"ETH": {"free": "0.6", "locked": "0", "total": "0.6"}})
        t.update_from_wallet(Venue.WALLET, {"ETH": Decimal("0")})
        planner = RebalancePlanner(t)
        plans = planner.plan("ETH")
        if plans:
            # Min operating = 0.5; max transferable = 0.6 - 0.5 = 0.1
            assert plans[0].amount <= Decimal("0.1") + Decimal("0.001")

    def test_custom_weights_used(self):
        planner = RebalancePlanner(
            _skewed(),
            threshold_pct=0.0,
            target_weights={Venue.BINANCE: Decimal("0.7"), Venue.WALLET: Decimal("0.3")}
        )
        plans = planner.plan("ETH")
        assert isinstance(plans, list)


class TestPlanAll:
    def test_skewed_assets_included(self):
        planner = RebalancePlanner(_multi_asset())
        result = planner.plan_all()
        assert "ETH" in result

    def test_balanced_assets_excluded(self):
        planner = RebalancePlanner(_balanced())
        assert planner.plan_all() == {}

    def test_values_are_lists_of_transfer_plans(self):
        planner = RebalancePlanner(_multi_asset())
        for asset, plans in planner.plan_all().items():
            assert isinstance(plans, list)
            assert all(isinstance(p, TransferPlan) for p in plans)


class TestCostSummary:
    def _plan(self, asset="ETH", fee="0.005", minutes=15) -> TransferPlan:
        return TransferPlan(Venue.BINANCE, Venue.WALLET, asset, Decimal("4"), Decimal(fee), minutes)

    def test_empty_returns_zeros(self):
        planner = RebalancePlanner(_balanced())
        result = planner.cost_summary([])
        assert result["total_transfers"] == 0
        assert result["total_fees_usd"] == Decimal("0")

    def test_single_plan(self):
        planner = RebalancePlanner(_skewed())
        result = planner.cost_summary([self._plan()])
        assert result["total_transfers"] == 1
        assert result["total_fees_usd"] == Decimal("0.005")
        assert result["assets_affected"] == ["ETH"]

    def test_assets_deduplicated(self):
        planner = RebalancePlanner(_multi_asset())
        plans = [self._plan("ETH"), self._plan("ETH")]
        result = planner.cost_summary(plans)
        assert result["assets_affected"].count("ETH") == 1

    def test_max_time_not_sum(self):
        planner = RebalancePlanner(_multi_asset())
        plans = [self._plan("ETH", "0.005", 15), self._plan("BTC", "0.0005", 30)]
        result = planner.cost_summary(plans)
        assert result["total_time_min"] == 30
