"""
Cross-venue transfer planner.

Detects when an asset's distribution across venues drifts beyond a threshold
and generates the minimal set of transfers needed to restore balance.
Never executes transfers - planning only.

CLI::
    python3 -m src.inventory.rebalancer --check
    python3 -m src.inventory.rebalancer --plan ETH
    python3 -m src.inventory.rebalancer --plan-all
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from decimal import Decimal
from src.inventory.tracker import VenueTracker, Venue

# Reference fee and timing data for common assets
WITHDRAWAL_FEES: dict[str, dict] = {
    "ETH": {
        "fee": Decimal("0.005"),
        "min_amount": Decimal("0.01"),
        "confirmations": 12,
        "eta_minutes": 15
    },
    "USDT": {
        "fee": Decimal("1.0"),
        "min_amount": Decimal("10.0"),
        "confirmations": 12,
        "eta_minutes": 15
    },
    "USDC": {
        "fee": Decimal("1.0"),
        "min_amount": Decimal("10.0"),
        "confirmations": 12,
        "eta_minutes": 15
    },
    "BTC": {
        "fee": Decimal("0.0005"),
        "min_amount": Decimal("0.001"),
        "confirmations": 3,
        "eta_minutes": 30
    },
}

# Minimum balance to keep at each venue to remain operational
MINIMUM_VENUE_BALANCE: dict[str, Decimal] = {
    "ETH": Decimal("0.5"),
    "USDT": Decimal("500"),
    "USDC": Decimal("500"),
    "BTC": Decimal("0.01")
}


@dataclass
class TransferPlan:
    """One recommended cross-venue transfer."""
    source: "Venue"
    destination: "Venue"
    asset: str
    amount: Decimal
    estimated_fee: Decimal
    estimated_minutes: int

    @property
    def net_received(self) -> Decimal:
        """Amount the destination venue will actually receive after fees."""
        return self.amount - self.estimated_fee


class RebalancePlanner:
    """
    Generates transfer plans to restore balanced cross-venue distribution.

    Parameters
    ----------
    tracker        : VenueTracker with current balance snapshots.
    threshold_pct  : maximum allowed deviation from equal split (default 30%).
    target_weights : optional {Venue: fraction} override for target distribution.
    """

    def __init__(
        self,
        tracker: "VenueTracker",
        threshold_pct: float = 30.0,
        target_weights: dict | None = None
    ) -> None:
        self._tracker = tracker
        self._threshold = threshold_pct
        self._weights = target_weights

    def check_all(self) -> list[dict]:
        """Return skew reports for all tracked assets."""
        return self._tracker.all_skews()

    def plan(self, asset: str) -> list[TransferPlan]:
        """
        Generate the minimum set of transfers to rebalance *asset*.

        Returns an empty list if the asset is within the threshold.
        Respects MINIMUM_VENUE_BALANCE so no venue goes below safe operating level.
        """
        skew = self._tracker.skew_report(asset)
        if skew["max_deviation_pct"] <= self._threshold:
            return []

        total = skew["total"]
        if total == 0:
            return []

        venues_data = skew["venues"]
        n = len(venues_data)
        if n < 2:
            return []

        def _lookup_venue(name: str) -> "Venue":
            for v in Venue:
                if v.value == name:
                    return v
            raise ValueError(f"Unknown venue name: {name!r}")

        if self._weights is not None:
            targets: dict[str, Decimal] = {
                (v.value if hasattr(v, "value") else str(v)): Decimal(str(frac))
                for v, frac in self._weights.items()
            }
        else:
            equal = Decimal("1") / Decimal(str(n))
            targets = {vname: equal for vname in venues_data}

        surplus: list[list] = []
        deficit: list[list] = []

        for vname, data in venues_data.items():
            target_frac = targets.get(vname, Decimal("1") / Decimal(str(n)))
            target_qty = total * target_frac
            delta = data["amount"] - target_qty
            if delta > 0:
                surplus.append([vname, delta])
            elif delta < 0:
                deficit.append([vname, abs(delta)])

        fee_info = WITHDRAWAL_FEES.get(asset, {})
        withdrawal_fee = fee_info.get("fee", Decimal("0"))
        min_withdrawal = fee_info.get("min_amount", Decimal("0"))
        eta = fee_info.get("eta_minutes", 0)
        min_operating = MINIMUM_VENUE_BALANCE.get(asset, Decimal("0"))

        plans: list[TransferPlan] = []
        si = di = 0

        while si < len(surplus) and di < len(deficit):
            src_name, src_avail = surplus[si]
            dst_name, dst_need = deficit[di]

            src_venue = _lookup_venue(src_name)
            src_bal = self._tracker._balances.get(src_venue, {}).get(asset)
            src_total = src_bal.total if src_bal is not None else Decimal("0")
            max_transferable = max(Decimal("0"), src_total - min_operating)

            transfer_qty = min(src_avail, dst_need, max_transferable)

            if transfer_qty >= min_withdrawal and transfer_qty > withdrawal_fee:
                plans.append(TransferPlan(
                    source=src_venue,
                    destination=_lookup_venue(dst_name),
                    asset=asset,
                    amount=transfer_qty,
                    estimated_fee=withdrawal_fee,
                    estimated_minutes=eta
                ))

            surplus[si][1] -= transfer_qty
            deficit[di][1] -= transfer_qty

            if surplus[si][1] <= Decimal("0"):
                si += 1
            if deficit[di][1] <= Decimal("0"):
                di += 1

        return plans

    def plan_all(self) -> dict[str, list[TransferPlan]]:
        """Plan transfers for every asset that needs rebalancing."""
        result: dict[str, list[TransferPlan]] = {}
        for skew in self.check_all():
            if skew["needs_rebalance"]:
                asset_plans = self.plan(skew["asset"])
                if asset_plans:
                    result[skew["asset"]] = asset_plans
        return result

    def cost_summary(self, plans: list[TransferPlan]) -> dict:
        """
        Summarize fees and logistics for a list of transfer plans.

        Returns total_transfers, total_fees_usd, total_time_min, assets_affected.
        """
        if not plans:
            return {
                "total_transfers": 0,
                "total_fees_usd": Decimal("0"),
                "total_time_min": 0,
                "assets_affected": []
            }

        total_fees = sum((p.estimated_fee for p in plans), Decimal("0"))
        max_time = max(p.estimated_minutes for p in plans)
        affected = sorted({p.asset for p in plans})

        return {
            "total_transfers": len(plans),
            "total_fees_usd": total_fees,
            "total_time_min": max_time,
            "assets_affected": affected
        }


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cross-venue rebalance planner",
        prog="python3 -m src.inventory.rebalancer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="Show skew for all assets")
    group.add_argument("--plan", metavar="ASSET", help="Generate plan for ASSET")
    group.add_argument("--plan-all", action="store_true", help="Plan all unbalanced assets")
    args = parser.parse_args(argv)

    tracker = VenueTracker([Venue.BINANCE, Venue.WALLET])
    tracker.update_from_cex(
        Venue.BINANCE,
        {
            "ETH": {"free": "300.0", "locked": "0"},
            "USDT": {"free": "90.0", "locked": "0"}
        },
    )
    tracker.update_from_wallet(Venue.WALLET, {"ETH": "9000.0", "USDT": "300.0"})

    planner = RebalancePlanner(tracker)

    if args.check:
        skews = planner.check_all()
        print(f"{'Asset':<8}  {'Total':>12}  {'Max Deviation (%)':>9}  {'Rebalance':>10}")
        print("-" * 46)
        for s in skews:
            flag = "YES" if s["needs_rebalance"] else "NO"
            print(f"{s['asset']:<8}  {float(s['total']):>12.3f}  {s['max_deviation_pct']:>9.3f}  {flag:>10}")
        return 0

    if args.plan:
        asset = args.plan.upper()
        plans = planner.plan(asset)
        if not plans:
            print(f"No rebalance needed for {asset}.")
            return 0
        print(f"Transfer plans for {asset}:")
        for i, p in enumerate(plans, 1):
            print(
                f"  [{i}] {p.source.value} → {p.destination.value}: "
                f"{p.amount} {p.asset}  (fee={p.estimated_fee}, "
                f"net={p.net_received}, ~{p.estimated_minutes}min)"
            )
        cost = planner.cost_summary(plans)
        print(f"\nTotal fees: {cost['total_fees_usd']}  |  Est. time: {cost['total_time_min']} min")
        return 0

    if args.plan_all:
        all_plans = planner.plan_all()
        if not all_plans:
            print("All assets are within threshold - no transfers needed.")
            return 0
        for asset, plans in all_plans.items():
            print(f"\n{asset}:")
            for p in plans:
                print(
                    f"  {p.source.value} → {p.destination.value}: "
                    f"{p.amount} (fee={p.estimated_fee}, ~{p.estimated_minutes} min)"
                )
        flat = [p for ps in all_plans.values() for p in ps]
        cost = planner.cost_summary(flat)
        print(
            f"\nTotal transfers: {cost['total_transfers']}  |  "
            f"Total fees: {cost['total_fees_usd']}"
        )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
