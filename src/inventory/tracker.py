"""
Multi-venue position and balance tracking.

Two complementary trackers serve different purposes:

FillLedger       - cost-basis P&L accounting using weighted-average method.
                   Records individual fills, tracks realized PnL, fees.

VenueTracker     - real-time view of where funds physically sit.
                   Supports CEX snapshots and on-chain wallet balances,
                   pre-flight arb validation, and skew detection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

Side = Literal["buy", "sell"]


# ---------------------------------------------------------------------------
# Venue enum
# ---------------------------------------------------------------------------

class Venue(str, Enum):
    BINANCE = "binance"
    WALLET = "wallet"


# ---------------------------------------------------------------------------
# FillLedger — cost-basis / realized PnL tracking
# ---------------------------------------------------------------------------

@dataclass
class Fill:
    """One recorded execution against a position."""
    asset: str
    side: Side
    qty: Decimal
    price: Decimal
    fee: Decimal
    recorded_at: int   # Unix milliseconds


@dataclass
class Position:
    """Running position state for one asset."""
    asset: str
    qty: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_cost: Decimal = field(default_factory=lambda: Decimal("0"))
    realized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))


class FillLedger:
    """
    Weighted-average cost-basis position tracker.

    All amounts are Decimal; float inputs are rejected.
    """

    def __init__(self) -> None:
        self._positions: dict[str, Position] = {}
        self._fills: list[Fill] = []

    def record(
        self,
        asset: str,
        side: Side,
        qty: Decimal,
        price: Decimal,
        fee: Decimal = Decimal("0"),
        ts: int | None = None
    ) -> None:
        """Record a fill; update running position and realized PnL."""
        if qty <= 0:
            raise ValueError(f"qty must be positive, got {qty}")
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        if fee < 0:
            raise ValueError(f"fee must be non-negative, got {fee}")
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        when = ts if ts is not None else int(time.time() * 1000)
        self._fills.append(Fill(asset=asset, side=side, qty=qty, price=price, fee=fee, recorded_at=when))

        pos = self._positions.setdefault(asset, Position(asset=asset))
        pos.total_fees += fee

        if side == "buy":
            total_value = pos.avg_cost * pos.qty + price * qty
            pos.qty += qty
            pos.avg_cost = total_value / pos.qty
        else:
            sell_qty = min(qty, pos.qty)
            if sell_qty > 0:
                pos.realized_pnl += (price - pos.avg_cost) * sell_qty - fee
                pos.qty -= sell_qty
                if pos.qty == 0:
                    pos.avg_cost = Decimal("0")

    def position(self, asset: str) -> Position:
        return self._positions.get(asset, Position(asset=asset))

    def open_positions(self) -> dict[str, Position]:
        return {a: p for a, p in self._positions.items() if p.qty > 0}

    def all_positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def unrealized_pnl(self, asset: str, mark: Decimal) -> Decimal:
        pos = self.position(asset)
        if pos.qty == 0:
            return Decimal("0")
        return (mark - pos.avg_cost) * pos.qty

    def fills_for(self, asset: str | None = None) -> list[Fill]:
        if asset is None:
            return list(self._fills)
        return [f for f in self._fills if f.asset == asset]


# ---------------------------------------------------------------------------
# VenueTracker - real-time multi-venue balance sheet
# ---------------------------------------------------------------------------

@dataclass
class AssetBalance:
    """Balance of one asset at one venue."""
    venue: Venue
    asset: str
    free: Decimal
    locked: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def total(self) -> Decimal:
        return self.free + self.locked


class VenueTracker:
    """
    Tracks where funds actually sit across CEX and on-chain venues.

    Balances are snapshots - calling update_* replaces the prior snapshot
    for that venue entirely.
    """

    def __init__(self, venues: list[Venue]) -> None:
        self._venues = list(venues)
        self._balances: dict[Venue, dict[str, AssetBalance]] = {v: {} for v in venues}

    def update_from_cex(self, venue: Venue, raw: dict) -> None:
        """
        Ingest balances from BinanceClient.fetch_balance() output.

        Args:
            venue: which CEX this data came from.
            raw:   {asset: {free, locked, total}} mapping.
        """
        snapshot: dict[str, AssetBalance] = {}
        for asset, info in raw.items():
            if not isinstance(info, dict):
                continue
            snapshot[asset] = AssetBalance(
                venue=venue,
                asset=asset,
                free=Decimal(str(info.get("free", 0))),
                locked=Decimal(str(info.get("locked", 0)))
            )
        self._balances[venue] = snapshot

    def update_from_wallet(self, venue: Venue, raw: dict) -> None:
        """
        Ingest on-chain wallet balances.

        Args:
            venue: wallet venue identifier.
            raw:   {asset: amount} mapping (all funds treated as free).
        """
        snapshot: dict[str, AssetBalance] = {}
        for asset, amount in raw.items():
            snapshot[asset] = AssetBalance(
                venue=venue,
                asset=asset,
                free=Decimal(str(amount))
            )
        self._balances[venue] = snapshot

    def snapshot(self) -> dict:
        """Return a full cross-venue portfolio snapshot."""
        venues_data: dict[str, dict] = {}
        totals: dict[str, Decimal] = {}

        for venue, assets in self._balances.items():
            vk = venue.value
            venues_data[vk] = {}
            for asset, bal in assets.items():
                venues_data[vk][asset] = {
                    "free": bal.free,
                    "locked": bal.locked,
                    "total": bal.total
                }
                totals[asset] = totals.get(asset, Decimal("0")) + bal.total

        return {
            "timestamp": datetime.now(tz=UTC),
            "venues": venues_data,
            "totals": totals
        }

    def available(self, venue: Venue, asset: str) -> Decimal:
        """Free (tradeable) balance of *asset* at *venue*."""
        bal = self._balances.get(venue, {}).get(asset)
        return bal.free if bal is not None else Decimal("0")

    def can_execute(
        self,
        buy_venue: Venue,
        buy_asset: str,
        buy_amount: Decimal,
        sell_venue: Venue,
        sell_asset: str,
        sell_amount: Decimal
    ) -> dict:
        """
        Pre-flight check: verify sufficient balances for both legs of an arb.

        Returns dict with can_execute bool, available/needed amounts, and reason.
        """
        buy_avail = self.available(buy_venue, buy_asset)
        sell_avail = self.available(sell_venue, sell_asset)

        problems: list[str] = []
        if buy_avail < buy_amount:
            problems.append(
                f"Insufficient {buy_asset} at {buy_venue.value}: "
                f"need {buy_amount}, have {buy_avail}"
            )
        if sell_avail < sell_amount:
            problems.append(
                f"Insufficient {sell_asset} at {sell_venue.value}: "
                f"need {sell_amount}, have {sell_avail}"
            )

        return {
            "can_execute": not problems,
            "buy_venue_available": buy_avail,
            "buy_venue_needed": buy_amount,
            "sell_venue_available": sell_avail,
            "sell_venue_needed": sell_amount,
            "reason": "; ".join(problems) if problems else None
        }

    def apply_trade(
        self,
        venue: Venue,
        side: Side,
        base_asset: str,
        quote_asset: str,
        base_qty: Decimal,
        quote_qty: Decimal,
        fee: Decimal,
        fee_asset: str
    ) -> None:
        """Update internal balances to reflect a completed trade."""
        if venue not in self._balances:
            self._balances[venue] = {}

        def _ensure(asset: str) -> AssetBalance:
            if asset not in self._balances[venue]:
                self._balances[venue][asset] = AssetBalance(
                    venue=venue, asset=asset, free=Decimal("0")
                )
            return self._balances[venue][asset]

        base_bal = _ensure(base_asset)
        quote_bal = _ensure(quote_asset)

        if side == "buy":
            base_bal.free += base_qty
            quote_bal.free -= quote_qty
        else:
            base_bal.free -= base_qty
            quote_bal.free += quote_qty

        _ensure(fee_asset).free -= fee

    def skew_report(self, asset: str) -> dict:
        """
        Compute cross-venue distribution skew for *asset*.

        Returns total, per-venue breakdown (amount, pct, deviation),
        max_deviation_pct, and a needs_rebalance flag (true if > 30%).
        """
        total = Decimal("0")
        per_venue: dict[str, Decimal] = {}

        for v in self._venues:
            bal = self._balances.get(v, {}).get(asset)
            amt = bal.total if bal is not None else Decimal("0")
            per_venue[v.value] = amt
            total += amt

        n = len(self._venues)
        equal_pct = 100.0 / n if n > 0 else 0.0
        max_dev = 0.0
        venues_detail: dict[str, dict] = {}

        for vname, amt in per_venue.items():
            pct = float(amt / total * 100) if total > 0 else 0.0
            dev = abs(pct - equal_pct)
            max_dev = max(max_dev, dev)
            venues_detail[vname] = {"amount": amt, "pct": pct, "deviation_pct": dev}

        return {
            "asset": asset,
            "total": total,
            "venues": venues_detail,
            "max_deviation_pct": max_dev,
            "needs_rebalance": max_dev > 30.0
        }

    def all_skews(self) -> list[dict]:
        """Return skew reports for every asset that appears across any venue."""
        all_assets: set[str] = set()
        for assets in self._balances.values():
            all_assets.update(assets.keys())
        return [self.skew_report(a) for a in sorted(all_assets)]
