"""
Arbitrage trade ledger and P&L reporting.

Two data classes represent the two legs of an arb trade:
  TradeLeg  - one execution (buy on DEX or sell on CEX, for example).
  ArbTrade  - a complete arb round-trip (buy leg + sell leg + gas).

PnLTracker accumulates ArbTrade records and produces summary statistics,
per-trade detail views, and CSV exports.

CLI::
    python3 -m src.inventory.pnl --summary
    python3 -m src.inventory.pnl --recent 5
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from src.inventory.tracker import Venue


@dataclass
class TradeLeg:
    """One execution leg within an arbitrage trade."""
    leg_id: str
    executed_at: datetime
    venue: "Venue"
    symbol: str
    side: str   # "buy" or "sell"
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str


@dataclass
class ArbTrade:
    """
    Complete arbitrage round-trip: one buy leg and one sell leg.

    Gross PnL  = sell_revenue − buy_cost
    Net PnL    = gross − all fees − gas
    """
    trade_id: str
    opened_at: datetime
    buy_leg: TradeLeg
    sell_leg: TradeLeg
    gas_usd: Decimal = field(default_factory=lambda: Decimal("0"))

    @property
    def notional(self) -> Decimal:
        """Buy-side notional value."""
        return self.buy_leg.price * self.buy_leg.quantity

    @property
    def gross_pnl(self) -> Decimal:
        """Revenue difference before fees."""
        return self.sell_leg.price * self.sell_leg.quantity - self.buy_leg.price * self.buy_leg.quantity

    @property
    def all_fees(self) -> Decimal:
        """Sum of both legs' fees plus gas."""
        return self.buy_leg.fee + self.sell_leg.fee + self.gas_usd

    @property
    def net_pnl(self) -> Decimal:
        """Gross P&L minus all costs."""
        return self.gross_pnl - self.all_fees

    @property
    def net_pnl_bps(self) -> Decimal:
        """Net P&L expressed in basis points of notional."""
        if self.notional == 0:
            return Decimal("0")
        return self.net_pnl / self.notional * Decimal("10000")


class PnLTracker:
    """
    Accumulates ArbTrade records and produces aggregate reports.

    Usage::

        tracker = PnLTracker()
        tracker.record(arb_trade)
        print(tracker.summary())
    """

    def __init__(self) -> None:
        self.trades: list[ArbTrade] = []

    def record(self, trade: ArbTrade) -> None:
        """Append a completed arb trade to the ledger."""
        self.trades.append(trade)

    def summary(self) -> dict:
        """
        Return aggregate statistics across all recorded trades.

        Keys: total_trades, total_pnl_usd, total_fees_usd, avg_pnl_per_trade,
              avg_pnl_bps, win_rate, best_trade_pnl, worst_trade_pnl,
              total_notional, sharpe_estimate, pnl_by_hour.
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "total_pnl_usd": Decimal("0"),
                "total_fees_usd": Decimal("0"),
                "avg_pnl_per_trade": Decimal("0"),
                "avg_pnl_bps": Decimal("0"),
                "win_rate": 0.0,
                "best_trade_pnl": Decimal("0"),
                "worst_trade_pnl": Decimal("0"),
                "total_notional": Decimal("0"),
                "sharpe_estimate": 0.0,
                "pnl_by_hour": {}
            }

        pnls = [t.net_pnl for t in self.trades]
        n = len(self.trades)

        total_pnl = sum(pnls, Decimal("0"))
        total_fees = sum((t.all_fees for t in self.trades), Decimal("0"))
        total_notional = sum((t.notional for t in self.trades), Decimal("0"))
        winners = sum(1 for p in pnls if p > 0)
        win_rate = winners / n * 100.0
        avg_pnl = total_pnl / n
        avg_bps = sum((t.net_pnl_bps for t in self.trades), Decimal("0")) / n

        if n >= 2:
            floats = [float(p) for p in pnls]
            std = statistics.stdev(floats)
            mean = sum(floats) / n
            sharpe = mean / std if std != 0.0 else 0.0
        else:
            sharpe = 0.0

        pnl_by_hour: dict[int, Decimal] = {}
        for t in self.trades:
            h = t.opened_at.hour
            pnl_by_hour[h] = pnl_by_hour.get(h, Decimal("0")) + t.net_pnl

        return {
            "total_trades": n,
            "total_pnl_usd": total_pnl,
            "total_fees_usd": total_fees,
            "avg_pnl_per_trade": avg_pnl,
            "avg_pnl_bps": avg_bps,
            "win_rate": win_rate,
            "best_trade_pnl": max(pnls),
            "worst_trade_pnl": min(pnls),
            "total_notional": total_notional,
            "sharpe_estimate": sharpe,
            "pnl_by_hour": pnl_by_hour
        }

    def recent(self, n: int = 10) -> list[dict]:
        """Return the last n trades as summary dicts, most recent first."""
        window = self.trades[-n:] if len(self.trades) > n else list(self.trades)
        result = []
        for t in reversed(window):
            result.append({
                "trade_id": t.trade_id,
                "opened_at": t.opened_at,
                "symbol": t.buy_leg.symbol,
                "buy_venue": t.buy_leg.venue,
                "sell_venue": t.sell_leg.venue,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "net_pnl_bps": t.net_pnl_bps,
                "all_fees": t.all_fees,
                "notional": t.notional
            })
        return result

    def export_csv(self, path: str) -> None:
        """Write all trades to a CSV file."""
        columns = [
            "trade_id", "opened_at", "symbol",
            "buy_venue", "sell_venue",
            "buy_price", "sell_price", "quantity",
            "gross_pnl", "all_fees", "net_pnl", "net_pnl_bps",
            "notional", "gas_usd"
        ]
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for t in self.trades:
                bv = t.buy_leg.venue.value if hasattr(t.buy_leg.venue, "value") else str(t.buy_leg.venue)
                sv = t.sell_leg.venue.value if hasattr(t.sell_leg.venue, "value") else str(t.sell_leg.venue)
                writer.writerow({
                    "trade_id": t.trade_id,
                    "opened_at": t.opened_at.isoformat(),
                    "symbol": t.buy_leg.symbol,
                    "buy_venue": bv,
                    "sell_venue": sv,
                    "buy_price": str(t.buy_leg.price),
                    "sell_price": str(t.sell_leg.price),
                    "quantity": str(t.buy_leg.quantity),
                    "gross_pnl": str(t.gross_pnl),
                    "all_fees": str(t.all_fees),
                    "net_pnl": str(t.net_pnl),
                    "net_pnl_bps": str(t.net_pnl_bps),
                    "notional": str(t.notional),
                    "gas_usd": str(t.gas_usd)
                })


# ---------------------------------------------------------------------------
# Demo data builder
# ---------------------------------------------------------------------------

def _build_demo_tracker() -> PnLTracker:
    from src.inventory.tracker import Venue

    tracker = PnLTracker()
    base_ts = datetime(2026, 3, 15, 3, 0, 0, tzinfo=UTC)

    scenarios = [
        (Decimal("2000"), Decimal("2001.25"), Decimal("1"), Decimal("0.40"), Decimal("0.40"), Decimal("0"), 9),
        (Decimal("2001"), Decimal("2001.90"), Decimal("1"), Decimal("0.40"), Decimal("0.40"), Decimal("0"), 10),
        (Decimal("2002"), Decimal("2001.40"), Decimal("1"), Decimal("0.40"), Decimal("0.40"), Decimal("0"), 13)
    ]

    for i, (bp, sp, qty, bf, sf, gas, mins) in enumerate(scenarios):
        ts = base_ts - timedelta(minutes=mins)
        buy_leg = TradeLeg(
            leg_id=f"buy-{i}", executed_at=ts, venue=Venue.WALLET,
            symbol="ETH/USDT", side="buy", quantity=qty,
            price=bp, fee=bf, fee_currency="USDT"
        )
        sell_leg = TradeLeg(
            leg_id=f"sell-{i}", executed_at=ts, venue=Venue.BINANCE,
            symbol="ETH/USDT", side="sell", quantity=qty,
            price=sp, fee=sf, fee_currency="USDT"
        )
        tracker.record(ArbTrade(
            trade_id=f"arb-{i}", opened_at=ts,
            buy_leg=buy_leg, sell_leg=sell_leg, gas_usd=gas
        ))

    return tracker


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arb trade P&L dashboard",
        prog="python3 -m src.inventory.pnl",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--summary", action="store_true", help="Aggregate summary")
    group.add_argument("--recent", metavar="N", type=int, nargs="?", const=10)
    args = parser.parse_args(argv)

    tracker = _build_demo_tracker()

    if args.summary:
        s = tracker.summary()
        w = 45
        print(f"\nPnL Summary (demo)\n{'═' * w}")
        print(f"Total Trades: {s['total_trades']}")
        print(f"Win Rate: {s['win_rate']:.1f}%")
        print(f"Total PnL: ${float(s['total_pnl_usd'])}")
        print(f"Total Fees: ${float(s['total_fees_usd'])}")
        print(f"Avg PnL/Trade: ${float(s['avg_pnl_per_trade']):.3f}")
        print(f"Avg PnL (bps): {float(s['avg_pnl_bps']):.3f} bps")
        print(f"Best Trade: ${float(s['best_trade_pnl'])}")
        print(f"Worst Trade: ${float(s['worst_trade_pnl'])}")
        print(f"Total Notional: ${float(s['total_notional'])}")
        print()

        for r in tracker.recent():
            ts = r["opened_at"].strftime("%H:%M")
            sym = r["symbol"].split("/")[0]
            bv = r["buy_venue"].value if hasattr(r["buy_venue"], "value") else str(r["buy_venue"])
            sv = r["sell_venue"].value if hasattr(r["sell_venue"], "value") else str(r["sell_venue"])
            pnl = float(r["net_pnl"])
            bps = float(r["net_pnl_bps"])
            sign = "+" if pnl >= 0 else ""
            print(f"  {ts}  {sym}  Buy {bv} / Sell {sv}  {sign}${pnl:.2f} ({bps:.1f} bps)")

        return 0

    if args.recent is not None:
        recent = tracker.recent(args.recent)
        print(f"{'Time':<6}  {'Symbol':<10}  {'Net PnL':>8}  {'bps':>6}  {'Notional':>12}")
        print("-" * 50)
        for r in recent:
            ts = r["opened_at"].strftime("%H:%M")
            sign = "+" if r["net_pnl"] >= 0 else ""
            print(
                f"{ts:<6}  {r['symbol']:<10}  "
                f"{sign}${float(r['net_pnl']):>6.2f}  "
                f"{float(r['net_pnl_bps']):>5.1f}  "
                f"${float(r['notional']):>10,.2f}"
            )
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
