"""
Order book depth analytics.

Wraps the dict produced by BinanceClient.fetch_order_book() and exposes
execution-quality metrics used by the arb checker:

  simulate_fill   - VWAP simulation for a given quantity
  liquidity_band  - available quantity within a basis-point band
  pressure_ratio  - bid/ask imbalance in [-1, +1]
  round_trip_cost - effective spread for a complete round-trip trade

CLI usage::
    python -m src.exchange.orderbook ETH/USDT
    python -m src.exchange.orderbook BTC/USDT --qty 0.5 --depth 20
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from dotenv import load_dotenv


class OrderBookAnalyzer:
    """Stateless analyzer that operates on a single order book snapshot."""

    def __init__(self, snapshot: dict) -> None:
        self._bids: list[tuple[Decimal, Decimal]] = snapshot.get("bids", [])
        self._asks: list[tuple[Decimal, Decimal]] = snapshot.get("asks", [])
        self._symbol: str = snapshot.get("symbol", "")
        self._ts: int = snapshot.get("timestamp") or 0
        self._mid: Decimal = snapshot.get("mid_price", Decimal("0"))

        raw_bb = snapshot.get("best_bid") or (Decimal("0"), Decimal("0"))
        raw_ba = snapshot.get("best_ask") or (Decimal("0"), Decimal("0"))
        self._best_bid: tuple[Decimal, Decimal] = raw_bb
        self._best_ask: tuple[Decimal, Decimal] = raw_ba

    # ------------------------------------------------------------------
    # Core analytics
    # ------------------------------------------------------------------

    def simulate_fill(self, direction: str, quantity: float) -> dict:
        """
        Walk the order book and compute the average fill price for *quantity*.

        Args:
            direction: "buy" (consume asks) or "sell" (consume bids).
            quantity:  base-asset quantity to fill.

        Returns dict with keys: avg_price, total_cost, slippage_bps,
        levels_consumed, fully_filled, fills.
        """
        if direction not in ("buy", "sell"):
            raise ValueError(f"direction must be 'buy' or 'sell', got {direction!r}")
        qty = Decimal(str(quantity))
        if qty <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")

        ladder = self._asks if direction == "buy" else self._bids
        reference = ladder[0][0] if ladder else Decimal("0")

        remaining = qty
        cost = Decimal("0")
        fills: list[dict] = []

        for px, avail in ladder:
            if remaining <= 0:
                break
            take = min(remaining, avail)
            leg_cost = px * take
            fills.append({"price": px, "qty": take, "cost": leg_cost})
            cost += leg_cost
            remaining -= take

        filled = qty - remaining
        avg_px = cost / filled if filled > 0 else Decimal("0")
        fully_filled = remaining <= 0

        if reference > 0 and filled > 0:
            if direction == "buy":
                slip = (avg_px - reference) / reference * Decimal("10000")
            else:
                slip = (reference - avg_px) / reference * Decimal("10000")
            slip = max(slip, Decimal("0"))
        else:
            slip = Decimal("0")

        return {
            "avg_price": avg_px,
            "total_cost": cost,
            "slippage_bps": slip,
            "levels_consumed": len(fills),
            "fully_filled": fully_filled,
            "fills": fills
        }

    def liquidity_band(self, side: str, band_bps: float) -> Decimal:
        """
        Sum of available quantity within *band_bps* basis points of the best price.

        Args:
            side:     "bid" or "ask".
            band_bps: Width of the price band in basis points.
        """
        if side not in ("bid", "ask"):
            raise ValueError(f"side must be 'bid' or 'ask', got {side!r}")

        factor = Decimal(str(band_bps)) / Decimal("10000")

        if side == "bid":
            if not self._bids:
                return Decimal("0")
            best = self._bids[0][0]
            cutoff = best * (1 - factor)
            return sum((q for p, q in self._bids if p >= cutoff), Decimal("0"))
        else:
            if not self._asks:
                return Decimal("0")
            best = self._asks[0][0]
            cutoff = best * (1 + factor)
            return sum((q for p, q in self._asks if p <= cutoff), Decimal("0"))

    def pressure_ratio(self, depth: int = 10) -> float:
        """
        Bid/ask volume imbalance in the range [-1.0, +1.0].
        Positive values indicate buy pressure; negative indicate sell pressure.
        """
        bid_vol = float(sum(q for _, q in self._bids[:depth]))
        ask_vol = float(sum(q for _, q in self._asks[:depth]))
        total = bid_vol + ask_vol
        if total == 0.0:
            return 0.0
        return (bid_vol - ask_vol) / total

    def round_trip_cost(self, quantity: float) -> Decimal:
        """
        Effective spread for a complete round-trip (buy then sell *quantity*).
        Returned in basis points of the mid price.
        """
        buy_result = self.simulate_fill("buy", quantity)
        sell_result = self.simulate_fill("sell", quantity)

        ask_avg = buy_result["avg_price"]
        bid_avg = sell_result["avg_price"]

        if self._mid == 0 or ask_avg == 0 or bid_avg == 0:
            return Decimal("0")

        return (ask_avg - bid_avg) / self._mid * Decimal("10000")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def timestamp(self) -> int:
        return self._ts

    @property
    def mid_price(self) -> Decimal:
        return self._mid

    @property
    def best_bid(self) -> tuple[Decimal, Decimal]:
        return self._best_bid

    @property
    def best_ask(self) -> tuple[Decimal, Decimal]:
        return self._best_ask

    @property
    def quoted_spread_bps(self) -> Decimal:
        """Best-level spread expressed in basis points."""
        bp, ap = self._best_bid[0], self._best_ask[0]
        if self._mid == 0 or bp == 0 or ap == 0:
            return Decimal("0")
        return (ap - bp) / self._mid * Decimal("10000")


# ---------------------------------------------------------------------------
# CLI renderer
# ---------------------------------------------------------------------------

def _fmt_price(p: Decimal) -> str:
    return f"${float(p):,.2f}"


def _fmt_qty(q: Decimal, sym: str = "") -> str:
    base = sym.split("/")[0] if "/" in sym else ""
    suffix = f" {base}" if base else ""
    return f"{float(q):.4f}{suffix}"


def _fmt_bps(b: Decimal) -> str:
    return f"{float(b):.2f} bps"


def _print_snapshot(analyser: OrderBookAnalyzer, qty: float) -> None:
    W = 54
    sep = "╠" + "═" * (W - 2) + "╣"
    top = "╔" + "═" * (W - 2) + "╗"
    bot = "╚" + "═" * (W - 2) + "╝"
    sym = analyser.symbol

    def row(text: str) -> str:
        return f"║  {text:<{W - 4}}║"

    ts_str = "N/A"
    if analyser.timestamp:
        ts_str = datetime.fromtimestamp(analyser.timestamp / 1000, tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

    bp, bq = analyser.best_bid
    ap, aq = analyser.best_ask
    spread_abs = ap - bp
    imbal = analyser.pressure_ratio()
    imbal_sign = "+" if imbal >= 0 else ""
    imbal_label = "buy pressure" if imbal > 0.05 else ("sell pressure" if imbal < -0.05 else "balanced")

    bd = analyser.liquidity_band("bid", 10)
    ad = analyser.liquidity_band("ask", 10)

    lines = [
        top,
        row(f"{sym} Order Book Analysis"),
        row(f"Timestamp: {ts_str}"),
        sep,
        row(f"Best Bid:    {_fmt_price(bp)} × {_fmt_qty(bq, sym)}"),
        row(f"Best Ask:    {_fmt_price(ap)} × {_fmt_qty(aq, sym)}"),
        row(f"Mid Price:   {_fmt_price(analyser.mid_price)}"),
        row(f"Spread:      {_fmt_price(spread_abs)} ({_fmt_bps(analyser.quoted_spread_bps)})"),
        sep,
        row("Depth (within 10 bps):"),
        row(f"  Bids: {_fmt_qty(bd, sym)} ({_fmt_price(bd * bp)})"),
        row(f"  Asks: {_fmt_qty(ad, sym)} ({_fmt_price(ad * ap)})"),
        row(f"Imbalance: {imbal_sign}{imbal:.2f} ({imbal_label})"),
        sep
    ]

    base = sym.split("/")[0] if "/" in sym else ""
    for label, q_val in [(qty, qty), (qty * 5, qty * 5)]:
        lines.append(row(f"Walk-the-book ({label} {base} buy):"))
        result = analyser.simulate_fill("buy", q_val)
        if result["fully_filled"]:
            lines.append(row(f"  Avg price:  {_fmt_price(result['avg_price'])}"))
            lines.append(row(f"  Slippage:   {_fmt_bps(result['slippage_bps'])}"))
            lines.append(row(f"  Levels:     {result['levels_consumed']}"))
        else:
            lines.append(row("  INSUFFICIENT LIQUIDITY"))
        lines.append(sep)

    eff = analyser.round_trip_cost(qty)
    lines.append(row(f"Effective spread ({qty} {base} round-trip): {_fmt_bps(eff)}"))
    lines.append(bot)
    print("\n".join(lines))


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live order book snapshot (Binance testnet)",
        prog="python3 -m src.exchange.orderbook"
    )
    parser.add_argument("symbol", help="e.g. ETH/USDT")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--qty", type=float, default=2.0)
    args = parser.parse_args(argv)

    load_dotenv()

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    try:
        from src.exchange.client import BinanceClient
        client = BinanceClient({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": True,
            "enableRateLimit": True
        })
        snapshot = client.fetch_order_book(args.symbol, limit=args.depth)
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    _print_snapshot(OrderBookAnalyzer(snapshot), qty=args.qty)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
