"""
End-to-end arbitrage opportunity assessor.

Connects:
  - A DEX pricing source (any object with get_dex_quote())
  - The BinanceClient (CEX order book + fees)
  - The VenueTracker (pre-flight inventory check)
  - The PnLTracker (optional trade recording)

Returns a structured assessment dict - does NOT execute any trades.

CLI:
    python3 -m src.integration.arb_checker ETH/USDT --size 2.0
    python3 -m src.integration.arb_checker ETH/USDT --size 2.0 --dex-price 2003.19
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from decimal import Decimal
from dotenv import load_dotenv

from src.exchange.orderbook import OrderBookAnalyzer
from src.inventory.pnl import PnLTracker
from src.inventory.tracker import Venue, VenueTracker

# Gas estimate constants
_GAS_UNITS_ESTIMATE = 150_000
_DEFAULT_CEX_FEE_BPS = Decimal("10")
_DEFAULT_DEX_FEE_BPS = Decimal("30")


class ArbChecker:
    """
    Evaluates whether a DEX ↔ CEX arbitrage opportunity is worth pursuing.

    Does not execute trades. Returns an assessment dict for downstream
    decision-making.
    """

    def __init__(
        self,
        dex_source,   # Object with .get_dex_quote(base, quote, size) -> dict
        cex_client,   # BinanceClient instance
        inventory: "VenueTracker",
        pnl_tracker: "PnLTracker"
    ) -> None:
        self._dex = dex_source
        self._cex = cex_client
        self._inventory = inventory
        self._pnl = pnl_tracker

    def assess(
        self,
        pair: str,
        size: float = 1.0,
        gas_gwei: int = 20,
        eth_usd_override: Decimal | None = None
    ) -> dict:
        """
        Full arb assessment for *pair*.

        Returns a dict with: pair, timestamp, dex_price, cex_bid, cex_ask,
        gap_bps, direction, estimated_costs_bps, estimated_net_pnl_bps,
        inventory_ok, executable, details.
        """

        base, quote = pair.split("/")
        size_dec = Decimal(str(size))
        now = datetime.now(tz=UTC)

        # --- DEX side ---
        dex_data = self._dex.get_dex_quote(base, quote, size_dec)
        dex_price: Decimal = dex_data["price"]
        dex_impact_bps: Decimal = Decimal(str(dex_data.get("impact_bps", "0")))
        dex_fee_bps: Decimal = Decimal(str(dex_data.get("fee_bps", str(_DEFAULT_DEX_FEE_BPS))))

        # --- CEX side ---
        raw_book = self._cex.fetch_order_book(pair)
        book = OrderBookAnalyzer(raw_book)
        cex_bid = book.best_bid[0]
        cex_ask = book.best_ask[0]

        # --- Direction and gap ---
        direction: str | None
        gap_bps: Decimal
        cex_slip_bps: Decimal

        if cex_bid > 0 and dex_price < cex_bid:
            direction = "buy_dex_sell_cex"
            gap_bps = (cex_bid - dex_price) / dex_price * Decimal("10000")
            walk = book.simulate_fill("sell", size)
            cex_slip_bps = walk["slippage_bps"]
        elif 0 < cex_ask < dex_price:
            direction = "buy_cex_sell_dex"
            gap_bps = (dex_price - cex_ask) / cex_ask * Decimal("10000")
            walk = book.simulate_fill("buy", size)
            cex_slip_bps = walk["slippage_bps"]
        else:
            direction = None
            gap_bps = Decimal("0")
            cex_slip_bps = Decimal("0")

        # --- CEX fees ---
        try:
            fees = self._cex.get_trading_fees(pair)
            cex_fee_bps = Decimal(str(fees.get("taker", "0.001"))) * Decimal("10000")
        except Exception:
            cex_fee_bps = _DEFAULT_CEX_FEE_BPS

        # --- Gas cost ---
        gas_eth = Decimal(_GAS_UNITS_ESTIMATE) * Decimal(gas_gwei) / Decimal("1_000_000_000")
        eth_usd = eth_usd_override if eth_usd_override is not None else dex_price
        gas_usd = gas_eth * eth_usd
        notional = dex_price * size_dec
        gas_bps = gas_usd / notional * Decimal("10000") if notional > 0 else Decimal("0")

        # --- Total costs and net PnL ---
        total_costs_bps = dex_fee_bps + dex_impact_bps + cex_fee_bps + cex_slip_bps + gas_bps
        net_pnl_bps = gap_bps - total_costs_bps

        # --- Inventory check ---
        if direction == "buy_dex_sell_cex":
            inv = self._inventory.can_execute(
                buy_venue=Venue.WALLET, buy_asset=quote, buy_amount=notional,
                sell_venue=Venue.BINANCE, sell_asset=base, sell_amount=size_dec
            )
        elif direction == "buy_cex_sell_dex":
            inv = self._inventory.can_execute(
                buy_venue=Venue.BINANCE, buy_asset=quote, buy_amount=notional,
                sell_venue=Venue.WALLET, sell_asset=base, sell_amount=size_dec
            )
        else:
            inv = {
                "can_execute": True,
                "buy_venue_available": Decimal("0"),
                "buy_venue_needed": Decimal("0"),
                "sell_venue_available": Decimal("0"),
                "sell_venue_needed": Decimal("0"),
                "reason": None
            }

        inventory_ok = inv["can_execute"]
        executable = direction is not None and net_pnl_bps > 0 and inventory_ok

        return {
            "pair": pair,
            "timestamp": now,
            "dex_price": dex_price,
            "cex_bid": cex_bid,
            "cex_ask": cex_ask,
            "gap_bps": gap_bps,
            "direction": direction,
            "estimated_costs_bps": total_costs_bps,
            "estimated_net_pnl_bps": net_pnl_bps,
            "inventory_ok": inventory_ok,
            "executable": executable,
            "details": {
                "dex_price_impact_bps": dex_impact_bps,
                "cex_slippage_bps": cex_slip_bps,
                "cex_fee_bps": cex_fee_bps,
                "dex_fee_bps": dex_fee_bps,
                "gas_cost_usd": gas_usd
            }
        }


class StaticDexSource:
    """
    DEX price shim that returns a fixed price for testing.
    Satisfies the interface expected by ArbChecker.
    """

    def __init__(
        self,
        price: Decimal | None = None,
        impact_bps: Decimal = Decimal("0"),
        fee_bps: Decimal = _DEFAULT_DEX_FEE_BPS,
        price_fn=None
    ) -> None:
        self._price = price
        self._impact = impact_bps
        self._fee = fee_bps
        self._fn = price_fn

    def get_dex_quote(self, base: str, quote: str, size: Decimal) -> dict:
        if self._fn is not None:
            return self._fn(base, quote, size)
        return {
            "price": self._price,
            "impact_bps": self._impact,
            "fee_bps": self._fee
        }


def _print_assessment(result: dict, size: float) -> None:
    pair = result["pair"]
    w = 43
    line = "═" * w

    dex_p = result["dex_price"]
    cex_bid = result["cex_bid"]
    cex_ask = result["cex_ask"]
    gap = result["gap_bps"]
    direction = result["direction"]
    d = result["details"]
    costs = result["estimated_costs_bps"]
    net = result["estimated_net_pnl_bps"]

    print(f"\n{line}")
    print(f"  ARB CHECK: {pair} (size: {size} {pair.split('/')[0]})")
    print(line)

    print("\nPrices:")
    print(f"  DEX (execution):  ${float(dex_p):>10,.2f}")
    print(f"  CEX best bid:     ${float(cex_bid):>10,.2f}")
    print(f"  CEX best ask:     ${float(cex_ask):>10,.2f}")

    if direction:
        print(f"\nGap: {float(gap):.1f} bps  [{direction.replace('_', ' ')}]")
    else:
        print("\nGap: 0.0 bps  [no opportunity]")

    print("\nCosts:")
    print(f"  DEX fee:           {float(d['dex_fee_bps']):>6.1f} bps")
    print(f"  DEX price impact:  {float(d['dex_price_impact_bps']):>6.1f} bps")
    print(f"  CEX fee:           {float(d['cex_fee_bps']):>6.1f} bps")
    print(f"  CEX slippage:      {float(d['cex_slippage_bps']):>6.1f} bps")
    print(f"  Gas:               ${float(d['gas_cost_usd']):.2f}")
    print(f"  {'─' * 30}")
    print(f"  Total costs:       {float(costs):>6.1f} bps")

    sign = "+" if net >= 0 else ""
    flag = "PROFITABLE" if net > 0 else "NOT PROFITABLE"
    print(f"\nNet PnL estimate: {sign}{float(net):.1f} bps  {flag}")

    print("\nInventory:")
    check = "DONE" if result["inventory_ok"] else "INSUFFICIENT"
    print(f"  Pre-flight check: {check}")

    if result["executable"]:
        verdict = "EXECUTE"
    elif not direction:
        verdict = "SKIP - no opportunity"
    elif not (net > 0):
        verdict = "SKIP - costs exceed gap"
    else:
        verdict = "SKIP - insufficient inventory"

    print(f"\nVerdict: {verdict}")
    print(line + "\n")


def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Arb checker - Binance testnet CEX and simulated DEX",
        prog="python3 -m src.integration.arb_checker"
    )
    parser.add_argument("pair", help="Trading pair, e.g. ETH/USDT")
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--dex-price", type=float, default=None)
    parser.add_argument("--gas-gwei", type=int, default=20)
    parser.add_argument("--depth", type=int, default=20)
    args = parser.parse_args(argv)

    load_dotenv()

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    try:
        from src.exchange.client import BinanceClient
        cex_client = BinanceClient({
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": True,
            "enableRateLimit": True,
        })
    except Exception as exc:
        print(f"Error connecting to exchange: {exc}", file=sys.stderr)
        return 1

    try:
        raw = cex_client.fetch_order_book(args.pair, limit=args.depth)
        mid = float(OrderBookAnalyzer(raw).mid_price)
    except Exception as exc:
        print(f"Error fetching order book: {exc}", file=sys.stderr)
        return 1

    dex_price = (
        Decimal(str(args.dex_price))
        if args.dex_price is not None
        else Decimal(str(mid)) * Decimal("0.998")
    )

    dex_source = StaticDexSource(price=dex_price, impact_bps=Decimal("1.2"), fee_bps=Decimal("30"))

    from src.inventory.pnl import PnLTracker
    from src.inventory.tracker import VenueTracker, Venue

    base, quote = args.pair.split("/")
    inventory = VenueTracker([Venue.BINANCE, Venue.WALLET])
    inventory.update_from_cex(
        Venue.BINANCE,
        {base: {"free": "300", "locked": "0"}, quote: {"free": "1500000", "locked": "0"}}
    )
    inventory.update_from_wallet(Venue.WALLET, {base: "300", quote: "1500000"})

    checker = ArbChecker(
        dex_source=dex_source,
        cex_client=cex_client,
        inventory=inventory,
        pnl_tracker=PnLTracker()
    )

    try:
        result = checker.assess(args.pair, size=args.size, gas_gwei=args.gas_gwei)
    except Exception as exc:
        print(f"Error running assessment: {exc}", file=sys.stderr)
        return 1

    _print_assessment(result, args.size)
    return 0


if __name__ == "__main__":
    sys.exit(_run_cli())
