"""
Trade-size impact analysis for a single liquidity pool.

Key concepts
------------
- "Human" units: amounts divided by 10^decimals, as a user would read them.
- "Raw" units: integer amounts as stored on-chain (wei for ETH, μUSDC etc.).
- All math stays in raw units; human units appear only for display.

CLI usage
---------
    python -m src.pricing.impact_analyzer \\
        0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc \\
        --sell USDC \\
        --amounts 1000,10000,100000 \\
        --rpc ETH_RPC_URL
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal

from src.chain.client import ChainClient
from src.core.types import Address, Token
from src.pricing.amm import PoolState

_WETH_LOWER = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"


# ---------------------------------------------------------------------------
# Core analyzer
# ---------------------------------------------------------------------------

class ImpactAnalyzer:
    """Analyses how execution quality changes as trade size grows."""

    def __init__(self, pool: PoolState) -> None:
        self.pool = pool

    def table(self, selling: Token, raw_amounts: list[int]) -> list[dict]:
        """
        Build one row per entry in *raw_amounts*.

        Each row:
            raw_in         int     - raw sold amount
            raw_out        int     - raw received amount
            spot           Decimal - marginal price (bought/sold, human units)
            fill           Decimal - actual fill price (bought/sold, human units)
            impact_pct     Decimal - (spot-fill)/spot × 100
        """
        if not raw_amounts:
            return []

        buy_token = (
            self.pool.right if selling == self.pool.left else self.pool.left
        )
        # Spot in human units = (raw_out/dec_out) / (raw_in/dec_in)
        raw_spot = self.pool.marginal_price(selling)
        human_spot = raw_spot * Decimal(10 ** selling.decimals) / Decimal(10 ** buy_token.decimals)
        if human_spot != 0:
            human_spot = Decimal(1) / human_spot  # Price as "in per out"

        rows = []
        for raw_in in raw_amounts:
            raw_out = self.pool.out_for_in(raw_in, selling)
            h_in = Decimal(raw_in) / Decimal(10 ** selling.decimals)
            h_out = Decimal(raw_out) / Decimal(10 ** buy_token.decimals)
            fill = h_in / h_out if h_out else Decimal(0)
            impact_pct = self.pool.slippage(raw_in, selling) * 100
            rows.append(dict(
                raw_in=raw_in, raw_out=raw_out,
                spot=human_spot, fill=fill, impact_pct=impact_pct
            ))
        return rows

    def max_trade_below(self, selling: Token, max_impact_pct: Decimal) -> int:
        """
        Binary-search the largest raw amount whose slippage stays at or below
        *max_impact_pct* percent.
        """
        if max_impact_pct <= 0:
            raise ValueError("max_impact_pct must be positive.")
        threshold = max_impact_pct / 100
        liq_in, _ = self.pool._orient(selling)
        lo, hi, best = 1, liq_in // 2, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.pool.slippage(mid, selling) <= threshold:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def cost_breakdown(
            self,
            raw_in: int,
            selling: Token,
            gas_gwei: int,
            gas_units: int = 150_000,
    ) -> dict:
        """
        Decompose a trade into gross output, gas overhead, and net output.

        Gas conversion to output-token units is only possible when one of the
        tokens is WETH; otherwise gas_in_output is reported as 0.
        """
        gross = self.pool.out_for_in(raw_in, selling)
        gas_wei = gas_gwei * 10 ** 9 * gas_units
        buy_token = self.pool.right if selling == self.pool.left else self.pool.left

        if buy_token.address.lower == _WETH_LOWER:
            gas_output = gas_wei
        elif selling.address.lower == _WETH_LOWER:
            gas_output = int(Decimal(gas_wei) * self.pool.marginal_price(selling))
        else:
            gas_output = 0

        net = max(0, gross - gas_output)
        s_dec = selling.decimals
        b_dec = buy_token.decimals
        h_net = Decimal(net) / Decimal(10 ** b_dec)
        h_in = Decimal(raw_in) / Decimal(10 ** s_dec)
        eff_px = h_net / h_in if h_in else Decimal(0)

        return dict(gross_out=gross, gas_wei=gas_wei,
                    gas_in_output=gas_output, net_out=net,
                    effective_price=eff_px)


# ---------------------------------------------------------------------------
# Terminal renderer
# ---------------------------------------------------------------------------

def _human(raw: int, decimals: int) -> str:
    v = Decimal(raw) / Decimal(10 ** decimals)
    return f"{v:,.4f}".rstrip("0").rstrip(".")


_W = [14, 14, 14, 11]


def _bar(left: str, medium: str, right: str) -> str:
    return left + medium.join("─" * (w + 2) for w in _W) + right


def _row(*cols: str) -> str:
    cells = [f" {c:>{_W[i]}} " for i, c in enumerate(cols)]
    return "│" + "│".join(cells) + "│"


def render(
        rows: list[dict],
        selling: Token,
        buying: Token,
        pool: PoolState,
        max_raw: int,
        max_impact_pct: Decimal
) -> str:
    liq_s, liq_b = pool._orient(selling)
    lines = [
        f"\nImpact analysis — {selling.symbol} → {buying.symbol}",
        f"Pool liquidity : {_human(liq_s, selling.decimals)} {selling.symbol} "
        f"/ {_human(liq_b, buying.decimals)} {buying.symbol}",
        "",
        _bar("┌", "┬", "┐"),
        _row(f"{selling.symbol} sold", f"{buying.symbol} received",
             "Fill price", "Impact"),
        _bar("├", "┼", "┤"),
    ]
    for r in rows:
        lines.append(_row(
            _human(r["raw_in"], selling.decimals),
            _human(r["raw_out"], buying.decimals),
            f"{r['fill']:,.2f}",
            f"{r['impact_pct']:.2f}%",
        ))
    lines += [
        _bar("└", "┴", "┘"),
        f"\nLargest trade under {max_impact_pct}% impact: "
        f"{_human(max_raw, selling.decimals)} {selling.symbol}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token resolution helper
# ---------------------------------------------------------------------------

def _resolve(pool: PoolState, identifier: str) -> Token:
    upper = identifier.strip().upper()
    if upper == pool.left.symbol.upper():
        return pool.left
    if upper == pool.right.symbol.upper():
        return pool.right
    try:
        a = Address(identifier)
        if a == pool.left.address:
            return pool.left
        if a == pool.right.address:
            return pool.right
    except Exception:
        pass
    raise ValueError(
        f"Token '{identifier}' not found in pool "
        f"({pool.left.symbol}/{pool.right.symbol})."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m src.pricing.impact_analyzer",
        description="Analyze swap price impact for a Uniswap-V2 pool.",
    )
    p.add_argument("pool", help="Pair contract address (0x...)")
    p.add_argument("--sell", required=True, help="Token being sold (symbol or address)")
    p.add_argument("--amounts", required=True,
                   help="Comma-separated trade sizes in human units (e.g. 1000,10000)")
    p.add_argument("--rpc", default="https://ethereum.publicnode.com")
    p.add_argument("--max-impact", default="1",
                   help="Impact threshold %% for max-trade-size line (default: 1)")
    args = p.parse_args(argv)

    try:
        max_impact = Decimal(args.max_impact)
    except Exception:
        print(f"Error: --max-impact '{args.max_impact}' is not a number.", file=sys.stderr)
        return 1

    try:
        client = ChainClient([args.rpc], max_retries=1)
        pool = PoolState.load(Address(args.pool), client)
    except Exception as exc:
        print(f"Error loading pool: {exc}", file=sys.stderr)
        return 1

    try:
        selling = _resolve(pool, args.sell)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    buying = pool.right if selling == pool.left else pool.left

    try:
        raw_amounts = [
            int(Decimal(s.strip()) * Decimal(10 ** selling.decimals))
            for s in args.amounts.split(",")
        ]
    except Exception as exc:
        print(f"Error parsing --amounts: {exc}", file=sys.stderr)
        return 1

    analyser = ImpactAnalyzer(pool)
    try:
        rows = analyser.table(selling, raw_amounts)
        max_raw = analyser.max_trade_below(selling, max_impact)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(render(rows, selling, buying, pool, max_raw, max_impact))
    return 0


if __name__ == "__main__":
    sys.exit(main())
