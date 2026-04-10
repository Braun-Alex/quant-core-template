"""
Constant-product market maker math for ERC-20 token pairs.

Design decisions:
- Everything that touches price uses integer arithmetic exclusively.
  Python's arbitrary-precision integers guarantee no rounding loss
  regardless of reserve magnitude, which is critical for matching
  on-chain Solidity behavior.
- The fee is expressed in basis points (1 bps = 0.01 %).
  Uniswap V2 uses 30 bps (0.30 %). The integer fee multiplier
  approach avoids any floating-point representation of 0.997.
- Immutable swap simulation returns a fresh pool state so callers
  can chain hypothetical swaps without side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from web3 import Web3

from src.core.types import Address, Token
from src.chain.client import ChainClient


# ---------------------------------------------------------------------------
# Minimal ABI slices — only what we need to bootstrap from chain
# ---------------------------------------------------------------------------

_PAIR_ABI = [
    {"name": "getReserves", "type": "function", "inputs": [],
     "outputs": [{"name": "reserve0", "type": "uint112"},
                 {"name": "reserve1", "type": "uint112"},
                 {"name": "blockTimestampLast", "type": "uint32"}],
     "stateMutability": "view"},
    {"name": "token0", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "address"}], "stateMutability": "view"},
    {"name": "token1", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "address"}], "stateMutability": "view"}
]

_TOKEN_ABI = [
    {"name": "symbol", "type": "function", "inputs": [],
     "outputs": [{"type": "string"}], "stateMutability": "view"},
    {"name": "decimals", "type": "function", "inputs": [],
     "outputs": [{"type": "uint8"}], "stateMutability": "view"}
]

_FEE_SCALE = 10_000   # Basis-point denominator


def _token_meta(w3: Web3, addr: str) -> Token:
    """Read symbol + decimals from chain; fall back gracefully on any error."""
    cs = Web3.to_checksum_address(addr)
    try:
        c = w3.eth.contract(address=cs, abi=_TOKEN_ABI)
        return Token(address=Address(cs),
                     symbol=c.functions.symbol().call(),
                     decimals=c.functions.decimals().call())
    except Exception:
        return Token(address=Address(cs), symbol=cs[:8], decimals=18)


# ---------------------------------------------------------------------------
# Pool state
# ---------------------------------------------------------------------------

@dataclass
class PoolState:
    """
    Snapshot of a Uniswap-V2-compatible liquidity pool.

    Attributes
    ----------
    contract  : on-chain address of the pair
    left      : the token whose amount is stored in qty_left
    right     : the other token
    qty_left  : raw reserve of *left*  (uint112 range)
    qty_right : raw reserve of *right* (uint112 range)
    fee_bps   : swap fee in basis points; default 30 (= 0.30 %)
    """

    contract: Address
    left: Token
    right: Token
    qty_left: int
    qty_right: int
    fee_bps: int = 30

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.qty_left <= 0 or self.qty_right <= 0:
            raise ValueError(
                "Both pool reserves must be strictly positive - "
                f"got qty_left={self.qty_left}, qty_right={self.qty_right}."
            )
        if not (0 <= self.fee_bps < _FEE_SCALE):
            raise ValueError(
                f"fee_bps must be in [0, {_FEE_SCALE - 1}], got {self.fee_bps}."
            )
        if self.left == self.right:
            raise ValueError("A pool cannot hold the same token on both sides.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _orient(self, selling: Token) -> tuple[int, int]:
        """
        Return (liquidity_of_sold_token, liquidity_of_bought_token).
        Raises ValueError when *selling* is not part of this pool.
        """
        if selling == self.left:
            return self.qty_left, self.qty_right
        if selling == self.right:
            return self.qty_right, self.qty_left
        raise ValueError(
            f"{selling.symbol} is not traded on this pool "
            f"({self.left.symbol}/{self.right.symbol})."
        )

    # ------------------------------------------------------------------
    # Core integer AMM formulas
    # ------------------------------------------------------------------

    def out_for_in(self, qty_in: int, selling: Token) -> int:
        """
        How many tokens will the buyer receive when selling *qty_in*?

        Derivation (Uniswap V2 exact match):
            net_in      = qty_in × (scale − fee_bps)
            amount_out  = net_in × liq_out
                          ─────────────────────────────────────
                          liq_in × scale + net_in

        Integer floor division matches Solidity's behavior.
        """
        if not isinstance(qty_in, int):
            raise TypeError(f"qty_in must be int, got {type(qty_in).__name__}.")
        if qty_in <= 0:
            raise ValueError(f"qty_in must be positive, got {qty_in}.")

        liq_in, liq_out = self._orient(selling)
        net_in = qty_in * (_FEE_SCALE - self.fee_bps)
        return (net_in * liq_out) // (liq_in * _FEE_SCALE + net_in)

    def in_for_out(self, qty_out: int, buying: Token) -> int:
        """
        Minimum tokens that must be sold to receive exactly *qty_out*.

        Uses ceiling division (+1) so the buyer always covers the full cost.
        """
        if not isinstance(qty_out, int):
            raise TypeError(f"qty_out must be int, got {type(qty_out).__name__}.")
        if qty_out <= 0:
            raise ValueError(f"qty_out must be positive, got {qty_out}.")

        # Orient from the *buying* side
        if buying == self.left:
            liq_in, liq_out = self.qty_right, self.qty_left
        elif buying == self.right:
            liq_in, liq_out = self.qty_left, self.qty_right
        else:
            raise ValueError(
                f"{buying.symbol} is not traded on this pool "
                f"({self.left.symbol}/{self.right.symbol})."
            )

        if qty_out >= liq_out:
            raise ValueError(
                f"Cannot buy {qty_out} - pool only holds {liq_out}."
            )

        numerator = liq_in * qty_out * _FEE_SCALE
        denominator = (liq_out - qty_out) * (_FEE_SCALE - self.fee_bps)
        return numerator // denominator + 1

    # ------------------------------------------------------------------
    # Price helpers  (Decimal - display / comparison only, never in math)
    # ------------------------------------------------------------------

    def marginal_price(self, selling: Token) -> Decimal:
        """Price at the margin: units of bought token per unit of sold token."""
        liq_in, liq_out = self._orient(selling)
        return Decimal(liq_out) / Decimal(liq_in)

    def fill_price(self, qty_in: int, selling: Token) -> Decimal:
        """Actual fill price for a concrete trade (bought / sold in raw units)."""
        received = self.out_for_in(qty_in, selling)
        return Decimal(0) if received == 0 else Decimal(received) / Decimal(qty_in)

    def slippage(self, qty_in: int, selling: Token) -> Decimal:
        """
        Fractional price impact relative to the marginal price.
        A return value of 0.01 means 1 % slippage.
        """
        mp = self.marginal_price(selling)
        if mp == 0:
            return Decimal(0)
        fp = self.fill_price(qty_in, selling)
        return (mp - fp) / mp

    # ------------------------------------------------------------------
    # State projection
    # ------------------------------------------------------------------

    def after_sell(self, qty_in: int, selling: Token) -> "PoolState":
        """
        Project pool state after a swap without mutating the current instance.
        Returns a brand-new PoolState reflecting updated reserves.
        """
        received = self.out_for_in(qty_in, selling)
        if selling == self.left:
            new_left = self.qty_left + qty_in
            new_right = self.qty_right - received
        else:
            new_left = self.qty_left - received
            new_right = self.qty_right + qty_in
        return PoolState(
            contract=self.contract,
            left=self.left,
            right=self.right,
            qty_left=new_left,
            qty_right=new_right,
            fee_bps=self.fee_bps,
        )

    # ------------------------------------------------------------------
    # Chain bootstrap
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, pair_addr: Address, client: "ChainClient") -> "PoolState":
        """Hydrate a PoolState by reading live data from the blockchain."""
        w3 = client._get_w3()
        pair = w3.eth.contract(
            address=Web3.to_checksum_address(pair_addr.checksum),
            abi=_PAIR_ABI
        )
        r0, r1, _ = pair.functions.getReserves().call()
        t0 = _token_meta(w3, pair.functions.token0().call())
        t1 = _token_meta(w3, pair.functions.token1().call())
        return cls(contract=pair_addr, left=t0, right=t1, qty_left=r0, qty_right=r1)
