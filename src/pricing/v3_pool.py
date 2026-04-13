"""
Uniswap V3 concentrated-liquidity pool math.

Background
----------
Unlike Uniswap V2, which distributes liquidity uniformly across all prices,
V3 lets liquidity providers concentrate their capital in a chosen price range
[tick_lower, tick_upper]. Only the liquidity that is *in range* participates
in a given swap.

This module implements a single-tick approximation, meaning we assume the
entire active liquidity stays in range for the duration of the trade. This
is a good approximation for small-to-medium trades and is the same approach
used by off-chain quoters before they send a quote request to the contract.

Key V3 concepts used here
--------------------------
sqrtPriceX96  : current √price encoded as a Q64.96 fixed-point integer
                (i.e. the actual √price = sqrtPriceX96 / 2**96).
liquidity     : the amount of virtual liquidity currently in-range.
fee_ppm       : fee expressed in parts-per-million (100=0.01%, 500=0.05%,
                3000=0.30%, 10000=1.00%).

Swap direction
--------------
zeroForOne  : selling token0, buying token1 → √price decreases
oneForZero  : selling token1, buying token0 → √price increases

Formulas (exact, integer arithmetic)
-------------------------------------
Given L (liquidity) and √P (sqrtPriceX96), for zeroForOne:
    net_in   = qty_in × (1_000_000 − fee_ppm) // 1_000_000
    extra    = ⌈net_in × √P / Q96⌉ (prevent under-estimation)
    new_√P   = L × √P // (L + extra)
    qty_out  = L × (√P − new_√P) // Q96

For oneForZero:
    net_in   = qty_in × (1_000_000 − fee_ppm) // 1_000_000
    new_√P   = √P + net_in × Q96 // L
    qty_out  = L × (new_√P − √P) × Q96 // (√P × new_√P)

Both formulas are derived from the V3 white paper and match the Solidity
contract output for small-to-medium trades within a single tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from web3 import Web3

from src.core.types import Address, Token
from src.pricing.amm import _token_meta

if TYPE_CHECKING:
    from src.chain.client import ChainClient

# Q64.96 denominator - all sqrt-price values are stored in this fixed-point format
Q96: int = 2 ** 96

# Fee denominator for V3 (fee_ppm / 1_000_000 = actual fee fraction)
_FEE_DENOM: int = 1_000_000

# Valid V3 fee tiers (parts-per-million)
_VALID_FEES: frozenset[int] = frozenset({100, 500, 3_000, 10_000})

# Minimal ABI to read V3 pool state
_V3_POOL_ABI = [
    {"name": "slot0", "type": "function", "inputs": [],
     "outputs": [{"name": "sqrtPriceX96", "type": "uint160"},
                 {"name": "tick", "type": "int24"},
                 {"name": "observationIndex", "type": "uint16"},
                 {"name": "observationCardinality", "type": "uint16"},
                 {"name": "observationCardinalityNext", "type": "uint16"},
                 {"name": "feeProtocol", "type": "uint8"},
                 {"name": "unlocked", "type": "bool"}], "stateMutability": "view"},
    {"name": "liquidity", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "uint128"}], "stateMutability": "view"},
    {"name": "token0", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "address"}], "stateMutability": "view"},
    {"name": "token1", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "address"}], "stateMutability": "view"},
    {"name": "fee", "type": "function", "inputs": [],
     "outputs": [{"name": "", "type": "uint24"}], "stateMutability": "view"},
]


@dataclass
class V3Pool:
    """
    Snapshot of a Uniswap V3 concentrated-liquidity pool.

    Attributes
    ----------
    contract       : on-chain address of the pool
    token0         : lower-sorted token (canonical V3 ordering)
    token1         : higher-sorted token
    sqrt_price_x96 : current √price in Q64.96 fixed-point format
    liquidity      : active in-range liquidity
    fee_ppm        : swap fee in parts-per-million (100/500/3000/10000)
    tick           : current tick index (informational only)
    """

    contract: Address
    token0: Token
    token1: Token
    sqrt_price_x96: int
    liquidity: int
    fee_ppm: int = 3_000
    tick: int = 0

    def __post_init__(self) -> None:
        if self.sqrt_price_x96 <= 0:
            raise ValueError(f"sqrt_price_x96 must be positive, got {self.sqrt_price_x96}.")
        if self.liquidity <= 0:
            raise ValueError(f"liquidity must be positive, got {self.liquidity}.")
        if self.fee_ppm not in _VALID_FEES:
            raise ValueError(
                f"fee_ppm must be one of {sorted(_VALID_FEES)}, got {self.fee_ppm}."
            )
        if self.token0 == self.token1:
            raise ValueError("token0 and token1 must be different tokens.")

    # ------------------------------------------------------------------
    # Core swap math  (integer only, single-tick approximation)
    # ------------------------------------------------------------------

    def out_for_in(self, qty_in: int, selling: Token) -> int:
        """
        Compute how many tokens the buyer receives for *qty_in* of *selling*.

        Raises TypeError / ValueError for invalid inputs, and ValueError when
        *selling* does not belong to this pool.
        """
        if not isinstance(qty_in, int):
            raise TypeError(f"qty_in must be int, got {type(qty_in).__name__}.")
        if qty_in <= 0:
            raise ValueError(f"qty_in must be positive, got {qty_in}.")

        sp = self.sqrt_price_x96
        L = self.liquidity
        # Deduct fee before applying to price curve
        net = qty_in * (_FEE_DENOM - self.fee_ppm) // _FEE_DENOM

        if selling == self.token0:
            # zeroForOne: price (√P) decreases
            extra = (net * sp + Q96 - 1) // Q96   # Ceiling division
            new_sp = (L * sp) // (L + extra)
            return L * (sp - new_sp) // Q96

        if selling == self.token1:
            # oneForZero: price (√P) increases
            new_sp = sp + (net * Q96) // L
            delta = new_sp - sp
            # token0 amount = L × Δ(1/√P) = L × delta × Q96 / (sp × new_sp)
            return L * delta * Q96 // (sp * new_sp)

        raise ValueError(
            f"{selling.symbol} does not belong to this pool "
            f"({self.token0.symbol}/{self.token1.symbol})."
        )

    # ------------------------------------------------------------------
    # Price helpers (Decimal — display only)
    # ------------------------------------------------------------------

    def spot_price(self, selling: Token) -> Decimal:
        """
        Current spot price expressed as units of bought-token per sold-token.

        For token0 → token1: price = (√P / Q96)²
        For token1 → token0: price = (Q96 / √P)²
        """
        if selling not in (self.token0, self.token1):
            raise ValueError(
                f"{selling.symbol} does not belong to this pool."
            )
        ratio = Decimal(self.sqrt_price_x96) / Decimal(Q96)
        if selling == self.token0:
            return ratio * ratio
        return Decimal(1) / (ratio * ratio)

    def price_impact(self, qty_in: int, selling: Token) -> Decimal:
        """
        Fractional slippage relative to spot price: (spot − fill) / spot.
        Returns 0 when spot is zero; returns 1 when output is zero.
        """
        sp = self.spot_price(selling)
        if sp == 0:
            return Decimal(0)
        received = self.out_for_in(qty_in, selling)
        if received == 0:
            return Decimal(1)
        fill = Decimal(received) / Decimal(qty_in)
        impact = (sp - fill) / sp
        # Clamp to [0, 1] — minor negative values can arise from fee rounding
        return max(Decimal(0), impact)

    # ------------------------------------------------------------------
    # Chain bootstrap
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, pool_addr: Address, client: "ChainClient") -> "V3Pool":
        """Read live pool state from the blockchain."""
        w3 = client._get_w3()
        pool = w3.eth.contract(
            address=Web3.to_checksum_address(pool_addr.checksum),
            abi=_V3_POOL_ABI
        )
        slot0 = pool.functions.slot0().call()
        sqrt_price_x96 = slot0[0]
        tick = slot0[1]
        liquidity = pool.functions.liquidity().call()
        fee_ppm = pool.functions.fee().call()
        t0 = _token_meta(w3, pool.functions.token0().call())
        t1 = _token_meta(w3, pool.functions.token1().call())
        return cls(
            contract=pool_addr,
            token0=t0, token1=t1,
            sqrt_price_x96=sqrt_price_x96,
            liquidity=liquidity,
            fee_ppm=fee_ppm,
            tick=tick
        )
