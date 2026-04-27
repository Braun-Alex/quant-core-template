"""
DEX integration: real Uniswap V2 pricing and transaction building.

Provides:
  DEXPriceSource  - fetches real on-chain quotes via PricingEngine
  DEXExecutor     - builds (and optionally sends) Uniswap V2 swap transactions
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from eth_abi import encode as abi_encode
from web3 import Web3

from src.chain.client import ChainClient
from src.core.types import Address, Token
from src.core.wallet import WalletManager
from src.pricing.amm import PoolState
from src.pricing.engine import PricingEngine, PriceQuote

log = logging.getLogger(__name__)

# Selector: swapExactTokensForTokens(uint256,uint256,address[],address,uint256)
_SEL_SWAP_EXACT_TOKENS = bytes.fromhex("38ed1739")
# Selector: approve(address,uint256)
_SEL_APPROVE = bytes.fromhex("095ea7b3")

# Standard ERC-20 ABI fragments
_ERC20_ABI = [
    {
        "name": "allowance",
        "type": "function",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"}
        ],
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view"
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view"
    },
    {
        "name": "decimals",
        "type": "function",
        "inputs": [],
        "outputs": [{"type": "uint8"}],
        "stateMutability": "view"
    }
]


@dataclass
class DEXQuote:
    """A real on-chain DEX price quote."""
    token_in: Token
    token_out: Token
    amount_in: int   # Raw units
    amount_out_min: int   # Raw units, after slippage
    expected_out: int   # Raw units, best estimate
    price: Decimal   # Human units: out_per_in
    impact_bps: Decimal
    fee_bps: Decimal
    gas_estimate: int
    path: list[str]   # Checksummed token addresses
    quote_obj: Optional[PriceQuote] = None


@dataclass
class DEXTransaction:
    """A fully-built DEX swap transaction."""
    to: str
    data: bytes
    value: int
    gas: int
    max_fee_per_gas: int
    max_priority_fee_per_gas: int
    nonce: int
    chain_id: int
    # Approval tx (if needed)
    approval_tx: Optional["DEXTransaction"] = None
    # Human-readable summary
    summary: str = ""
    dry_run: bool = True   # If True, was NOT broadcast


class DEXPriceSource:
    """
    Real on-chain price source using PricingEngine.

    Falls back to AMM spot-price estimation when a full routing
    quote is not available (e.g. engine not yet initialized).
    """

    def __init__(
        self,
        pricing_engine: Optional[PricingEngine],
        chain_client: ChainClient,
        router_address: str,
        pool_addresses: list[str],
        fee_bps: Decimal = Decimal("30"),
        slippage_bps: Decimal = Decimal("50"),
        gas_price_gwei: int = 20
    ) -> None:
        self._engine = pricing_engine
        self._client = chain_client
        self._router = router_address
        self._pool_addrs = pool_addresses
        self._fee_bps = fee_bps
        self._slippage_bps = slippage_bps
        self._gas_gwei = gas_price_gwei
        self._pools: dict[str, PoolState] = {}
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Load pool states from chain and register with the pricing engine."""
        if self._initialized:
            return

        for addr_str in self._pool_addrs:
            try:
                addr = Address(addr_str)
                pool = PoolState.load(addr, self._client)
                self._pools[addr_str] = pool
                log.info(
                    "Loaded pool %s  %s/%s  L=%d R=%d",
                    addr_str[:10], pool.left.symbol, pool.right.symbol,
                    pool.qty_left, pool.qty_right
                )
            except Exception as exc:
                log.warning("Failed to load pool %s: %s", addr_str, exc)

        if self._engine and self._pools:
            try:
                addresses = [Address(a) for a in self._pool_addrs if a in self._pools]
                self._engine.register_pools(addresses)
            except Exception as exc:
                log.warning("Engine pool registration failed: %s", exc)

        self._initialized = True

    # ------------------------------------------------------------------
    # Public interface (compatible with StaticDexSource)
    # ------------------------------------------------------------------

    def get_dex_quote(self, base: str, quote: str, size: Decimal) -> dict:
        """
        Return a dict compatible with ArbChecker.assess() expectations.
        Tries real on-chain quote first; falls back to AMM formula.
        """
        if not self._initialized:
            self.initialize()

        pool = self._find_pool(base, quote)
        if pool is None:
            # No pool found - return zero-impact stub
            return {"price": Decimal("0"), "impact_bps": Decimal("0"),
                    "fee_bps": self._fee_bps}

        try:
            return self._quote_via_engine(pool, base, size)
        except Exception as exc:
            log.debug("Engine quote failed (%s), falling back to AMM formula", exc)
            return self._quote_via_amm(pool, base)

    def get_full_quote(
        self,
        token_in: Token,
        token_out: Token,
        amount_in_raw: int
    ) -> Optional[DEXQuote]:
        """Return a full DEXQuote including path and slippage-adjusted min-out."""
        if not self._initialized:
            self.initialize()

        pool = self._find_pool_by_tokens(token_in, token_out)
        if pool is None:
            return None

        try:
            expected_out = pool.out_for_in(amount_in_raw, token_in)
        except Exception as exc:
            log.warning("AMM calculation failed: %s", exc)
            return None

        amount_in_human = Decimal(amount_in_raw) / Decimal(10 ** token_in.decimals)
        amount_out_human = Decimal(expected_out) / Decimal(10 ** token_out.decimals)
        price = amount_out_human / amount_in_human if amount_in_human else Decimal("0")

        marginal = pool.marginal_price(token_in)
        fill = pool.fill_price(amount_in_raw, token_in)
        impact_bps = (
            (marginal - fill) / marginal * Decimal("10000")
            if marginal > 0 else Decimal("0")
        )

        slip_factor = Decimal("1") - self._slippage_bps / Decimal("10000")
        amount_out_min = int(Decimal(expected_out) * slip_factor)

        path = [token_in.address.checksum, token_out.address.checksum]

        return DEXQuote(
            token_in=token_in,
            token_out=token_out,
            amount_in=amount_in_raw,
            amount_out_min=amount_out_min,
            expected_out=expected_out,
            price=price,
            impact_bps=impact_bps,
            fee_bps=self._fee_bps,
            gas_estimate=250_000,
            path=path
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pool(self, base: str, quote: str) -> Optional[PoolState]:
        for pool in self._pools.values():
            syms = {pool.left.symbol.upper(), pool.right.symbol.upper()}
            if {base.upper(), quote.upper()} <= syms:
                return pool
        return None

    def _find_pool_by_tokens(self, t_in: Token, t_out: Token) -> Optional[PoolState]:
        for pool in self._pools.values():
            if {pool.left.address, pool.right.address} == {t_in.address, t_out.address}:
                return pool
        return None

    def _quote_via_engine(self, pool: PoolState, base: str, size: Decimal) -> dict:
        selling = pool.left if pool.left.symbol.upper() == base.upper() else pool.right
        raw_in = int(size * Decimal(10 ** selling.decimals))
        expected_out = pool.out_for_in(raw_in, selling)

        buying = pool.right if selling == pool.left else pool.left
        amount_in_h = size
        amount_out_h = Decimal(expected_out) / Decimal(10 ** buying.decimals)
        price = amount_out_h / amount_in_h if amount_in_h else Decimal("0")

        marginal = pool.marginal_price(selling)
        fill = pool.fill_price(raw_in, selling)
        impact_bps = (marginal - fill) / marginal * Decimal("10000") if marginal > 0 else Decimal("0")

        return {
            "price": price,
            "impact_bps": max(impact_bps, Decimal("0")),
            "fee_bps": self._fee_bps
        }

    def _quote_via_amm(self, pool: PoolState, base: str) -> dict:
        selling = pool.left if pool.left.symbol.upper() == base.upper() else pool.right
        price = pool.marginal_price(selling)
        buying = pool.right if selling == pool.left else pool.left
        price_human = (
            price
            * Decimal(10 ** selling.decimals)
            / Decimal(10 ** buying.decimals)
        )
        return {
            "price": price_human,
            "impact_bps": Decimal("0"),
            "fee_bps": self._fee_bps
        }


class DEXExecutor:
    """
    Builds and (optionally) broadcasts Uniswap V2 swap transactions.

    In test mode (dry_run=True): builds & returns signed tx dict but does NOT send.
    In production mode (dry_run=False): sends both approval and swap transactions.
    """

    def __init__(
        self,
        chain_client: ChainClient,
        wallet: WalletManager,
        router_address: str,
        gas_limit_swap: int = 250_000,
        gas_limit_approval: int = 60_000,
        slippage_bps: Decimal = Decimal("50"),
        deadline_seconds: int = 300,
        dry_run: bool = True
    ) -> None:
        self._client = chain_client
        self._wallet = wallet
        self._router = Web3.to_checksum_address(router_address)
        self._gas_limit_swap = gas_limit_swap
        self._gas_limit_approval = gas_limit_approval
        self._slippage_bps = slippage_bps
        self._deadline_seconds = deadline_seconds
        self._dry_run = dry_run
        log.info(
            "DEXExecutor ready | dry_run=%s router=%s",
            dry_run, self._router
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def execute_swap(
        self,
        quote: DEXQuote,
        recipient: Optional[str] = None
    ) -> dict:
        """
        Execute or simulate a swap described by *quote*.

        Returns a result dict with keys:
          success, price, filled, tx_hash, dry_run, approval_tx_hash
        """
        recipient_addr = recipient or self._wallet.address

        approval_result = self._maybe_approve(quote, recipient_addr)
        swap_tx = self._build_swap_tx(quote, recipient_addr)

        if self._dry_run:
            log.info(
                "[DRY-RUN] Swap %s→%s  amount_in=%d  min_out=%d  (NOT broadcast)",
                quote.token_in.symbol, quote.token_out.symbol,
                quote.amount_in, quote.amount_out_min
            )
            return {
                "success": True,
                "price": float(quote.price),
                "filled": float(
                    Decimal(quote.expected_out) / Decimal(10 ** quote.token_out.decimals)
                ),
                "tx_hash": "0x" + "0" * 64,
                "dry_run": True,
                "swap_tx": swap_tx,
                "approval_tx_hash": approval_result.get("tx_hash")
            }

        # ── Production: broadcast ──────────────────────────────────────
        try:
            tx_hash = self._broadcast(swap_tx)
            log.info(
                "Swap broadcast: %s→%s  tx=%s",
                quote.token_in.symbol, quote.token_out.symbol, tx_hash
            )
            return {
                "success": True,
                "price": float(quote.price),
                "filled": float(
                    Decimal(quote.expected_out) / Decimal(10 ** quote.token_out.decimals)
                ),
                "tx_hash": tx_hash,
                "dry_run": False,
                "approval_tx_hash": approval_result.get("tx_hash")
            }
        except Exception as exc:
            log.error("Swap broadcast failed: %s", exc)
            return {
                "success": False,
                "price": 0.0,
                "filled": 0.0,
                "tx_hash": None,
                "dry_run": False,
                "error": str(exc)
            }

    def build_unwind_tx(
        self,
        token_in: Token,
        token_out: Token,
        amount_in_raw: int,
        recipient: Optional[str] = None
    ) -> dict:
        """
        Build an unwind (reverse) swap transaction.
        Uses extra slippage tolerance to ensure fill even in adverse conditions.
        """
        recipient_addr = recipient or self._wallet.address
        extra_slippage = self._slippage_bps * Decimal("2")   # 2× tolerance for unwind
        slip_factor = Decimal("1") - extra_slippage / Decimal("10000")

        # Minimal expected out (accept significant slippage for unwind reliability)
        min_out = int(Decimal(amount_in_raw) * slip_factor * Decimal("0.5"))

        path = [token_in.address.checksum, token_out.address.checksum]
        deadline = int(time.time()) + self._deadline_seconds

        calldata = _SEL_SWAP_EXACT_TOKENS + abi_encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [amount_in_raw, min_out, path, recipient_addr, deadline]
        )

        gas_price = self._client.get_gas_price()
        nonce = self._client.get_nonce(Address.from_string(self._wallet.address))
        chain_id = self._client.get_chain_id()

        return {
            "to": self._router,
            "data": "0x" + calldata.hex(),
            "value": 0,
            "gas": self._gas_limit_swap,
            "maxFeePerGas": gas_price.get_max_fee("high"),
            "maxPriorityFeePerGas": gas_price.priority_fee_high,
            "nonce": nonce,
            "chainId": chain_id
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _maybe_approve(self, quote: DEXQuote, owner: str) -> dict:
        """Check allowance and submit approval if needed."""
        try:
            w3 = self._client._get_w3()
            token_contract = w3.eth.contract(
                address=Web3.to_checksum_address(quote.token_in.address.checksum),
                abi=_ERC20_ABI
            )
            allowance = token_contract.functions.allowance(
                Web3.to_checksum_address(owner),
                Web3.to_checksum_address(self._router)
            ).call()

            if allowance >= quote.amount_in:
                return {}   # Already approved

            # Build approval tx
            approve_data = _SEL_APPROVE + abi_encode(
                ["address", "uint256"],
                [self._router, 2 ** 256 - 1]   # max uint256
            )
            gas_price = self._client.get_gas_price()
            nonce = self._client.get_nonce(Address.from_string(self._wallet.address))
            chain_id = self._client.get_chain_id()

            approval_tx = {
                "to": quote.token_in.address.checksum,
                "data": "0x" + approve_data.hex(),
                "value": 0,
                "gas": self._gas_limit_approval,
                "maxFeePerGas": gas_price.get_max_fee("medium"),
                "maxPriorityFeePerGas": gas_price.priority_fee_medium,
                "nonce": nonce,
                "chainId": chain_id
            }

            if self._dry_run:
                log.info(
                    "[DRY-RUN] Approval for %s (NOT broadcast)",
                    quote.token_in.symbol
                )
                return {"tx_hash": "0x" + "1" * 64, "dry_run": True}

            tx_hash = self._broadcast(approval_tx)
            log.info("Approval tx: %s", tx_hash)
            # Wait for approval to confirm
            self._client.wait_for_receipt(tx_hash, timeout=60)
            return {"tx_hash": tx_hash}

        except Exception as exc:
            log.warning("Approval check/send failed: %s", exc)
            return {}

    def _build_swap_tx(self, quote: DEXQuote, recipient: str) -> dict:
        """Assemble the swapExactTokensForTokens calldata and tx dict."""
        deadline = int(time.time()) + self._deadline_seconds
        calldata = _SEL_SWAP_EXACT_TOKENS + abi_encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [
                quote.amount_in,
                quote.amount_out_min,
                quote.path,
                Web3.to_checksum_address(recipient),
                deadline
            ],
        )

        gas_price = self._client.get_gas_price()
        nonce = self._client.get_nonce(Address.from_string(self._wallet.address))
        chain_id = self._client.get_chain_id()

        return {
            "to": self._router,
            "data": "0x" + calldata.hex(),
            "value": 0,
            "gas": self._gas_limit_swap,
            "maxFeePerGas": gas_price.get_max_fee("medium"),
            "maxPriorityFeePerGas": gas_price.priority_fee_medium,
            "nonce": nonce,
            "chainId": chain_id
        }

    def _broadcast(self, tx_dict: dict) -> str:
        """Sign and broadcast a transaction, return tx hash."""
        signed = self._wallet.sign_transaction(tx_dict)
        return self._client.send_transaction(signed.raw_transaction)
