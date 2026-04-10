"""
Transaction simulation and fork-state control for local Anvil instances.

Design
------
ForkedChain   - wraps a Web3 connection to a local Anvil/Hardhat node and
                exposes Foundry-equivalent cheatcodes as Python methods.
                No business logic here - purely infrastructure.

TradeSimulator - higher-level facade that uses ForkedChain to verify AMM
                 math against live on-chain reserves and to execute actual
                 swaps inside a fork.

Foundry cheatcode equivalents
------------------------------
ForkedChain.checkpoint()         →  vm.snapshot()
ForkedChain.restore(id)          →  vm.revertTo(id)
ForkedChain.fund_eth(addr, wei)  →  vm.deal(addr, amount)
ForkedChain.act_as(addr)         →  vm.startPrank(addr)
ForkedChain.stop_acting_as(addr) →  vm.stopPrank()
ForkedChain.advance_time(ts)     →  vm.warp(ts)
ForkedChain.advance_blocks(n)    →  vm.roll(block.number + n)
ForkedChain.set_erc20_balance    →  deal(token, account, amount)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from eth_abi import decode as abi_decode
from eth_abi import encode as abi_encode
from web3 import Web3

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.router import SwapPath

# 4-byte selectors used in eth_call payloads
_SEL_AMOUNTS_OUT = bytes.fromhex("d06ca61f")   # getAmountsOut(uint256,address[])
_SEL_RESERVES = bytes.fromhex("0902f1ac")   # getReserves()
_SEL_SWAP_EXACT = bytes.fromhex("38ed1739")   # swapExactTokensForTokens(...)

_V2_ROUTER_ADDR = Address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")
_NULL_SENDER = Address("0x0000000000000000000000000000000000000001")


@dataclass
class ExecutionReceipt:
    """Outcome of a simulated or executed swap."""
    ok: bool
    qty_out: int
    gas_used: int
    error: str | None
    logs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# ForkedChain — Anvil cheatcode wrapper
# ---------------------------------------------------------------------------

class ForkedChain:
    """
    Low-level interface to a local Anvil fork.

    All state-manipulation methods mirror Foundry's vm cheatcodes so that
    Python fork tests read similarly to Solidity fork tests.
    """

    def __init__(self, w3: Web3) -> None:
        self._w3 = w3

    @classmethod
    def connect(cls, rpc_url: str) -> "ForkedChain":
        """Build from an HTTP RPC endpoint string."""
        return cls(Web3(Web3.HTTPProvider(rpc_url)))

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def eth_call(self, tx: dict) -> bytes:
        """Execute a read-only call and return raw bytes."""
        return self._w3.eth.call(tx)

    # ------------------------------------------------------------------
    # Snapshot / revert   (vm.snapshot / vm.revertTo)
    # ------------------------------------------------------------------

    def checkpoint(self) -> int:
        """Save EVM state; returns an opaque handle."""
        resp = self._w3.provider.make_request("evm_snapshot", [])
        return int(resp["result"], 16)

    def restore(self, handle: int) -> None:
        """Roll back to a previously saved checkpoint."""
        self._w3.provider.make_request("evm_revert", [hex(handle)])

    # ------------------------------------------------------------------
    # Account cheatcodes   (vm.deal / vm.startPrank / vm.stopPrank)
    # ------------------------------------------------------------------

    def fund_eth(self, addr: Address, wei: int) -> None:
        """Set the ETH balance of *addr* to *wei*."""
        self._w3.provider.make_request("anvil_setBalance",
                                       [addr.checksum, hex(wei)])

    def act_as(self, addr: Address) -> None:
        """Send subsequent transactions as *addr* without its private key."""
        self._w3.provider.make_request("anvil_impersonateAccount", [addr.checksum])

    def stop_acting_as(self, addr: Address) -> None:
        """Stop impersonating *addr*."""
        self._w3.provider.make_request("anvil_stopImpersonatingAccount", [addr.checksum])

    # ------------------------------------------------------------------
    # Block / time   (vm.warp / vm.roll)
    # ------------------------------------------------------------------

    def advance_time(self, unix_ts: int) -> None:
        """Set the next block timestamp to *unix_ts*, then mine one block."""
        self._w3.provider.make_request("evm_setNextBlockTimestamp", [unix_ts])
        self._w3.provider.make_request("evm_mine", [])

    def advance_blocks(self, count: int = 1) -> None:
        """Mine *count* empty blocks."""
        for _ in range(count):
            self._w3.provider.make_request("evm_mine", [])

    def jump_to_block(self, target: int) -> None:
        """Mine blocks until block.number reaches *target*."""
        current = self._w3.eth.block_number
        if target < current:
            raise ValueError(
                f"Cannot rewind: current block {current}, requested {target}."
            )
        self.advance_blocks(target - current)

    # ------------------------------------------------------------------
    # Token balance surgery   (Forge's deal(token, account, amount))
    # ------------------------------------------------------------------

    def set_erc20_balance(
        self,
        token_addr: Address,
        holder: Address,
        amount: int,
        mapping_slot: int = 0
    ) -> None:
        """
        Write directly to the ERC-20 balances storage slot.
        Equivalent to Forge's ``deal(token, holder, amount)``.

        *mapping_slot* is the Solidity storage slot that holds the
        ``mapping(address => uint256)`` for balances (0 for most tokens,
        9 for USDC, etc.).
        """
        slot = Web3.keccak(abi_encode(["address", "uint256"],
                           [holder.checksum, mapping_slot]))
        value = "0x" + format(amount, "064x")
        self._w3.provider.make_request(
            "hardhat_setStorageAt", [token_addr.checksum, slot.hex(), value]
        )

    # ------------------------------------------------------------------
    # Transaction broadcast
    # ------------------------------------------------------------------

    def broadcast(self, tx: dict) -> str:
        """Broadcast a transaction (possibly impersonated) and return its hash."""
        h = self._w3.eth.send_transaction(tx)
        return "0x" + h.hex() if isinstance(h, bytes) else str(h)

    def receipt(self, tx_hash: str) -> dict | None:
        raw = self._w3.eth.get_transaction_receipt(tx_hash)
        return dict(raw) if raw else None


# ---------------------------------------------------------------------------
# TradeSimulator — business-logic facade
# ---------------------------------------------------------------------------

class TradeSimulator:
    """
    Verifies swap math against a live fork and executes swaps for testing.
    """

    def __init__(self, chain: ForkedChain) -> None:
        self._chain = chain

    @classmethod
    def from_url(cls, rpc_url: str) -> "TradeSimulator":
        return cls(ForkedChain.connect(rpc_url))

    @property
    def chain(self) -> ForkedChain:
        """Direct access to the underlying ForkedChain for cheatcode use."""
        return self._chain

    # ------------------------------------------------------------------
    # Read-only simulation   (no state change)
    # ------------------------------------------------------------------

    def quote_via_router(
        self,
        router: Address,
        qty_in: int,
        token_path: list[str],
        caller: Address
    ) -> ExecutionReceipt:
        """
        Call getAmountsOut on *router* and return the last element
        (= tokens received).  No state change; pure view call.
        """
        calldata = _SEL_AMOUNTS_OUT + abi_encode(
            ["uint256", "address[]"], [qty_in, token_path]
        )
        tx = {"to": router.checksum, "from": caller.checksum,
              "data": "0x" + calldata.hex()}
        try:
            raw = self._chain.eth_call(tx)
            (amounts,) = abi_decode(["uint256[]"], raw)
            gas_est = 150_000 + 100_000 * (len(token_path) - 1)
            return ExecutionReceipt(ok=True, qty_out=amounts[-1],
                                    gas_used=gas_est, error=None)
        except Exception as exc:
            return ExecutionReceipt(ok=False, qty_out=0, gas_used=0, error=str(exc))

    def verify_path(
        self,
        path: SwapPath,
        qty_in: int
    ) -> ExecutionReceipt:
        """
        Re-run *path* hop-by-hop using live reserves fetched from the fork.
        Detects any drift between locally cached reserves and current chain state.
        """
        try:
            current = qty_in
            for pool, token_sold in zip(path.pools, path.tokens):
                r0, r1 = self._fetch_reserves(pool.contract)
                live = PoolState(
                    contract=pool.contract,
                    left=pool.left, right=pool.right,
                    qty_left=r0, qty_right=r1,
                    fee_bps=pool.fee_bps
                )
                current = live.out_for_in(current, token_sold)
            return ExecutionReceipt(ok=True, qty_out=current,
                                    gas_used=path.gas_estimate(), error=None)
        except Exception as exc:
            return ExecutionReceipt(ok=False, qty_out=0, gas_used=0, error=str(exc))

    # ------------------------------------------------------------------
    # State-changing execution  (actual fork transaction)
    # ------------------------------------------------------------------

    def execute(
        self,
        router: Address,
        qty_in: int,
        min_out: int,
        token_path: list[str],
        recipient: Address,
        deadline: int = 2 ** 32 - 1
    ) -> ExecutionReceipt:
        """
        Execute swapExactTokensForTokens on the fork.

        Prepare state first using ForkedChain cheatcodes:
            chain.fund_eth(recipient, ...)
            chain.act_as(recipient)
        then call this method, then optionally chain.restore(snapshot).
        """
        calldata = _SEL_SWAP_EXACT + abi_encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [qty_in, min_out, token_path, recipient.checksum, deadline],
        )
        tx = {"to": router.checksum, "from": recipient.checksum,
              "data": "0x" + calldata.hex()}
        try:
            tx_hash = self._chain.broadcast(tx)
            rec = self._chain.receipt(tx_hash)
            gas = rec.get("gasUsed", 0) if rec else 0
            return ExecutionReceipt(ok=True, qty_out=0, gas_used=gas,
                                    error=None,
                                    logs=rec.get("logs", []) if rec else [])
        except Exception as exc:
            return ExecutionReceipt(ok=False, qty_out=0, gas_used=0, error=str(exc))

    # ------------------------------------------------------------------
    # Cross-check helper
    # ------------------------------------------------------------------

    def cross_check(
        self,
        pool: PoolState,
        qty_in: int,
        selling: Token
    ) -> dict:
        """
        Compare our offline AMM formula against getAmountsOut from the fork.
        A non-zero difference indicates stale cached reserves or a formula bug.
        """
        local = pool.out_for_in(qty_in, selling)
        buying = pool.right if selling == pool.left else pool.left
        receipt = self.quote_via_router(
            router=_V2_ROUTER_ADDR,
            qty_in=qty_in,
            token_path=[selling.address.checksum, buying.address.checksum],
            caller=_NULL_SENDER
        )
        forked = receipt.qty_out
        delta = abs(local - forked)
        return dict(local=local, forked=forked, delta=delta, match=(local == forked))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fetch_reserves(self, pair_addr: Address) -> tuple[int, int]:
        raw = self._chain.eth_call(
            {"to": pair_addr.checksum, "data": "0x" + _SEL_RESERVES.hex()}
        )
        r0, r1, _ = abi_decode(["uint112", "uint112", "uint32"], raw)
        return r0, r1
