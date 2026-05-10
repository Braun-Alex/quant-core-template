"""
Anvil fork DEX inventory provisioner for demo/simulation mode.

Problem
-------
Binance Demo Trading gives unlimited virtual CEX balance.
The DEX side has no equivalent - Arbitrum Mainnet requires real tokens,
and Arbitrum Sepolia has no reliable liquidity pools.

Solution
--------
Fork Arbitrum Mainnet with Anvil. The fork preserves all real pool states
and reserves (ARB/USDC pool, etc.) while allowing us to inject test tokens
into the hot wallet via Foundry's vm.deal() cheatcode (anvil_setBalance /
hardhat_setStorageAt). This gives:

  ✓ Realistic DEX prices (real mainnet reserves)
  ✓ Realistic slippage (real pool depth)
  ✓ Real Uniswap V2 contract behavior
  ✓ Unlimited test tokens (via cheatcodes)
  ✗ No real counterparty - fills simulate against the forked state

Combined with Binance Demo Trading on the CEX side, this creates the most
realistic possible simulation of CEX-DEX arbitrage without real risking.

Architecture
------------
  Binance Demo Trading  ← real-time mainnet orderbook, virtual fills
  Anvil fork (Arbitrum) ← real pool reserves, cheatcode token injection

Usage
-----
    # Start Anvil fork (Makefile target: make fork-arb)
    # anvil --fork-url $ARBITRUM_RPC_URL --port 8545

    provisioner = AnvilInventoryProvisioner(
        rpc_url="http://localhost:8545",
        wallet_address="0xYourAddress"
    )
    await provisioner.fund_demo_wallet()

Token storage slots (ERC-20 balances mapping)
----------------------------------------------
Most ERC-20 tokens use storage slot 0 for the balances mapping.
Exceptions:
  ARB   → slot 51  (OpenZeppelin ERC20Votes with EIP712 storage)
  USDC  → slot 9   (native USDC on Arbitrum, Circle's implementation)
  WETH  → slot 0   (standard WETH9)

If a token's slot is wrong, the balance write silently fails.
Use `find_storage_slot()` to detect the correct slot if needed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv
from web3 import Web3
from eth_abi import encode as abi_encode

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token registry for Arbitrum One
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrumToken:
    symbol: str
    address: str   # Checksummed
    decimals: int
    balance_slot: int   # Solidity storage slot for balances mapping

    @property
    def checksum(self) -> str:
        return Web3.to_checksum_address(self.address)


# Verified storage slots for Arbitrum One tokens
ARB_TOKENS = {
    "ARB": ArbitrumToken(
        symbol="ARB",
        address="0x912CE59144191C1204E64559FE8253a0e49E6548",
        decimals=18,
        balance_slot=51   # ERC20Votes: slot 51 (after permit/nonce storage)
    ),
    "USDC": ArbitrumToken(
        symbol="USDC",
        address="0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
        decimals=6,
        balance_slot=9   # Circle USDC: slot 9
    ),
    "WETH": ArbitrumToken(
        symbol="WETH",
        address="0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
        decimals=18,
        balance_slot=0   # Standard WETH9: slot 0
    ),
    "MAGIC": ArbitrumToken(
        symbol="MAGIC",
        address="0x539bdE0d7Dbd336b79148AA742883198BBF60342",
        decimals=18,
        balance_slot=0
    ),
    "GMX": ArbitrumToken(
        symbol="GMX",
        address="0xfc5A1A6EB076a2C7aD06eD22C90d7E710E35ad0a",
        decimals=18,
        balance_slot=0
    ),
    "PENDLE": ArbitrumToken(
        symbol="PENDLE",
        address="0x0c880f6761F1af8d9Aa9C466984b80DAb9a8c9e8",
        decimals=18,
        balance_slot=0
    )
}

# Known Uniswap V2 pool addresses on Arbitrum One
ARB_POOLS = {
    "ARB/USDC": "0x81FdAC61b65E58e0D7F3BA5c5b55C3FFE0753D5",
    "WETH/USDC": "0x905dfCD5649217c42684f23958568e533C711Aa3",
    "MAGIC/USDC": "0x6F9D20B0Dde6B2f50C46DF1A7e9E5E26d1AEA88",
    "GMX/USDC": "0x1aEEdD3727A6431b8F070C0aFaA81Cc74f273882"
}

ERC20_ABI = [
    {
        "name": "transfer",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}]
    }
]

WHALES = {
    "USDC": "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8",
    "WETH": "0x82aF49447D8a07e3bd95BD0d56f35241523fBab1",
    "ARB": "0x912CE59144191C1204E64559FE8253a0e49E6548"
}


# ---------------------------------------------------------------------------
# Demo inventory amounts
# ---------------------------------------------------------------------------

@dataclass
class DemoInventory:
    """Target token balances for demo/simulation mode."""
    eth_wei: int = int(0.3 * 10**18)   # 0.3 ETH for gas
    token_amounts: dict[str, int] = field(default_factory=lambda: {
        "ARB": int(20_000 * 10**18),   # 20,000 ARB
        "USDC": int(2_000 * 10**6),   # $2,000 USDC
        "WETH": int(0.3 * 10**18),   # 0.3 WETH
        "MAGIC": int(5_000 * 10**18),   # 5,000 MAGIC
        "GMX": int(100 * 10 ** 18),   # 100 GMX
        "PENDLE": int(1_000 * 10**18)   # 1,000 PENDLE
    })


# ---------------------------------------------------------------------------
# Provisioner
# ---------------------------------------------------------------------------

class AnvilInventoryProvisioner:
    """
    Funds a wallet with test tokens on a local Anvil fork of Arbitrum.

    Requires:
      - Anvil running: anvil --fork-url $ARBITRUM_RPC_URL --port 8545
      - The fork must include the token contracts (any recent block works)

    Usage:

        p = AnvilInventoryProvisioner(
            rpc_url="http://localhost:8545",
            wallet_address="0xYourHotWallet"
        )
        p.fund_demo_wallet()   # Synchronous
    """

    def __init__(
        self,
        rpc_url: str,
        wallet_address: str,
        inventory: Optional[DemoInventory] = None
    ) -> None:
        self._rpc = rpc_url
        self._wallet = Web3.to_checksum_address(wallet_address)
        self._inv = inventory or DemoInventory()
        self._slot_cache: dict[str, int] = {}
        self._w3 = Web3(Web3.HTTPProvider(rpc_url))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fund_demo_wallet(self, tokens: Optional[list[str]] = None) -> dict[str, bool]:
        """
        Fund the demo wallet with ETH and ERC-20 tokens.

        Parameters
        ----------
        tokens : list of token symbols to fund (default: all in DemoInventory)

        Returns a dict of {symbol: success} for each token funded.
        """
        if not self._is_anvil():
            raise RuntimeError(
                f"RPC at {self._rpc} is not an Anvil fork. "
                "Start Anvil with: anvil --fork-url $ARBITRUM_RPC_URL --port 8545"
            )

        results: dict[str, bool] = {}

        # 1. Fund ETH for gas
        eth_ok = self._fund_eth(self._inv.eth_wei)
        results["ETH"] = eth_ok
        if eth_ok:
            log.info(
                "Funded %s with %s ETH (gas)",
                self._wallet[:10],
                self._inv.eth_wei / 10**18
            )

        # 2. Fund ERC-20 tokens
        target_tokens = tokens or list(self._inv.token_amounts.keys())
        for symbol in target_tokens:
            token = ARB_TOKENS.get(symbol)
            if token is None:
                log.warning("Unknown token: %s - skipping", symbol)
                results[symbol] = False
                continue

            amount_raw = self._inv.token_amounts.get(symbol, 0)
            if amount_raw == 0:
                results[symbol] = False
                continue

            ok = self._fund_erc20_impersonation(token, amount_raw)
            results[symbol] = ok
            amount_human = amount_raw / 10**token.decimals
            if ok:
                log.info(
                    "Funded %s with %.2f %s",
                    self._wallet[:10], amount_human, symbol
                )
            else:
                log.warning(
                    "Failed to fund %s - slot %d may be wrong; "
                    "use find_storage_slot() to detect",
                    symbol, token.balance_slot
                )

        return results

    def verify_balances(self, tokens: Optional[list[str]] = None) -> dict[str, Decimal]:
        """
        Read and return actual on-chain balances for the wallet.
        Call after fund_demo_wallet() to confirm writes succeeded.
        """
        balances: dict[str, Decimal] = {}

        # ETH balance
        eth_raw = self._w3.eth.get_balance(self._wallet)
        balances["ETH"] = Decimal(eth_raw) / Decimal(10**18)

        target_tokens = tokens or list(ARB_TOKENS.keys())
        for symbol in target_tokens:
            token = ARB_TOKENS.get(symbol)
            if token is None:
                continue
            raw = self._read_erc20_balance(token)
            balances[symbol] = Decimal(raw) / Decimal(10**token.decimals)

        return balances

    def find_storage_slot(self, token_address: str, max_slots: int = 100) -> Optional[int]:
        """
        Detect the correct storage slot for a token's balances mapping
        by writing a sentinel value to each slot and reading back.

        Use this if a token's balance write silently fails.
        """
        addr = Web3.to_checksum_address(token_address)
        sentinel = 12345 * 10**18   # Distinctive amount unlikely to exist naturally

        for slot in range(max_slots):
            try:
                self._write_storage(addr, slot, sentinel)
                actual = self._read_erc20_balance_at(addr)
                if actual == sentinel:
                    log.info("Found balance slot %d for %s", slot, addr)
                    # Restore to zero
                    self._write_storage(addr, slot, 0)
                    return slot
                # Restore to zero before next attempt
                self._write_storage(addr, slot, 0)
            except Exception as exc:
                log.info("Slot %d probe failed: %s", slot, exc)
                continue

        log.warning("Could not find balance slot for %s in first %d slots", addr, max_slots)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _is_anvil(self) -> bool:
        """Check that the RPC is an Anvil fork (not a real node)."""
        try:
            resp = self._w3.provider.make_request("anvil_nodeInfo", [])
            return "result" in resp
        except Exception:
            # Try hardhat fallback
            try:
                resp = self._w3.provider.make_request("hardhat_metadata", [])
                return "result" in resp
            except Exception:
                return False

    def _fund_eth(self, amount_wei: int) -> bool:
        try:
            self._w3.provider.make_request(
                "anvil_setBalance",
                [self._wallet, hex(amount_wei)]
            )
            return True
        except Exception as exc:
            log.error("ETH funding failed: %s", exc)
            return False

    def _impersonate(self, address: str):
        self._w3.provider.make_request("anvil_impersonateAccount", [address])
        self._w3.provider.make_request(
            "anvil_setBalance",
            [address, hex(10 ** 20)]
        )

    def _fund_erc20_impersonation(self, token: ArbitrumToken, amount: int) -> bool:
        try:
            whale = WHALES.get(token.symbol)
            if not whale:
                raise RuntimeError(f"No whale for {token.symbol}")

            whale = Web3.to_checksum_address(whale)

            self._impersonate(whale)

            contract = self._w3.eth.contract(
                address=token.checksum,
                abi=ERC20_ABI
            )

            tx = contract.functions.transfer(
                self._wallet,
                amount
            ).transact({"from": whale})

            self._w3.eth.wait_for_transaction_receipt(tx)

            return True

        except Exception as exc:
            log.error("Impersonation funding failed for %s: %s", token.symbol, exc)
            return False

    def _fund_erc20(self, token: ArbitrumToken, amount_raw: int) -> bool:
        """Write amount_raw directly into the token's balances[wallet] slot."""
        try:
            slot = self._slot_cache.get(token.symbol)

            if slot is None:
                slot = self.find_storage_slot(token.address)
                if slot is None:
                    raise RuntimeError(f"Slot not found for {token.symbol}")
                self._slot_cache[token.symbol] = slot

            self._write_storage(token.checksum, slot, amount_raw)
            return True

        except Exception as exc:
            log.error("ERC-20 funding failed for %s: %s", token.symbol, exc)
            return False

    def _write_storage(
        self, contract: str, slot: int, value: int
    ) -> None:
        """Write a value to an ERC-20 storage slot for self._wallet."""
        # Storage key for mapping(address => uint256):
        # keccak256(abi.encode(address, slot))
        slot_key = Web3.keccak(
            abi_encode(["address", "uint256"], [self._wallet, slot])
        )
        value_hex = "0x" + format(value, "064x")

        # Try anvil_setStorageAt first, fall back to hardhat_setStorageAt
        for method in ("anvil_setStorageAt", "hardhat_setStorageAt"):
            try:
                self._w3.provider.make_request(
                    method,
                    [contract, slot_key.hex(), value_hex]
                )
                return
            except Exception:
                continue
        raise RuntimeError("Neither anvil_setStorageAt nor hardhat_setStorageAt succeeded")

    def _read_erc20_balance(self, token: ArbitrumToken) -> int:
        return self._read_erc20_balance_at(token.checksum)

    def _read_erc20_balance_at(self, contract: str) -> int:
        """Read ERC-20 balance of self._wallet via eth_call."""
        # balanceOf(address) selector
        selector = bytes.fromhex("70a08231")
        calldata = selector + abi_encode(["address"], [self._wallet])
        try:
            raw = self._w3.eth.call({
                "to": contract,
                "data": "0x" + calldata.hex()
            })
            if len(raw) >= 32:
                return int.from_bytes(raw[:32], "big")
        except Exception:
            pass
        return 0
