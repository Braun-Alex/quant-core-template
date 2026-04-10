"""
Live mempool surveillance for pending DEX swap transactions.

How it works
------------
1. A WebSocket connection subscribes to newPendingTransactions.
2. Each hash is resolved to a full transaction via eth_getTransactionByHash.
3. The calldata's 4-byte selector is matched against a registry of known
   DEX function signatures.
4. Matched calldata is ABI-decoded into a PendingSwap record.
5. The caller-supplied handler is invoked for every decoded swap.

Threading model
---------------
All I/O runs inside an asyncio event loop. Each incoming hash spawns a
short-lived Task so that slow RPC calls do not stall the subscription loop.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal

from eth_abi import decode as abi_decode
from web3 import AsyncWeb3

from src.core.types import Address

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signature registry
# ---------------------------------------------------------------------------

#: Maps the 4-byte hex selector to (protocol_name, function_name)
KNOWN_SELECTORS: dict[str, tuple[str, str]] = {
    "0x38ed1739": ("UniswapV2", "swapExactTokensForTokens"),
    "0x7ff36ab5": ("UniswapV2", "swapExactETHForTokens"),
    "0x18cbafe5": ("UniswapV2", "swapExactTokensForETH"),
    "0x5ae401dc": ("UniswapV3", "multicall")
}

# ABI parameter types per selector (calldata bytes *after* the 4-byte prefix)
_PARAM_TYPES: dict[str, tuple[str, ...]] = {
    "0x38ed1739": ("uint256", "uint256", "address[]", "address", "uint256"),
    "0x7ff36ab5": ("uint256", "address[]", "address", "uint256"),
    "0x18cbafe5": ("uint256", "uint256", "address[]", "address", "uint256"),
    "0x5ae401dc": ("uint256", "bytes[]")
}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class PendingSwap:
    """
    Decoded swap transaction observed in the mempool.

    Notes
    -----
    - *token_sold* / *token_bought* are None when native ETH is involved,
      because ETH has no ERC-20 contract address.
    - *expected_out* must be set externally (by querying the relevant pool)
      before reading *implied_slippage*.
    """

    tx_hash: str
    router_addr: str
    protocol: str
    fn_name: str
    token_sold: Address | None
    token_bought: Address | None
    qty_in: int
    min_qty_out: int
    deadline: int
    trader: Address
    gas_price: int
    expected_out: int | None = field(default=None)

    @property
    def implied_slippage(self) -> Decimal:
        """
        Inferred slippage tolerance: (expected − minimum) / expected.
        Requires *expected_out* to be populated first.
        """
        if self.expected_out is None:
            raise ValueError(
                "Set expected_out before reading implied_slippage. "
                "Use the relevant pool's out_for_in() method."
            )
        if self.expected_out == 0:
            raise ValueError("expected_out cannot be zero.")
        return Decimal(self.expected_out - self.min_qty_out) / Decimal(self.expected_out)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class MempoolWatcher:
    """
    Subscribes to pending transactions and invokes *handler* for each
    decoded swap.

    Parameters
    ----------
    ws_endpoint : WebSocket RPC URL (wss://...)
    handler     : callable that receives a PendingSwap
    """

    def __init__(self, ws_endpoint: str, handler: Callable[[PendingSwap], None]) -> None:
        self.ws_endpoint = ws_endpoint
        self.handler = handler

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def watch(self) -> None:
        """Open the WebSocket, subscribe, and process transactions indefinitely."""
        async with AsyncWeb3(AsyncWeb3.WebSocketProvider(self.ws_endpoint)) as w3:
            await w3.eth.subscribe("newPendingTransactions")
            async for msg in w3.socket.process_subscriptions():
                raw = msg.get("result") or (msg.get("params", {}).get("result"))
                if raw is None:
                    continue
                tx_hash = "0x" + raw.hex() if isinstance(raw, bytes) else str(raw)
                asyncio.create_task(self._process(w3, tx_hash))

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    async def _process(self, w3: AsyncWeb3, tx_hash: str) -> None:
        try:
            raw_tx = await w3.eth.get_transaction(tx_hash)
            if raw_tx is None:
                return
            swap = self.decode(dict(raw_tx))
            if swap is not None:
                self.handler(swap)
        except Exception as exc:
            log.debug("Skipping %s: %s", tx_hash, exc)

    def decode(self, tx: dict) -> PendingSwap | None:
        """
        Attempt to decode *tx* as a known swap.
        Returns None for non-swap transactions or unrecognized selectors.
        """
        payload: str | bytes = tx.get("input") or tx.get("data") or b""

        if isinstance(payload, bytes):
            if len(payload) < 4:
                return None
            selector = "0x" + payload[:4].hex()
            body = payload[4:]
        else:
            if len(payload) < 10:
                return None
            selector = payload[:10].lower()
            try:
                body = bytes.fromhex(payload[10:])
            except ValueError:
                return None

        if selector not in KNOWN_SELECTORS:
            return None

        protocol, fn_name = KNOWN_SELECTORS[selector]

        try:
            fields = self._decode_body(selector, body)
        except Exception as exc:
            log.debug("ABI decode failed for %s: %s", selector, exc)
            return None

        # ETH-in swaps carry their input amount in tx.value
        if selector == "0x7ff36ab5":
            fields["qty_in"] = tx.get("value", 0)

        raw_hash = tx.get("hash", "")
        tx_hash_str = "0x" + raw_hash.hex() if isinstance(raw_hash, bytes) else str(raw_hash)
        from_addr = tx.get("from") or "0x" + "0" * 40
        gas_price = tx.get("gasPrice") or tx.get("maxFeePerGas") or 0

        try:
            return PendingSwap(
                tx_hash = tx_hash_str,
                router_addr = str(tx.get("to") or ""),
                protocol = protocol,
                fn_name = fn_name,
                token_sold = Address(fields["token_sold"]) if fields["token_sold"] else None,
                token_bought = Address(fields["token_bought"]) if fields["token_bought"] else None,
                qty_in = fields["qty_in"],
                min_qty_out = fields["min_qty_out"],
                deadline = fields["deadline"],
                trader = Address(from_addr),
                gas_price = gas_price
            )
        except Exception as exc:
            log.debug("Could not build PendingSwap: %s", exc)
            return None

    @staticmethod
    def _decode_body(selector: str, body: bytes) -> dict:
        """ABI-decode the calldata body and normalise into a common dict shape."""
        types = _PARAM_TYPES.get(selector)
        if types is None:
            raise ValueError(f"No ABI registered for selector {selector!r}.")

        decoded = abi_decode(types, body)

        if selector == "0x7ff36ab5":   # swapExactETHForTokens
            min_out, path, _to, dl = decoded
            return dict(qty_in=0, min_qty_out=min_out, deadline=dl,
                        token_sold=None,
                        token_bought=path[-1] if path else None)

        if selector == "0x18cbafe5":   # swapExactTokensForETH
            qty_in, min_out, path, _to, dl = decoded
            return dict(qty_in=qty_in, min_qty_out=min_out, deadline=dl,
                        token_sold=path[0] if path else None,
                        token_bought=None)

        if selector == "0x38ed1739":   # swapExactTokensForTokens
            qty_in, min_out, path, _to, dl = decoded
            return dict(qty_in=qty_in, min_qty_out=min_out, deadline=dl,
                        token_sold=path[0] if path else None,
                        token_bought=path[-1] if path else None)

        if selector == "0x5ae401dc":   # UniswapV3 multicall
            dl, _inner = decoded
            return dict(qty_in=0, min_qty_out=0, deadline=dl,
                        token_sold=None, token_bought=None)

        raise ValueError(f"Unhandled selector: {selector!r}")
