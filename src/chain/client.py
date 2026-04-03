import time
from dataclasses import dataclass
from typing import Optional, cast

from web3 import Web3
from web3.types import (
    TxParams,
    BlockIdentifier,
    Wei,
    ChecksumAddress,
    HexStr
)

from src.core.types import Address, TokenAmount, TransactionRequest, TransactionReceipt
from .errors import (
    RPCError,
    TransactionFailed,
    InsufficientFunds,
    NonceTooLow,
    ReplacementUnderpriced,
)


@dataclass
class GasPrice:
    """Current gas price information."""
    base_fee: int
    priority_fee_low: int
    priority_fee_medium: int
    priority_fee_high: int

    def get_max_fee(self, priority: str = "medium", buffer: float = 1.2) -> int:
        """Calculate maxFeePerGas with buffer for base fee increase."""
        prio_map = {
            "low": self.priority_fee_low,
            "medium": self.priority_fee_medium,
            "high": self.priority_fee_high
        }
        prio = prio_map.get(priority, self.priority_fee_medium)
        return int(self.base_fee * buffer + prio)


class ChainClient:
    def __init__(self, rpc_urls: list[str], timeout: int = 30, max_retries: int = 3):
        self.rpc_urls = rpc_urls
        self.timeout = timeout
        self.max_retries = max_retries
        self._w3s = [
            Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": timeout}))
            for url in rpc_urls
        ]

    def _get_w3(self) -> Web3:
        for w3 in self._w3s:
            if w3.is_connected():
                return w3
        raise RPCError("All RPC endpoints are unreachable")

    def _retry(self, func):
        for attempt in range(self.max_retries):
            try:
                return func()
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
        raise RPCError("Max retries exceeded")

    def get_chain_id(self) -> int:
        return self._get_w3().eth.chain_id

    def get_balance(self, address: Address) -> TokenAmount:
        def call():
            w3 = self._get_w3()
            checksum_addr: ChecksumAddress = cast(ChecksumAddress, address.checksum)
            bal: int = w3.eth.get_balance(checksum_addr)
            return TokenAmount(raw=bal, decimals=18, symbol="ETH")
        return self._retry(call)

    def get_nonce(self, address: Address, block: BlockIdentifier = "pending") -> int:
        def call():
            w3 = self._get_w3()
            checksum_addr: ChecksumAddress = cast(ChecksumAddress, address.checksum)
            return w3.eth.get_transaction_count(checksum_addr, block)
        return self._retry(call)

    def get_gas_price(self) -> GasPrice:
        """Returns current gas price info (base fee, priority fee estimates)."""
        def call():
            w3 = self._get_w3()
            block = w3.eth.get_block("latest")
            base_fee: int = block.get("baseFeePerGas", 0)

            try:
                priority_fee: Wei = getattr(w3.eth, "max_priority_fee", None) or Wei(2_000_000_000)
            except Exception:
                priority_fee = Wei(2_000_000_000)

            return GasPrice(
                base_fee=base_fee,
                priority_fee_low=int(priority_fee) // 2,
                priority_fee_medium=int(priority_fee),
                priority_fee_high=int(priority_fee) * 2,
            )
        return self._retry(call)

    def estimate_gas(self, tx: TransactionRequest) -> int:
        def call():
            w3 = self._get_w3()
            tx_dict = tx.to_dict()
            params: TxParams = cast(TxParams, tx_dict)
            return w3.eth.estimate_gas(params)
        return self._retry(call)

    def send_transaction(self, signed_tx: bytes) -> str:
        """Send and return tx hash. Does NOT wait for confirmation."""
        def call():
            w3 = self._get_w3()
            try:
                tx_hash = w3.eth.send_raw_transaction(signed_tx)
                return tx_hash.hex() if isinstance(tx_hash, (bytes, bytearray)) else str(tx_hash)
            except Exception as err:
                msg = str(err).lower()
                if "nonce too low" in msg:
                    raise NonceTooLow(str(err)) from err
                if "insufficient funds" in msg or "insufficient balance" in msg:
                    raise InsufficientFunds(str(err)) from err
                if "underpriced" in msg or "replacement" in msg:
                    raise ReplacementUnderpriced(str(err)) from err
                raise RPCError(f"Failed to send transaction: {err}") from err
        return self._retry(call)

    def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 120,
        poll_interval: float = 1.0,
    ) -> TransactionReceipt:
        """Wait for transaction confirmation."""
        def call():
            w3 = self._get_w3()
            hash_str: HexStr = cast(HexStr, tx_hash)
            receipt = w3.eth.wait_for_transaction_receipt(
                hash_str,
                timeout=timeout,
                poll_latency=poll_interval
            )
            parsed = TransactionReceipt.from_web3(dict(receipt))

            if not parsed.status:
                raise TransactionFailed(tx_hash, parsed)

            return parsed
        return self._retry(call)

    def get_transaction(self, tx_hash: str) -> dict:
        def call():
            w3 = self._get_w3()
            hash_str: HexStr = cast(HexStr, tx_hash)
            tx = w3.eth.get_transaction(hash_str)
            return dict(tx)
        return self._retry(call)

    def get_receipt(self, tx_hash: str) -> Optional[TransactionReceipt]:
        def call():
            w3 = self._get_w3()
            hash_str: HexStr = cast(HexStr, tx_hash)
            receipt = w3.eth.get_transaction_receipt(hash_str)
            if receipt is None:
                return None
            return TransactionReceipt.from_web3(dict(receipt))
        return self._retry(call)

    def call(self, tx: TransactionRequest, block: BlockIdentifier = "latest") -> bytes:
        """Simulate transaction without sending."""
        def call_inside():
            w3 = self._get_w3()
            tx_dict = tx.to_dict()
            params: TxParams = cast(TxParams, tx_dict)
            return w3.eth.call(params, block)
        return self._retry(call_inside)
