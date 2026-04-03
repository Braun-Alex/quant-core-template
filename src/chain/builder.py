from src.core.types import Address, TokenAmount, TransactionRequest
from src.core.wallet import WalletManager
from .client import ChainClient
from eth_account.datastructures import SignedTransaction
from .errors import TransactionFailed


class TransactionBuilder:
    """
    Fluent builder for transactions.

    Usage:
        tx = (TransactionBuilder(client, wallet)
            .to(recipient)
            .value(TokenAmount.from_human("0.1", 18))
            .data(calldata)
            .with_gas_estimate()
            .with_gas_price("high")
            .build())
    """
    def __init__(self, client: ChainClient, wallet: WalletManager):
        self.client = client
        self.wallet = wallet
        self._to: Address | None = None
        self._value: TokenAmount | None = None
        self._data: bytes = b""
        self._nonce: int | None = None
        self._gas_limit: int | None = None
        self._max_fee_per_gas: int | None = None
        self._max_priority_fee: int | None = None

        self._chain_id = self.client.get_chain_id()

    def to(self, address: Address):
        self._to = address
        return self

    def value(self, amount: TokenAmount):
        self._value = amount
        return self

    def data(self, calldata: bytes):
        self._data = calldata
        return self

    def nonce(self, nonce: int):
        """Explicit nonce (for replacement or batch)."""
        self._nonce = nonce
        return self

    def gas_limit(self, limit: int):
        self._gas_limit = limit
        return self

    def with_gas_estimate(self, buffer: float = 1.2):
        """Estimate gas and set limit with buffer."""
        if not self._to:
            raise ValueError("Error: to() must be set before gas estimation")
        temp_tx = TransactionRequest(
            to=self._to,
            value=self._value or TokenAmount(0, 18),
            data=self._data,
            chain_id=self._chain_id
        )
        gas = self.client.estimate_gas(temp_tx)
        self._gas_limit = int(gas * buffer)
        return self

    def with_gas_price(self, priority: str = "medium"):
        """Set gas price based on current network conditions."""
        gp = self.client.get_gas_price()
        self._max_fee_per_gas = gp.get_max_fee(priority)
        prio_map = {"low": gp.priority_fee_low, "medium": gp.priority_fee_medium, "high": gp.priority_fee_high}
        self._max_priority_fee = prio_map.get(priority, gp.priority_fee_medium)
        return self

    def build(self) -> TransactionRequest:
        """Validate and return transaction request."""
        if not self._to:
            raise ValueError("Error: to() is required")
        if self._value is None:
            self._value = TokenAmount(0, 18)
        if self._nonce is None:
            self._nonce = self.client.get_nonce(Address.from_string(self.wallet.address))
        if self._gas_limit is None or self._max_fee_per_gas is None:
            raise ValueError("Call with_gas_estimate() and with_gas_price() before build()")

        return TransactionRequest(
            to=self._to,
            value=self._value,
            data=self._data,
            nonce=self._nonce,
            gas_limit=self._gas_limit,
            max_fee_per_gas=self._max_fee_per_gas,
            max_priority_fee=self._max_priority_fee,
            chain_id=self._chain_id
        )

    def build_and_sign(self) -> SignedTransaction:
        """Build, sign, and return ready-to-send transaction."""
        tx = self.build()
        return self.wallet.sign_transaction(tx.to_dict())

    def send(self) -> str:
        """Build, sign, send, return tx hash."""
        signed = self.build_and_sign()
        return self.client.send_transaction(signed.raw_transaction)

    def send_and_wait(self, timeout: int = 120):
        """Build, sign, send, wait for confirmation."""
        tx_hash = self.send()
        receipt = self.client.wait_for_receipt(tx_hash, timeout)
        if not receipt.status:
            raise TransactionFailed(tx_hash, receipt)
        return receipt
