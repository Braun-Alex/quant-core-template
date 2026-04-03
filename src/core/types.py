from dataclasses import dataclass
from typing import Optional, Any
from decimal import Decimal
from eth_utils import is_address, to_checksum_address


@dataclass(frozen=True)
class Address:
    """Ethereum address with validation and checksumming."""
    value: str

    def __post_init__(self):
        if not isinstance(self.value, str):
            raise ValueError("Address must be string")

        if not is_address(self.value):
            raise ValueError(f"Invalid Ethereum address: {self.value}")

        object.__setattr__(self, "value", to_checksum_address(self.value))

    @classmethod
    def from_string(cls, s: str) -> "Address":
        return cls(s)

    @property
    def checksum(self) -> str:
        return self.value

    @property
    def lower(self) -> str:
        return self.value.lower()

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Address):
            return self.lower == other.lower
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.lower)

    def __repr__(self) -> str:
        return f"Address({self.value})"


@dataclass(frozen=True)
class TokenAmount:
    """
    Represents a token amount with proper decimal handling.

    Internally stores raw integer (wei-equivalent).
    Provides human-readable formatting.
    """
    raw: int   # Raw amount (e.g., wei)
    decimals: int   # Token decimals (e.g., 18 for ETH, 6 for USDC)
    symbol: Optional[str] = None

    @classmethod
    def from_human(cls, amount: str | Decimal | int, decimals: int, symbol: Optional[str] = None) -> "TokenAmount":
        """Create from human-readable amount (e.g., '1.5' ETH)."""
        if isinstance(amount, str):
            amount = Decimal(amount)
        elif isinstance(amount, float):
            raise ValueError("Floating point numbers are prohibited in TokenAmount")
        elif isinstance(amount, int):
            amount = Decimal(amount)

        raw = int(amount * (Decimal(10) ** decimals))
        return cls(raw=raw, decimals=decimals, symbol=symbol)

    @property
    def human(self) -> Decimal:
        """Returns human-readable decimal."""
        return Decimal(self.raw) / Decimal(10 ** self.decimals)

    def __add__(self, other: "TokenAmount") -> "TokenAmount":
        # Validate same decimals
        if self.decimals != other.decimals:
            raise ValueError(f"Decimals mismatch: {self.decimals} != {other.decimals}")
        return TokenAmount(self.raw + other.raw, self.decimals, self.symbol or other.symbol)

    def __mul__(self, factor: int | Decimal) -> "TokenAmount":
        if isinstance(factor, float):
            raise ValueError("Floating point multiplication prohibited")
        new_raw = int(Decimal(self.raw) * Decimal(factor))
        return TokenAmount(new_raw, self.decimals, self.symbol)

    def __str__(self) -> str:
        return f"{self.human}{self.symbol or ''}"


@dataclass(frozen=True, eq=False)
class Token:
    """
    Represents an ERC-20 token with its on-chain metadata.

    Identity is by address only — two Token instances at the same address
    are equal regardless of symbol/decimals (those are metadata, not identity).
    We use eq=False to override the dataclass-generated __eq__ and define our own.
    """
    address: Address
    symbol: str
    decimals: int

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Token):
            return self.address == other.address   # Delegates to Address.__eq__ (case-insensitive)
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.address.lower)

    def __repr__(self) -> str:
        return f"Token({self.symbol}, {self.address.checksum})"


@dataclass
class TransactionRequest:
    """A transaction ready to be signed."""
    to: Address
    value: TokenAmount
    data: bytes = b""
    nonce: Optional[int] = None
    gas_limit: Optional[int] = None
    max_fee_per_gas: Optional[int] = None
    max_priority_fee: Optional[int] = None
    chain_id: int = 1

    def to_dict(self) -> dict:
        """Convert to web3-compatible dict."""
        d = {
            "to": self.to.checksum,
            "value": self.value.raw,
            "data": "0x" + self.data.hex() if self.data else "0x",
            "chainId": self.chain_id,
        }
        if self.nonce is not None:
            d["nonce"] = self.nonce
        if self.gas_limit is not None:
            d["gas"] = self.gas_limit
        if self.max_fee_per_gas is not None:
            d["maxFeePerGas"] = self.max_fee_per_gas
        if self.max_priority_fee is not None:
            d["maxPriorityFeePerGas"] = self.max_priority_fee
        return d


@dataclass
class TransactionReceipt:
    """Parsed transaction receipt."""
    tx_hash: str
    block_number: int
    status: bool   # True = success
    gas_used: int
    effective_gas_price: int
    logs: list

    @property
    def tx_fee(self) -> TokenAmount:
        """Returns transaction fee as TokenAmount."""
        return TokenAmount(
            raw=self.gas_used * self.effective_gas_price,
            decimals=18,
            symbol="ETH"
        )

    @classmethod
    def from_web3(cls, receipt: dict) -> "TransactionReceipt":
        """Parse from web3 receipt dict."""
        tx_hash = receipt["transactionHash"]
        if isinstance(tx_hash, bytes):
            tx_hash = "0x" + tx_hash.hex()
        return cls(
            tx_hash=tx_hash,
            block_number=receipt["blockNumber"],
            status=bool(receipt.get("status", 0)),
            gas_used=receipt["gasUsed"],
            effective_gas_price=receipt.get("effectiveGasPrice", receipt.get("gasPrice", 0)),
            logs=receipt.get("logs", [])
        )
