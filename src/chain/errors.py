from typing import Optional
from src.core.types import TransactionReceipt


class ChainError(Exception):
    """Base exception for all chain-related errors."""
    pass


class RPCError(ChainError):
    """Raised when RPC request fails."""
    def __init__(self, message: str, code: Optional[int] = None):
        self.code = code
        super().__init__(message)


class TransactionFailed(ChainError):
    """Raised when transaction reverted on-chain."""
    def __init__(self, tx_hash: str, receipt: TransactionReceipt):
        self.tx_hash = tx_hash
        self.receipt = receipt
        super().__init__(f"Transaction {tx_hash} reverted (status = False)")


class InsufficientFunds(ChainError):
    """Raised when account has insufficient balance for transaction."""
    def __init__(self, message: str = "Insufficient funds for transaction"):
        super().__init__(message)


class NonceTooLow(ChainError):
    """Raised when nonce is too low (transaction already mined or pending)."""
    def __init__(self, message: str = "Nonce too low"):
        super().__init__(message)


class ReplacementUnderpriced(ChainError):
    """Raised when trying to replace transaction with insufficient gas price."""
    def __init__(self, message: str = "Replacement transaction underpriced"):
        super().__init__(message)


class GasEstimationError(ChainError):
    """Raised when gas estimation fails."""
    pass


class ContractLogicError(ChainError):
    """Raised when smart contract reverts with custom error."""
    def __init__(self, message: str, revert_reason: Optional[str] = None):
        self.revert_reason = revert_reason
        super().__init__(message or f"Contract reverted: {revert_reason}")
