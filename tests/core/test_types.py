import pytest
from src.core.types import Address, TokenAmount, Token, TransactionRequest, TransactionReceipt


# --- Address tests ---

class TestAddress:
    def test_invalid_address_raises_error(self):
        """Verify that invalid address formats raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            Address("Invalid-address")

        with pytest.raises(ValueError, match="Address must be string"):
            Address(123456789)   # Type: ignore

    def test_address_case_insensitivity(self):
        """Verify that addresses are equal regardless of casing."""
        addr_lower = "0x71c7656ec7ab88b098defb751b7401b5f6d8976f"
        addr_upper = "0x71C7656EC7AB88B098DEFB751B7401B5F6D8976F"

        a1 = Address(addr_lower)
        a2 = Address(addr_upper)

        assert a1 == a2
        assert a1.lower == a2.lower
        assert hash(a1) == hash(a2)


# --- TokenAmount tests ---

class TestTokenAmount:
    def test_from_human_conversion(self):
        """Verify correct conversion from human-readable string to raw (wei) integer."""
        amount = TokenAmount.from_human("1.5", 18)
        assert amount.raw == 1_500_000_000_000_000_000
        assert amount.decimals == 18

    def test_adding_different_decimals_raises_error(self):
        """Verify that adding amounts with different decimals raises a ValueError."""
        eth_amount = TokenAmount.from_human("1", 18)
        usdc_amount = TokenAmount.from_human("1", 6)

        with pytest.raises(ValueError, match="Decimals mismatch"):
            _ = eth_amount + usdc_amount

    def test_float_prohibition(self):
        """Ensure floating point numbers are prohibited to prevent precision issues."""
        # Creation via float
        with pytest.raises(ValueError, match="Floating point numbers are prohibited"):
            TokenAmount.from_human(1.5, 18)   # Type: ignore

        # Multiplication by float
        amount = TokenAmount.from_human("1", 18)
        with pytest.raises(ValueError, match="Floating point multiplication prohibited"):
            _ = amount * 1.5   # Type: ignore

    def test_arithmetic_precision_no_internal_floats(self):
        """Verify arithmetic uses Decimal/int to maintain exact precision."""
        amount = TokenAmount.from_human("0.000000000000000001", 18)   # 1 wei
        multiplied = amount * 3
        assert multiplied.raw == 3
        assert isinstance(multiplied.raw, int)


# --- Token tests ---

class TestToken:
    @pytest.fixture
    def addr_1(self):
        return Address("0x71C7656EC7AB88B098DEFB751B7401B5F6D8976F")

    @pytest.fixture
    def addr_2(self):
        return Address("0x0000000000000000000000000000000000000000")

    def test_token_equality_by_address_only(self, addr_1):
        """Tokens should be equal if their addresses match, ignoring other metadata."""
        token_a = Token(address=addr_1, symbol="DAI", decimals=18)
        token_b = Token(address=addr_1, symbol="FAKE", decimals=6)

        assert token_a == token_b
        assert hash(token_a) == hash(token_b)

    def test_token_inequality_different_addresses(self, addr_1, addr_2):
        """Tokens with different addresses must not be equal."""
        token_a = Token(address=addr_1, symbol="ETH", decimals=18)
        token_b = Token(address=addr_2, symbol="ETH", decimals=18)

        assert token_a != token_b
        assert hash(token_a) != hash(token_b)


# --- Transaction tests ---

class TestTransactions:
    def test_transaction_request_to_dict(self):
        """Verify transaction object converts to a Web3-compatible dictionary correctly."""
        addr = Address("0x71C7656EC7AB88B098DEFB751B7401B5F6D8976F")
        amount = TokenAmount.from_human("1", 18)
        tx = TransactionRequest(to=addr, value=amount, nonce=5, chain_id=1)

        d = tx.to_dict()
        assert d["to"] == addr.checksum
        assert d["value"] == 10 ** 18
        assert d["nonce"] == 5
        assert d["data"] == "0x"

    def test_transaction_receipt_fee_calculation(self):
        """Verify that tx_fee calculation handles gas parameters correctly."""
        receipt_data = {
            "transactionHash": b"\x01" * 32,
            "blockNumber": 100,
            "status": 1,
            "gasUsed": 21000,
            "effectiveGasPrice": 20_000_000_000,   # 20 gwei
        }
        receipt = TransactionReceipt.from_web3(receipt_data)

        expected_fee_raw = 21000 * 20_000_000_000
        assert receipt.tx_fee.raw == expected_fee_raw
        assert receipt.tx_fee.symbol == "ETH"
        assert receipt.tx_fee.decimals == 18
