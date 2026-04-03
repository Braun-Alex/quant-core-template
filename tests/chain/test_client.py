import pytest
from unittest.mock import MagicMock, patch
from src.chain.client import ChainClient
from src.core.types import Address
from src.chain.errors import (
    RPCError,
    InsufficientFunds,
    NonceTooLow,
    ReplacementUnderpriced
)


class TestChainClient:
    @pytest.fixture
    def mock_w3(self):
        """
        Creates a mocked Web3 instance.
        By default, it simulates a successful connection.
        """
        with patch("src.chain.client.Web3") as mocked:
            w3_instance = mocked.return_value
            w3_instance.is_connected.return_value = True
            yield w3_instance

    @pytest.fixture
    def client(self, mock_w3):
        """
        Initializes ChainClient with two fake RPC URLs and a low retry count for speed.
        """
        return ChainClient(["http://rpc1.com", "http://rpc2.com"], max_retries=2)

    # --- Connection & Retry logic ---

    def test_rpc_fallback_mechanism(self, mock_w3):
        """
        Scenario: the first RPC URL is down, but the second one is active.
        Expectation: ChainClient should automatically switch to the reachable provider.
        """
        with patch("src.chain.client.Web3") as mocked:
            # Setup: first provider fails connection check, second succeeds
            w3_failed = MagicMock()
            w3_failed.is_connected.return_value = False
            w3_success = MagicMock()
            w3_success.is_connected.return_value = True

            mocked.side_effect = [w3_failed, w3_success]
            client = ChainClient(["url1", "url2"])

            # Verify that the internal _get_w3 method returns the working instance
            assert client._get_w3() == w3_success

    def test_max_retries_and_exponential_backoff(self, client, mock_w3):
        """
        Scenario: all RPC calls fail repeatedly.
        Expectation: the client should attempt the call exactly 'max_retries' times
        before propagating the exception.
        """
        mock_w3.eth.get_balance.side_effect = Exception("Node timeout")
        addr = Address.from_string("0x" + "1" * 40)

        with pytest.raises(Exception, match="Node timeout"):
            client.get_balance(addr)

        # Count internal retries (should match the max_retries parameter)
        assert mock_w3.eth.get_balance.call_count == 2

    # --- RPC error mapping ---

    @pytest.mark.parametrize("rpc_error, expected_exception", [
        ("Insufficient funds for gas * price + value", InsufficientFunds),
        ("Nonce too low: next expected 5, got 4", NonceTooLow),
        ("Replacement transaction underpriced", ReplacementUnderpriced),
        ("Some unknown RPC error", RPCError),
    ])
    def test_send_transaction_error_classification(self, client, mock_w3, rpc_error, expected_exception):
        """
        Scenario: the network returns a raw error string during transaction broadcast.
        Expectation: the client must parse the string and raise a domain-specific exception.
        """
        mock_w3.eth.send_raw_transaction.side_effect = Exception(rpc_error)

        with pytest.raises(expected_exception):
            client.send_transaction(b"signed_binary_data")

    # --- Data retrieval ---

    def test_get_gas_price_estimation(self, client, mock_w3):
        """
        Scenario: fetching the current network gas state.
        Expectation: returns a GasPrice object with calculated priority tiers (low/med/high).
        """
        # Mock latest block base fee and the priority fee estimate
        mock_w3.eth.get_block.return_value = {"baseFeePerGas": 40_000_000_000}   # 40 Gwei
        mock_w3.eth.max_priority_fee = 1_500_000_000   # 1.5 Gwei

        gp = client.get_gas_price()
        assert gp.base_fee == 40_000_000_000
        assert gp.priority_fee_medium == 1_500_000_000
        assert gp.priority_fee_high == 3_000_000_000   # High should be 2x medium
