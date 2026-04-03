import pytest
from unittest.mock import MagicMock

from src.chain.builder import TransactionBuilder
from src.chain.client import GasPrice
from src.core.wallet import WalletManager
from src.core.types import Address, TokenAmount


class TestTransactionBuilder:

    @pytest.fixture
    def temp_env(self, tmp_path):
        """Creates a temporary .env file for testing to avoid touching project root."""
        env_file = tmp_path / ".env"
        return str(env_file)

    @pytest.fixture
    def wallet(self, temp_env):
        """
        Generates a fresh, unique wallet for each test run.
        The private key is never hardcoded and is managed via WalletManager.generate.
        """
        return WalletManager.generate(env_path=temp_env)

    @pytest.fixture
    def mock_client(self):
        """
        Mocks ChainClient to simulate network responses.
        Isolates builder logic from actual RPC calls.
        """
        client = MagicMock()
        client.get_chain_id.return_value = 1
        client.get_nonce.return_value = 42
        client.get_gas_price.return_value = GasPrice(
            base_fee=100_000_000_000,   # 100 Gwei
            priority_fee_low=1_000_000_000,   # 1 Gwei
            priority_fee_medium=2_000_000_000,
            priority_fee_high=4_000_000_000
        )
        client.estimate_gas.return_value = 21000
        return client

    # --- Validation tests ---

    def test_build_requires_recipient_address(self, mock_client, wallet):
        """Expectation: raise ValueError if 'to()' was never called."""
        builder = TransactionBuilder(mock_client, wallet).value(TokenAmount(100, 18))
        with pytest.raises(ValueError, match="to\\(\\) is required"):
            builder.build()

    def test_build_requires_gas_estimation_step(self, mock_client, wallet):
        """Expectation: raise ValueError if gas estimation steps are skipped."""
        builder = TransactionBuilder(mock_client, wallet).to(Address.from_string("0x" + "a" * 40))
        with pytest.raises(ValueError, match="Call with_gas_estimate"):
            builder.build()

    # --- Assembly & Logic ---

    def test_full_transaction_construction(self, mock_client, wallet):
        """Scenario: complete fluent call chain to create a transfer."""
        recipient = Address.from_string("0x" + "b" * 40)
        amount = TokenAmount.from_human("1.5", 18)

        tx_request = (
            TransactionBuilder(mock_client, wallet)
            .to(recipient)
            .value(amount)
            .with_gas_price("medium")
            .with_gas_estimate(buffer=1.2)
            .build()
        )

        assert tx_request.to == recipient
        assert tx_request.value.raw == 1.5 * 10 ** 18
        assert tx_request.nonce == 42
        assert tx_request.gas_limit == int(21000 * 1.2)
        assert tx_request.max_priority_fee == 2_000_000_000

    def test_nonce_override_prevents_network_query(self, mock_client, wallet):
        """Scenario: manual nonce override should bypass client.get_nonce()."""
        manual_nonce = 100
        builder = (
            TransactionBuilder(mock_client, wallet)
            .to(Address.from_string("0x" + "c" * 40))
            .nonce(manual_nonce)
            .with_gas_price()
            .with_gas_estimate()
        )
        tx = builder.build()

        assert tx.nonce == manual_nonce
        mock_client.get_nonce.assert_not_called()

    # --- Cryptographic integration ---

    def test_sign_transaction_output(self, mock_client, wallet):
        """
        Scenario: building and signing a transaction with a dynamically generated wallet.
        Expectation: SignedTransaction with correct fields and signature components.
        """
        builder = (
            TransactionBuilder(mock_client, wallet)
            .to(Address.from_string("0x" + "d" * 40))
            .with_gas_price()
            .with_gas_estimate()
        )

        signed = builder.build_and_sign()

        # Check for our custom snake_case attributes
        assert hasattr(signed, "raw_transaction")
        assert len(signed.raw_transaction) > 0

        # Signature components validation
        assert signed.v is not None
        assert signed.r is not None
        assert signed.s is not None

        # Extra check: ensure the hash is generated
        assert hasattr(signed, "hash")
        assert len(signed.hash) > 0
