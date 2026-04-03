import json
import pytest

from eth_account import Account
from eth_account.messages import encode_defunct
from pydantic import SecretStr

from src.core.wallet import WalletManager


class TestWalletManager:

    @pytest.fixture
    def temp_env(self, tmp_path):
        """Creates a temporary .env file for testing."""
        env_file = tmp_path / ".env"
        return str(env_file)

    @pytest.fixture
    def test_wallet(self, temp_env):
        """Generates a wallet specifically for a test run and saves it to temp .env."""
        return WalletManager.generate(env_path=temp_env)

    # --- Security & Environment tests ---

    def test_generate_saves_to_env_file(self, temp_env):
        """Verify that generate() writes the key and matches the internal secret."""
        wm = WalletManager.generate(env_path=temp_env, env_var="TEST_KEY")

        with open(temp_env, "r") as f:
            content = f.read()

        assert "TEST_KEY=0x" in content
        # Use get_secret_value() to check content without leaking it in logs if assert fails
        raw_key = wm._private_key.get_secret_value()
        assert raw_key in content

    def test_from_env_loading(self, temp_env, monkeypatch):
        """Verify that from_env properly reads the key from environment variables."""
        # Setup: generate and then simulate environment loading
        wm_original = WalletManager.generate(env_path=temp_env, env_var="KEY_FOR_ENV")
        raw_key = wm_original._private_key.get_secret_value()

        # Mocking the environment variable for the process
        monkeypatch.setenv("KEY_FOR_ENV", raw_key)
        wm_loaded = WalletManager.from_env(env_var="KEY_FOR_ENV")

        assert wm_loaded.address == wm_original.address

    def test_string_representation_security(self, test_wallet):
        """Ensure private key is masked even in string outputs."""
        output_str = str(test_wallet)
        output_repr = repr(test_wallet)
        raw_key = test_wallet._private_key.get_secret_value()

        assert test_wallet.address in output_str
        assert raw_key not in output_str
        assert raw_key not in output_repr
        # Verify that the secret internal attribute shows as masked if accessed
        assert "**********" in str(test_wallet._private_key)

    def test_error_handling_masks_private_key(self):
        """Verify that invalid key initialization does not leak the input in the error."""
        invalid_key = "0xnot-a-hex-key-123456789"
        with pytest.raises(Exception) as exc_info:
            WalletManager(SecretStr(invalid_key))

        # Check that the raw invalid string is not in the exception message
        assert invalid_key not in str(exc_info.value)

    # --- Signing validation tests ---

    def test_sign_empty_message_raises_error(self, test_wallet):
        """Verify that signing an empty or whitespace-only message is prohibited."""
        with pytest.raises(ValueError, match="Signing empty message is prohibited"):
            test_wallet.sign_message("")

        with pytest.raises(ValueError, match="Signing empty message is prohibited"):
            test_wallet.sign_message("   ")

    def test_sign_invalid_eip712_data_raises_error(self, test_wallet):
        """Verify that missing components for EIP-712 signing raise a ValueError early."""
        domain = {"name": "Test"}
        types = {"Person": [{"name": "name", "type": "string"}]}
        value = {"name": "Alice"}

        # Test missing domain
        with pytest.raises(ValueError, match="Invalid EIP-712 data"):
            test_wallet.sign_typed_data({}, types, value)

        # Test missing types
        with pytest.raises(ValueError, match="Invalid EIP-712 data"):
            test_wallet.sign_typed_data(domain, {}, value)

        # Test missing value
        with pytest.raises(ValueError, match="Invalid EIP-712 data"):
            test_wallet.sign_typed_data(domain, types, {})

    # --- Functional & Crypto tests ---

    def test_wallet_generation(self, temp_env):
        """Verify that a generated wallet has a valid Ethereum address and masked key."""
        # We pass temp_env to avoid creating a real .env in the project root
        new_wallet = WalletManager.generate(env_path=temp_env)

        assert new_wallet.address.startswith("0x")
        assert len(new_wallet.address) == 42
        # Ensure the internal secret is masked
        assert "**********" in str(new_wallet._private_key)

    def test_message_signing_and_verification(self, test_wallet):
        """Verify that a signed message can be recovered to the correct address."""
        msg = "Verify me!"
        signed = test_wallet.sign_message(msg)

        # Use eth_account to recover the address from the signature
        recovered_addr = Account.recover_message(
            signable_message=encode_defunct(text=msg),
            signature=signed.signature
        )
        assert recovered_addr == test_wallet.address

    def test_keyfile_storage(self, test_wallet, tmp_path):
        """Verify encrypting to and decrypting from a keyfile (JSON keystore)."""
        path = tmp_path / "test_wallet.json"
        password = "secure_password_123"

        # Save the current wallet to a keystore
        test_wallet.to_keyfile(str(path), password)
        assert path.exists()

        # Check if it is a valid JSON keystore structure
        with open(path) as f:
            data = json.load(f)
            assert "crypto" in data or "scrypt" in data

        # Load back from the keystore and verify the address matches
        loaded_wallet = WalletManager.from_keyfile(str(path), password)
        assert loaded_wallet.address == test_wallet.address
        # Verify the loaded wallet also uses SecretStr masking
        assert "**********" in str(loaded_wallet._private_key)
