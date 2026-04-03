import os
import json
from typing import Dict, Any
from dotenv import load_dotenv
from eth_account import Account
from eth_account.messages import encode_defunct, encode_typed_data
from eth_account.datastructures import SignedMessage, SignedTransaction
from pydantic import SecretStr


class WalletManager:
    """
    Manages wallet operations: key loading, signing, verification.
    Uses SecretStr to prevent private key leakage in logs and traces.
    """

    def __init__(self, private_key: str | SecretStr):
        # Convert to string if SecretStr passed, ensure 0x prefix
        raw_key = private_key.get_secret_value() if isinstance(private_key, SecretStr) else private_key

        if not raw_key.startswith("0x"):
            raw_key = "0x" + raw_key

        self._private_key = SecretStr(raw_key)
        self._account = Account.from_key(self._private_key.get_secret_value())

    @classmethod
    def from_env(cls, env_var: str = "PRIVATE_KEY") -> "WalletManager":
        """Load private key from environment variable."""
        load_dotenv()
        key = os.getenv(env_var)
        if not key:
            raise ValueError(f"{env_var} not found in environment")
        return cls(SecretStr(key))

    @classmethod
    def generate(cls, env_path: str = ".env", env_var: str = "PRIVATE_KEY") -> "WalletManager":
        """Generate a new random wallet and append its private key to a .env file."""
        acct = Account.create()
        # Consistently use 0x prefix in the .env file
        private_key_raw = "0x" + acct.key.hex()

        prefix = ""
        if os.path.exists(env_path) and os.path.getsize(env_path) > 0:
            with open(env_path, "r") as f:
                if not f.read().endswith("\n"):
                    prefix = "\n"

        line = f"{prefix}{env_var}={private_key_raw}\n"
        with open(env_path, "a") as f:
            f.write(line)

        return cls(SecretStr(private_key_raw))

    @property
    def address(self) -> str:
        """Returns checksummed address."""
        return self._account.address

    def sign_message(self, message: str) -> SignedMessage:
        if not message or not message.strip():
            raise ValueError("Signing empty message is prohibited")
        return self._account.sign_message(encode_defunct(text=message))

    def sign_typed_data(self, domain: Dict[str, Any], types: Dict[str, Any], value: Dict[str, Any]) -> SignedMessage:
        if not all([domain, types, value]):
            raise ValueError("Invalid EIP-712 data")
        signable = encode_typed_data(domain_data=domain, message_types=types, message_data=value)
        return self._account.sign_message(signable)

    def sign_transaction(self, tx: dict) -> SignedTransaction:
        return self._account.sign_transaction(tx)

    def __repr__(self) -> str:
        # SecretStr in __init__ is already safe, but we keep this as backup
        return f"WalletManager(address={self.address})"

    def __str__(self) -> str:
        return self.__repr__()

    @classmethod
    def from_keyfile(cls, path: str, password: str) -> "WalletManager":
        with open(path) as f:
            encrypted = f.read()
        acct = Account.decrypt(encrypted, password)
        return cls(SecretStr(acct.hex()))

    def to_keyfile(self, path: str, password: str) -> None:
        encrypted = self._account.encrypt(password)
        with open(path, "w") as f:
            json.dump(encrypted, f, indent=2)
