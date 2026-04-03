import json
from typing import Any
from eth_hash.auto import keccak


class CanonicalSerializer:
    """
    Produces deterministic JSON for signing.

    Rules:
    - Keys sorted alphabetically (recursive)
    - No whitespace
    - Numbers as-is (but prefer string amounts)
    - Consistent unicode handling
    """
    @staticmethod
    def _canonicalize(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: CanonicalSerializer._canonicalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [CanonicalSerializer._canonicalize(item) for item in obj]
        if isinstance(obj, float):
            raise ValueError("Floating point numbers are prohibited in canonical serialization")
        if isinstance(obj, (int, str, bool, type(None))):
            return obj
        if isinstance(obj, bytes):
            return obj.hex()
        raise TypeError(f"Unsupported type: {type(obj)}")

    @staticmethod
    def serialize(obj: Any) -> bytes:
        """Returns canonical bytes representation."""
        canon = CanonicalSerializer._canonicalize(obj)
        return json.dumps(canon, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    @staticmethod
    def hash(obj: Any) -> bytes:
        """Returns keccak256 of canonical serialization."""
        return keccak(CanonicalSerializer.serialize(obj))
