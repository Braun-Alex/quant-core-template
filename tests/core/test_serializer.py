from typing import Any
import pytest
from src.core.serializer import CanonicalSerializer


def verify_determinism(obj: Any, iterations: int = 1000) -> bool:
    """Helper function to verify serialization is deterministic over N iterations."""
    first = CanonicalSerializer.serialize(obj)
    for _ in range(iterations):
        if CanonicalSerializer.serialize(obj) != first:
            return False
    return True


class TestCanonicalSerializer:

    def test_nested_mixed_key_order(self):
        """Verify that dictionaries with different key insertion orders produce identical output."""
        obj1 = {"a": 1, "b": {"z": 10, "y": 20}, "c": [3, 2, 1]}
        obj2 = {"c": [3, 2, 1], "a": 1, "b": {"y": 20, "z": 10}}

        ser1 = CanonicalSerializer.serialize(obj1)
        ser2 = CanonicalSerializer.serialize(obj2)

        assert ser1 == ser2
        # Ensure no whitespace and keys are sorted (a, then b, then c)
        assert ser1.decode() == '{"a":1,"b":{"y":20,"z":10},"c":[3,2,1]}'

    def test_unicode_handling(self):
        """Verify consistent handling of non-ASCII characters and emojis."""
        obj = {"msg": "Hello 🤝 Ethereum", "util": "λ"}
        serialized = CanonicalSerializer.serialize(obj)

        assert b"\\u03bb" in serialized   # Lambda
        assert b"\\ud83e\\udd1d" in serialized   # Handshake emoji
        assert verify_determinism(obj)

    def test_large_integers(self):
        """Verify that integers exceeding 2^53 (JS unsafe) are handled correctly as-is."""
        # Standard JS numbers lose precision above 2^53 - 1
        large_int = 2 ** 256 - 1
        obj = {"uint256": large_int}

        serialized = CanonicalSerializer.serialize(obj)
        # Python handles arbitrary precision, so it should stay as a raw number in JSON
        assert serialized == f'{{"uint256":{large_int}}}'.encode()
        assert verify_determinism(obj)

    def test_none_and_empty_structures(self):
        """Verify handling of None (null), empty dicts, and empty lists."""
        obj = {
            "a": None,
            "b": [],
            "c": {},
            "d": [None, {}]
        }
        serialized = CanonicalSerializer.serialize(obj)
        assert serialized == b'{"a":null,"b":[],"c":{},"d":[null,{}]}'
        assert verify_determinism(obj)

    def test_float_prohibition(self):
        """Verify that floating point numbers raise a ValueError to prevent rounding issues."""
        obj_with_float = {"amount": 1.5}

        with pytest.raises(ValueError, match="Floating point numbers are prohibited"):
            CanonicalSerializer.serialize(obj_with_float)

    def test_bytes_to_hex_conversion(self):
        """Verify that bytes are automatically converted to hex strings."""
        obj = {"data": b"\x01\x02\x03\xff"}
        serialized = CanonicalSerializer.serialize(obj)
        assert b"010203ff" in serialized
        assert verify_determinism(obj)

    def test_unsupported_types(self):
        """Verify that unsupported types (like sets or custom classes) raise TypeError."""

        class Custom:
            pass

        with pytest.raises(TypeError, match="Unsupported type"):
            CanonicalSerializer.serialize({"val": Custom()})

        with pytest.raises(TypeError, match="Unsupported type"):
            CanonicalSerializer.serialize({"val": {1, 2, 3}})   # Sets are not supported

    def test_hash_consistency(self):
        """Verify that the keccak hash is consistent with the serialization."""
        obj = {"Just": "Value"}
        h1 = CanonicalSerializer.hash(obj)
        h2 = CanonicalSerializer.hash(obj)

        assert len(h1) == 32
        assert h1 == h2

    def test_deep_recursion_sorting(self):
        """
        Verify that sorting is applied at every level of a deeply nested structure.
        Ensures that 'q' coming before 'a' at level 5 is corrected.
        """
        deep_obj = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "q_key": "last",
                            "a_key": "first"
                        }
                    }
                }
            }
        }
        serialized = CanonicalSerializer.serialize(deep_obj)
        # The output must have "a_key" before "q_key" even inside level 4
        expected = b'{"level1":{"level2":{"level3":{"level4":{"a_key":"first","q_key":"last"}}}}}'
        assert serialized == expected

    def test_list_containing_dicts(self):
        """
        Verify that dictionaries inside lists are also canonicalized.
        Standard JSON sort_keys=True works here, but we must ensure our recursive
        _canonicalize handles the list traversal correctly.
        """
        obj = {
            "data": [
                {"delta": 4, "gamma": 3},
                {"beta": 2, "alpha": 1}
            ]
        }
        serialized = CanonicalSerializer.serialize(obj)

        expected = b'{"data":[{"delta":4,"gamma":3},{"alpha":1,"beta":2}]}'

        assert serialized == expected

    def test_complex_mixed_nesting(self):
        """
        A stress test with a mix of lists, dicts, bytes, and None.
        This ensures that the recursive logic does not lose track of types
        when switching between list and dict contexts.
        """
        obj = {
            "root": [
                {
                    "b_inner": b"\xff\x00",
                    "a_inner": [
                        {"y": None, "x": True},
                        {"z": False}
                    ]
                },
                "Value",
                123456789
            ]
        }
        serialized = CanonicalSerializer.serialize(obj)

        # Verify specific segments of the canonical string:
        # 1) a_inner should come before b_inner
        # 2) x should come before y
        # 3) bytes should be hex string "ff00"
        decoded = serialized.decode()
        assert '"a_inner":[{"x":true,"y":null},{"z":false}],"b_inner":"ff00"' in decoded
        assert verify_determinism(obj)

    def test_extreme_nesting_limit(self):
        """
        Verify that the serializer handles reasonably deep structures
        without hitting Python's recursion limit (default is usually 1000).
        """
        curr = {"final": "value"}
        for i in range(100):
            curr = {f"depth_{99 - i}": curr}

        # This should not raise RecursionError
        serialized = CanonicalSerializer.serialize(curr)
        assert serialized.startswith(b'{"depth_0":{"depth_1":')
        assert b'"final":"value"' in serialized
