"""
Validation test suite: ensuring blockchain address integrity.
"""

from src.validators import AddressValidator

# --- Positive tests ---


def test_valid_btc_p2pkh():
    """Verify P2PKH address (starts with 1)."""
    assert AddressValidator.is_valid_btc_p2pkh(
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa")


def test_valid_btc_segwit():
    """Verify SegWit P2WPKH address (starts with bc1q)."""
    assert AddressValidator.is_valid_btc_segwit(
        "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    )


def test_valid_btc_taproot():
    """Verify Taproot P2TR address (starts with bc1p)."""
    assert AddressValidator.is_valid_btc_taproot(
        "bc1p5d7rjq7nd6kczy7xe86ea9ge6cr93v6y02n3v7"
    )


def test_valid_eth_address():
    """Verify standard Ethereum (ERC-20) HEX address."""
    assert AddressValidator.is_valid_eth(
        "0x742d35Cc6634C0532925a3b844Bc454e4438f44e")


# --- Negative tests ---


def test_invalid_btc_base58_chars():
    """Reject P2PKH addresses containing invalid characters (like 'O', 'I')."""
    # Base58 excludes 0, O, I, l
    assert not AddressValidator.is_valid_btc_p2pkh(
        "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNO")


def test_cross_format_rejection():
    """Ensure P2PKH validator rejects SegWit format."""
    segwit_addr = "bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh"
    assert not AddressValidator.is_valid_btc_p2pkh(segwit_addr)


def test_invalid_eth_format():
    """Reject ETH addresses with missing 0x or invalid hex chars."""
    assert not AddressValidator.is_valid_eth(
        "742d35Cc6634C0532925a3b844Bc454e4438f44e")
    assert not AddressValidator.is_valid_eth(
        "0xGHIJKLMNOPQRSTUVWXYZ")


def test_empty_and_none_inputs():
    """Ensure methods handle empty strings."""
    assert not AddressValidator.is_valid_btc_p2pkh("")
    assert not AddressValidator.is_valid_btc_segwit("")
    assert not AddressValidator.is_valid_btc_taproot("")
    assert not AddressValidator.is_valid_eth("")
