"""
Validation module: blockchain address integrity checks.
Supports:
- BTC: P2PKH, SegWit (P2WPKH, Bech32), Taproot (P2TR, Bech32m).
- ETH: Standard Ethereum HEX format.
"""

import re


class AddressValidator:
    # P2PKH: starts with '1'
    BTC_P2PKH_REGEX = r"^1[a-km-zA-NP-Z1-9]{25,34}$"
    # SegWit (P2WPKH): starts with 'bc1q' (Bech32)
    BTC_SEGWIT_REGEX = r"^bc1q[a-z0-9]{38,58}$"
    # Taproot (P2TR): starts with 'bc1p' (Bech32m)
    BTC_TAPROOT_REGEX = r"^bc1p[a-z0-9]{38,58}$"
    # Ethereum: standard 0x + 40 HEX chars
    ETH_REGEX = r"^0x[a-fA-F0-9]{40}$"

    @staticmethod
    def is_valid_btc_p2pkh(address: str) -> bool:
        """Validate P2PKH Bitcoin address."""
        return bool(address and re.match(AddressValidator.BTC_P2PKH_REGEX,
                                         address))

    @staticmethod
    def is_valid_btc_segwit(address: str) -> bool:
        """Validate SegWit Bitcoin address (P2WPKH)."""
        return bool(address and re.match(AddressValidator.BTC_SEGWIT_REGEX,
                                         address))

    @staticmethod
    def is_valid_btc_taproot(address: str) -> bool:
        """Validate Taproot Bitcoin address (P2TR)."""
        return bool(address and re.match(AddressValidator.BTC_TAPROOT_REGEX,
                                         address))

    @staticmethod
    def is_valid_eth(address: str) -> bool:
        """Validate Ethereum address format."""
        return bool(address and re.match(AddressValidator.ETH_REGEX,
                                         address))
