"""
Tests for PendingSwap and MempoolWatcher.
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from eth_abi import encode as abi_encode

from src.core.types import Address
from src.pricing.mempool import MempoolWatcher, PendingSwap

USDC = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
WETH = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
DAI = "0x6B175474E89094C44Da98b954EedeAC495271d0F"
ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
SENDER = "0x0000000000000000000000000000000000000003"
DL = 9_999_999_999


# ---------------------------------------------------------------------------
# Calldata factories
# ---------------------------------------------------------------------------

def _calldata(sel: str, types, values) -> str:
    return "0x" + sel + abi_encode(types, values).hex()


def buy_with_eth(min_out=480 * 10**15) -> str:
    return _calldata("7ff36ab5",
                     ["uint256", "address[]", "address", "uint256"],
                     [min_out, [WETH, DAI], SENDER, DL])


def sell_for_eth(qty=2_000 * 10**6, min_out=9 * 10**17) -> str:
    return _calldata("18cbafe5",
                     ["uint256", "uint256", "address[]", "address", "uint256"],
                     [qty, min_out, [USDC, WETH], SENDER, DL])


def sell_tokens(qty=1_000 * 10**6, min_out=490 * 10**15) -> str:
    return _calldata("38ed1739",
                     ["uint256", "uint256", "address[]", "address", "uint256"],
                     [qty, min_out, [USDC, WETH], SENDER, DL])


def v3_multicall() -> str:
    return _calldata("5ae401dc", ["uint256", "bytes[]"], [DL, [b"\xde\xad"]])


def _tx(data, value=0) -> dict:
    return {"hash": "0xabc", "from": SENDER, "to": ROUTER,
            "input": data, "value": value, "gasPrice": 10**9}


def _watcher() -> MempoolWatcher:
    return MempoolWatcher("wss://fake", lambda _: None)


# ===========================================================================
# PendingSwap
# ===========================================================================

class TestPendingSwap:
    def _make(self, min_out=490 * 10**15, expected=None) -> PendingSwap:
        return PendingSwap(
            tx_hash="0xdead",
            router_addr=ROUTER,
            protocol="UniswapV2",
            fn_name="swapExactTokensForTokens",
            token_sold=Address(USDC),
            token_bought=Address(WETH),
            qty_in=1_000 * 10**6,
            min_qty_out=min_out,
            deadline=DL,
            trader=Address(SENDER),
            gas_price=10**9,
            expected_out=expected
        )

    def test_implied_slippage_zero(self):
        ps = self._make(min_out=1_000, expected=1_000)
        assert ps.implied_slippage == Decimal(0)

    def test_fields_set_correctly(self):
        ps = self._make()
        assert ps.protocol == "UniswapV2"
        assert ps.qty_in == 1_000 * 10**6

    def test_implied_slippage_correct(self):
        ps = self._make(min_out=990, expected=1_000)
        assert ps.implied_slippage == Decimal("0.01")


# ===========================================================================
# _decode_body
# ===========================================================================

class TestDecodeBody:
    def _body(self, types, values) -> bytes:
        return abi_encode(types, values)

    def test_sell_tokens(self):
        body = self._body(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [1_000 * 10**6, 490 * 10**15, [USDC, WETH], SENDER, DL],
        )
        d = MempoolWatcher._decode_body("0x38ed1739", body)
        assert d["qty_in"] == 1_000 * 10**6
        assert d["min_qty_out"] == 490 * 10**15
        assert d["token_sold"].lower() == USDC.lower()
        assert d["token_bought"].lower() == WETH.lower()

    def test_sell_for_eth_no_token_bought(self):
        body = self._body(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [2_000 * 10**6, 9 * 10**17, [USDC, WETH], SENDER, DL],
        )
        d = MempoolWatcher._decode_body("0x18cbafe5", body)
        assert d["token_bought"] is None

    def test_truncated_calldata_raises(self):
        with pytest.raises(Exception):
            MempoolWatcher._decode_body("0x38ed1739", b"\x00" * 8)

    def test_buy_with_eth_no_token_sold(self):
        body = self._body(
            ["uint256", "address[]", "address", "uint256"],
            [480 * 10**15, [WETH, DAI], SENDER, DL],
        )
        d = MempoolWatcher._decode_body("0x7ff36ab5", body)
        assert d["token_sold"] is None
        assert d["qty_in"] == 0

    def test_unknown_selector_raises(self):
        with pytest.raises(ValueError, match="No ABI"):
            MempoolWatcher._decode_body("0xdeadbeef", b"\x00" * 32)


# ===========================================================================
# MempoolWatcher.decode
# ===========================================================================

class TestDecodeMethod:
    def setup_method(self):
        self.w = _watcher()

    def test_returns_none_for_short(self):
        assert self.w.decode(_tx("0x1234")) is None

    def test_parses_sell_tokens(self):
        result = self.w.decode(_tx(sell_tokens()))
        assert result is not None
        assert result.fn_name == "swapExactTokensForTokens"
        assert result.qty_in == 1_000 * 10**6

    def test_buy_with_eth_uses_tx_value(self):
        result = self.w.decode(_tx(buy_with_eth(), value=3 * 10**18))
        assert result is not None
        assert result.qty_in == 3 * 10**18
        assert result.token_sold is None

    def test_sell_for_eth_no_token_bought(self):
        result = self.w.decode(_tx(sell_for_eth()))
        assert result is not None
        assert result.token_bought is None

    def test_returns_none_for_unknown_selector(self):
        assert self.w.decode(_tx("0xdeadbeef" + "00" * 32)) is None

    def test_returns_none_for_empty(self):
        assert self.w.decode(_tx("")) is None

    def test_bytes_input_accepted(self):
        raw = bytes.fromhex("38ed1739") + abi_encode(
            ["uint256", "uint256", "address[]", "address", "uint256"],
            [1_000 * 10**6, 490 * 10**15, [USDC, WETH], SENDER, DL],
        )
        result = self.w.decode(_tx(raw))
        assert result is not None

    def test_v3_multicall_detected(self):
        result = self.w.decode(_tx(v3_multicall()))
        assert result is not None
        assert result.protocol == "UniswapV3"

    def test_max_fee_fallback(self):
        tx = {"hash": "0xabc", "from": SENDER, "to": ROUTER,
              "input": sell_tokens(), "value": 0, "maxFeePerGas": 25 * 10**9}
        result = self.w.decode(tx)
        assert result is not None
        assert result.gas_price == 25 * 10**9

    def test_bytes_hash_converted_to_hex(self):
        tx = _tx(sell_tokens())
        tx["hash"] = bytes.fromhex("ab" * 32)
        result = self.w.decode(tx)
        assert result is not None
        assert result.tx_hash == "0x" + "ab" * 32

    def test_bad_from_address_returns_none(self):
        tx = _tx(sell_tokens())
        tx["from"] = "not-valid"
        assert self.w.decode(tx) is None

    def test_invalid_hex_returns_none(self):
        assert self.w.decode(_tx("0x38ed1739" + "ZZ" * 10)) is None


# ===========================================================================
# Async watch
# ===========================================================================

class TestWatcher:
    async def _run(self, messages, tx_map):
        received = []
        w = MempoolWatcher("wss://fake", received.append)

        async def _subs():
            for m in messages:
                yield m

        mock_w3 = MagicMock()
        mock_w3.eth.subscribe = AsyncMock()
        mock_w3.eth.get_transaction = AsyncMock(side_effect=lambda h: tx_map.get(h))
        mock_w3.socket.process_subscriptions = _subs
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_w3)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("src.pricing.mempool.AsyncWeb3", return_value=ctx):
            await w.watch()
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            await asyncio.gather(*tasks)
        return received

    @pytest.mark.asyncio
    async def test_no_call_for_unknown_selector(self):
        h = "0xdeadbeef"
        tx = {**_tx("0xdeadbeef" + "00" * 32), "hash": h}
        received = await self._run([{"params": {"result": h}}], {h: tx})
        assert received == []

    @pytest.mark.asyncio
    async def test_handler_called_for_swap(self):
        h = "0xdeadbeef"
        tx = {**_tx(sell_tokens()), "hash": h}
        received = await self._run([{"params": {"result": h}}], {h: tx})
        assert len(received) == 1
        assert received[0].fn_name == "swapExactTokensForTokens"
