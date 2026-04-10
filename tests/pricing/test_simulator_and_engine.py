"""
Tests for ForkedChain, TradeSimulator, and PricingEngine / PriceQuote.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest
from eth_abi import encode as abi_encode

from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.pricing.engine import PriceQuote, PricingEngine, PricingError
from src.pricing.fork_simulator import ExecutionReceipt, ForkedChain, TradeSimulator
from src.pricing.mempool import PendingSwap
from src.pricing.router import SwapPath

USDC = Token(Address("0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"), "USDC", 6)
WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
DAI = Token(Address("0x6B175474E89094C44Da98b954EedeAC495271d0F"), "DAI", 18)

PAIR_A = Address("0x0000000000000000000000000000000000000011")
PAIR_B = Address("0x0000000000000000000000000000000000000022")
CALLER = Address("0x0000000000000000000000000000000000000001")
ROUTER = Address("0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D")

POOL = PoolState(PAIR_A, USDC, WETH, 100_000 * 10**6, 50 * 10**18, fee_bps=30)


def _engine() -> PricingEngine:
    e = PricingEngine.__new__(PricingEngine)
    e._client = MagicMock()
    e._simulator = MagicMock(spec=TradeSimulator)
    e._watcher = MagicMock()
    e._pools = {}
    e._finder = None
    e.seen_swaps = []
    return e


def _enc_amounts(amounts):
    return abi_encode(["uint256[]"], [amounts])


def _enc_reserves(r0, r1):
    return abi_encode(["uint112", "uint112", "uint32"], [r0, r1, 0])


def _chain() -> tuple[ForkedChain, MagicMock]:
    w3 = MagicMock()
    return ForkedChain(w3), w3


def _path(pool=POOL) -> SwapPath:
    return SwapPath([pool], [USDC, WETH])


def _sim() -> tuple[TradeSimulator, MagicMock]:
    chain, w3 = _chain()
    return TradeSimulator(chain), w3


def _swap(sold=USDC.address, bought=WETH.address) -> PendingSwap:
    return PendingSwap(
        tx_hash="0xabc", router_addr=str(ROUTER), protocol="UniswapV2",
        fn_name="swapExactTokensForTokens",
        token_sold=sold, token_bought=bought,
        qty_in=1_000 * 10**6, min_qty_out=490 * 10**15,
        deadline=9_999_999_999, trader=CALLER, gas_price=10**9,
    )


# ===========================================================================
# ExecutionReceipt
# ===========================================================================

class TestExecutionReceipt:
    def test_empty_logs_by_default(self):
        assert ExecutionReceipt(True, 0, 0, None).logs == []

    def test_ok(self):
        r = ExecutionReceipt(ok=True, qty_out=100, gas_used=50_000, error=None)
        assert r.ok and r.qty_out == 100

    def test_failed(self):
        r = ExecutionReceipt(ok=False, qty_out=0, gas_used=0, error="revert")
        assert not r.ok and r.error == "revert"


# ===========================================================================
# TradeSimulator
# ===========================================================================

class TestTradeSimulator:
    def test_quote_via_router_ok(self):
        sim, w3 = _sim()
        out = 490 * 10**15
        w3.eth.call.return_value = _enc_amounts([1_000 * 10**6, out])
        r = sim.quote_via_router(ROUTER, 1_000 * 10**6,
                                 [USDC.address.checksum, WETH.address.checksum], CALLER)
        assert r.ok and r.qty_out == out

    def test_quote_via_router_fail_on_revert(self):
        sim, w3 = _sim()
        w3.eth.call.side_effect = Exception("revert")
        r = sim.quote_via_router(ROUTER, 10**30,
                                 [USDC.address.checksum, WETH.address.checksum], CALLER)
        assert not r.ok

    def test_verify_path_matches_local(self):
        sim, w3 = _sim()
        w3.eth.call.return_value = _enc_reserves(POOL.qty_left, POOL.qty_right)
        path = _path()
        qty = 1_000 * 10**6
        r = sim.verify_path(path, qty)
        assert r.ok
        assert r.qty_out == POOL.out_for_in(qty, USDC)

    def test_cross_check_match(self):
        sim, w3 = _sim()
        qty = 1_000 * 10**6
        expected = POOL.out_for_in(qty, USDC)
        w3.eth.call.return_value = _enc_amounts([qty, expected])
        result = sim.cross_check(POOL, qty, USDC)
        assert result["match"] is True and result["delta"] == 0

    def test_execute_ok(self):
        sim, w3 = _sim()
        w3.eth.send_transaction.return_value = bytes.fromhex("ab" * 32)
        w3.eth.get_transaction_receipt.return_value = {"gasUsed": 180_000, "logs": []}
        r = sim.execute(ROUTER, 1_000 * 10**6, 490 * 10**15,
                        [USDC.address.checksum, WETH.address.checksum], CALLER)
        assert r.ok and r.gas_used == 180_000

    def test_cross_check_mismatch(self):
        sim, w3 = _sim()
        qty = 1_000 * 10**6
        local = POOL.out_for_in(qty, USDC)
        w3.eth.call.return_value = _enc_amounts([qty, local - 50])
        result = sim.cross_check(POOL, qty, USDC)
        assert result["match"] is False and result["delta"] == 50

    def test_verify_path_fail_on_rpc_error(self):
        sim, w3 = _sim()
        w3.eth.call.side_effect = Exception("timeout")
        r = sim.verify_path(_path(), 1_000 * 10**6)
        assert not r.ok

    def test_execute_fail_on_exception(self):
        sim, w3 = _sim()
        w3.eth.send_transaction.side_effect = RuntimeError("out of gas")
        r = sim.execute(ROUTER, 10**30, 0,
                        [USDC.address.checksum, WETH.address.checksum], CALLER)
        assert not r.ok and "out of gas" in r.error


# ===========================================================================
# PriceQuote
# ===========================================================================

class TestPriceQuote:
    def _q(self, expected, verified) -> PriceQuote:
        return PriceQuote(
            path=_path(), qty_in=1_000 * 10**6,
            expected_net=expected, verified_out=verified,
            gas_used=250_000, created_at=time.time()
        )

    def test_exact_match_trustworthy(self):
        assert self._q(1_000, 1_000).trustworthy

    def test_zero_both_trustworthy(self):
        assert self._q(0, 0).trustworthy

    def test_zero_expected_nonzero_verified_not_trustworthy(self):
        assert not self._q(0, 1).trustworthy

    def test_large_gap_not_trustworthy(self):
        assert not self._q(1_000_000, 970_000).trustworthy   # 3 %


# ===========================================================================
# PricingEngine
# ===========================================================================

class TestPricingEngine:
    def test_refresh_unknown_pool_raises(self):
        with pytest.raises(KeyError):
            _engine().refresh(PAIR_B)

    def test_no_pools_raises(self):
        with pytest.raises(PricingError, match="No pools"):
            _engine().get_quote(USDC, WETH, 1_000 * 10**6, gas_gwei=1)

    def test_get_quote_success(self):
        e = _engine()
        e._pools[PAIR_A] = POOL
        e._finder = MagicMock()
        net = POOL.out_for_in(1_000 * 10**6, USDC) - 1_000
        e._finder.find_optimal.return_value = (_path(), net)
        gross = POOL.out_for_in(1_000 * 10**6, USDC)
        e._simulator.verify_path.return_value = ExecutionReceipt(True, gross, 250_000, None)
        q = e.get_quote(USDC, WETH, 1_000 * 10**6, gas_gwei=1)
        assert isinstance(q, PriceQuote)
        assert q.expected_net == net

    def test_handle_pending_queues_relevant(self):
        e = _engine()
        e._pools[PAIR_A] = POOL
        e._handle_pending(_swap(USDC.address, WETH.address))
        assert len(e.seen_swaps) == 1

    def test_get_quote_no_route_raises(self):
        e = _engine()
        e._pools[PAIR_A] = POOL
        e._finder = MagicMock()
        e._finder.find_optimal.side_effect = ValueError("No route found")
        with pytest.raises(PricingError, match="No route found"):
            e.get_quote(USDC, DAI, 1_000 * 10**6, gas_gwei=1)

    def test_get_quote_simulation_failure_raises(self):
        e = _engine()
        e._pools[PAIR_A] = POOL
        e._finder = MagicMock()
        e._finder.find_optimal.return_value = (_path(), 490 * 10**15)
        e._simulator.verify_path.return_value = ExecutionReceipt(False, 0, 0, "revert")
        with pytest.raises(PricingError, match="Fork verification failed"):
            e.get_quote(USDC, WETH, 1_000 * 10**6, gas_gwei=1)
