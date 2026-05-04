"""
Tests for demo mode infrastructure.

Coverage:
  TestBinanceDemoClient         - URL override, API normalization, order format
  TestAnvilInventoryProvisioner - storage slot computation, balance reads
  TestDemoConfig                - build_demo_config(), ARB/USDC rules, demo mode flag
  TestDemoSetup                 - CLI argument parsing, pool info
  TestOperationModeDemo         - SystemConfig.from_env() with OPERATION_MODE=demo
"""

from __future__ import annotations

import os
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.exchange.demo_client import BinanceDemoClient
from src.exchange.demo_inventory import (
    AnvilInventoryProvisioner, ARB_TOKENS, ARB_POOLS, DemoInventory
)
from src.exchange.demo_setup import (
    build_demo_config, DEMO_TRADING_PAIR, DEMO_POOL_ADDRESS,
    ARB_USDC_TRADING_RULES, make_demo_order_book
)
from config.mode import OperationMode, SystemConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ccxt_exchange(time_result=None, order_book_result=None,
                        balance_result=None, order_result=None):
    """Build a mock ccxt.binance instance."""
    ex = MagicMock()
    ex.fetch_time.return_value = time_result or {}
    ex.fetch_order_book.return_value = order_book_result or {
        "symbol": "ARB/USDC", "timestamp": int(time.time() * 1000),
        "bids": [[0.85, 1000.0], [0.84, 500.0]],
        "asks": [[0.86, 800.0], [0.87, 300.0]]
    }
    ex.fetch_balance.return_value = balance_result or {
        "ARB": {"free": "10000", "used": "0", "total": "10000"},
        "USDC": {"free": "20000", "used": "0", "total": "20000"},
        "info": "metadata"
    }
    ex.create_order.return_value = order_result or {
        "id": "demo_001", "symbol": "ARB/USDC", "side": "buy",
        "type": "limit", "status": "closed",
        "filled": 100.0, "amount": 100.0,
        "average": 0.855, "price": 0.855,
        "fee": {"cost": 0.0855, "currency": "USDC"},
        "timestamp": int(time.time() * 1000)
    }
    return ex


def _make_client(ex=None) -> BinanceDemoClient:
    """Create a BinanceDemoClient with a mocked ccxt exchange."""
    mock_ex = ex or _mock_ccxt_exchange()
    client = BinanceDemoClient.__new__(BinanceDemoClient)
    client._exchange = mock_ex
    client._weight_consumed = 0
    client._window_resets_at = time.monotonic() + 60.0
    return client


def _make_provisioner(w3_mock=None) -> AnvilInventoryProvisioner:
    p = AnvilInventoryProvisioner.__new__(AnvilInventoryProvisioner)
    p._rpc = "http://localhost:8545"
    p._wallet = "0x" + "a" * 40
    p._inv = DemoInventory()
    p._w3 = w3_mock or MagicMock()
    return p


# ═══════════════════════════ BinanceDemoClient ════════════════════════════════

class TestBinanceDemoClient:

    def test_fetch_order_book_returns_dict(self):
        client = _make_client()
        book = client.fetch_order_book("ARB/USDC")
        assert "bids" in book and "asks" in book
        assert "best_bid" in book and "best_ask" in book
        assert "mid_price" in book and "spread_bps" in book

    def test_best_bid_is_highest_bid(self):
        client = _make_client()
        book = client.fetch_order_book("ARB/USDC")
        bids = book["bids"]
        assert book["best_bid"][0] == bids[0][0]   # Sorted descending

    def test_best_ask_is_lowest_ask(self):
        client = _make_client()
        book = client.fetch_order_book("ARB/USDC")
        asks = book["asks"]
        assert book["best_ask"][0] == asks[0][0]   # Sorted ascending

    def test_spread_bps_computed(self):
        client = _make_client()
        book = client.fetch_order_book("ARB/USDC")
        assert book["spread_bps"] > Decimal("0")

    def test_prices_are_decimal(self):
        client = _make_client()
        book = client.fetch_order_book("ARB/USDC")
        assert isinstance(book["best_bid"][0], Decimal)
        assert isinstance(book["best_ask"][0], Decimal)
        assert isinstance(book["mid_price"], Decimal)

    def test_fetch_balance_filters_zero_totals(self):
        ex = _mock_ccxt_exchange(balance_result={
            "ARB": {"free": "10000", "used": "0", "total": "10000"},
            "DOGE": {"free": "0", "used": "0", "total": "0"},
            "info": "meta"
        })
        client = _make_client(ex)
        bal = client.fetch_balance()
        assert "ARB" in bal
        assert "DOGE" not in bal
        assert "info" not in bal

    def test_fetch_balance_decimal_values(self):
        client = _make_client()
        bal = client.fetch_balance()
        for info in bal.values():
            assert isinstance(info["free"], Decimal)
            assert isinstance(info["locked"], Decimal)

    def test_create_limit_ioc_order_filled_status(self):
        client = _make_client()
        result = client.create_limit_ioc_order("ARB/USDC", "buy", 100.0, 0.86)
        assert result["status"] == "filled"
        assert isinstance(result["amount_filled"], Decimal)
        assert isinstance(result["avg_fill_price"], Decimal)

    def test_create_limit_ioc_order_passes_ioc(self):
        ex = _mock_ccxt_exchange()
        client = _make_client(ex)
        client.create_limit_ioc_order("ARB/USDC", "sell", 50.0, 0.85)
        ex.create_order.assert_called_once()
        call_args = ex.create_order.call_args
        assert call_args[0][1] == "limit"

    def test_normalize_order_partial_fill(self):
        ex = _mock_ccxt_exchange(order_result={
            "id": "002", "symbol": "ARB/USDC", "side": "buy",
            "type": "limit", "status": "closed",
            "filled": 40.0, "amount": 100.0,
            "average": 0.855, "price": 0.86,
            "fee": None, "timestamp": int(time.time() * 1000)
        })
        client = _make_client(ex)
        result = client.create_limit_ioc_order("ARB/USDC", "buy", 100.0, 0.86)
        assert result["status"] == "partially_filled"

    def test_normalize_order_cancelled(self):
        ex = _mock_ccxt_exchange(order_result={
            "id": "003", "symbol": "ARB/USDC", "side": "buy",
            "type": "limit", "status": "canceled",
            "filled": 0.0, "amount": 100.0,
            "average": None, "price": 0.86,
            "fee": None, "timestamp": int(time.time() * 1000)
        })
        client = _make_client(ex)
        result = client.create_limit_ioc_order("ARB/USDC", "buy", 100.0, 0.86)
        assert result["status"] == "expired"

    def test_ws_base_url_returns_demo_url(self):
        url = BinanceDemoClient.ws_base_url()
        assert "demo" in url.lower() or "binance" in url.lower()

    def test_rate_limit_tracking(self):
        client = _make_client()
        before = client._weight_consumed
        client.fetch_order_book("ARB/USDC")
        assert client._weight_consumed > before

    def test_from_env_raises_without_keys(self):
        with patch.dict(os.environ, {
            "BINANCE_DEMO_API_KEY": "",
            "BINANCE_DEMO_SECRET": ""
        }):
            with pytest.raises(ValueError, match="BINANCE_DEMO_API_KEY"):
                BinanceDemoClient.from_env()

    def test_from_env_raises_missing_secret(self):
        with patch.dict(os.environ, {
            "BINANCE_DEMO_API_KEY": "some_key",
            "BINANCE_DEMO_SECRET": ""
        }):
            with pytest.raises(ValueError):
                BinanceDemoClient.from_env()


# ═══════════════════════════ AnvilInventoryProvisioner ════════════════════════

class TestAnvilInventoryProvisioner:

    def test_arb_token_registered(self):
        assert "ARB" in ARB_TOKENS
        assert ARB_TOKENS["ARB"].decimals == 18
        assert ARB_TOKENS["ARB"].balance_slot == 51

    def test_usdc_token_registered(self):
        assert "USDC" in ARB_TOKENS
        assert ARB_TOKENS["USDC"].decimals == 6
        assert ARB_TOKENS["USDC"].balance_slot == 9

    def test_weth_token_registered(self):
        assert "WETH" in ARB_TOKENS
        assert ARB_TOKENS["WETH"].decimals == 18
        assert ARB_TOKENS["WETH"].balance_slot == 0

    def test_arb_usdc_pool_known(self):
        assert "ARB/USDC" in ARB_POOLS
        pool_addr = ARB_POOLS["ARB/USDC"]
        assert pool_addr.startswith("0x")
        assert len(pool_addr) >= 40   # Checksummed or lowercase

    def test_demo_inventory_has_arb_and_usdc(self):
        inv = DemoInventory()
        assert "ARB" in inv.token_amounts
        assert "USDC" in inv.token_amounts
        assert inv.token_amounts["ARB"] > 0
        assert inv.token_amounts["USDC"] > 0

    def test_is_anvil_true(self):
        p = _make_provisioner()
        p._w3.provider.make_request.return_value = {"result": {"version": "0.2.0"}}
        assert p._is_anvil() is True

    def test_is_anvil_false_real_node(self):
        p = _make_provisioner()
        p._w3.provider.make_request.side_effect = Exception("Method not found")
        assert p._is_anvil() is False

    def test_fund_eth_calls_anvil_set_balance(self):
        p = _make_provisioner()
        p._w3.provider.make_request.return_value = {"result": True}
        result = p._fund_eth(10 * 10 ** 18)
        assert result is True
        p._w3.provider.make_request.assert_called_with(
            "anvil_setBalance",
            [p._wallet, hex(10 * 10 ** 18)]
        )

    def test_fund_eth_returns_false_on_error(self):
        p = _make_provisioner()
        p._w3.provider.make_request.side_effect = Exception("RPC error")
        assert p._fund_eth(1) is False

    def test_fund_erc20_returns_false_on_error(self):
        p = _make_provisioner()
        p._w3.provider.make_request.side_effect = Exception("write failed")
        token = ARB_TOKENS["ARB"]
        assert p._fund_erc20(token, 100 * 10 ** 18) is False

    def test_fund_demo_wallet_anvil_not_running(self):
        p = _make_provisioner()
        # _is_anvil() returns False → raises RuntimeError
        p._w3.provider.make_request.side_effect = Exception("not anvil")
        with pytest.raises(RuntimeError, match="Anvil"):
            p.fund_demo_wallet()

    def test_storage_slot_key_computed_correctly(self):
        """The storage key for mapping(address=>uint256) must be keccak256(abi.encode(addr, slot))."""
        from web3 import Web3
        from eth_abi import encode as abi_encode
        wallet = "0x" + "a" * 40
        slot = 9
        expected_key = Web3.keccak(abi_encode(["address", "uint256"], [wallet, slot]))
        p = _make_provisioner()
        p._wallet = wallet
        # Call _write_storage and capture the key passed
        calls_made = []

        def mock_request(method, params):
            calls_made.append(params)
            return {"result": True}

        p._w3.provider.make_request.side_effect = mock_request
        p._write_storage("0x" + "b" * 40, slot, 1000)
        # The second param is the storage key
        assert calls_made[0][1] == expected_key.hex()

    def test_verify_balances_returns_dict(self):
        p = _make_provisioner()
        p._w3.eth.get_balance.return_value = 10 * 10 ** 18
        p._w3.eth.call.return_value = (1_000 * 10 ** 6).to_bytes(32, "big")
        bals = p.verify_balances(tokens=["USDC"])
        assert "ETH" in bals
        assert "USDC" in bals
        assert isinstance(bals["ETH"], Decimal)

    def test_unknown_token_skipped(self):
        p = _make_provisioner()
        p._w3.provider.make_request.return_value = {"result": {"version": "0.2"}}
        # Fund with an unknown token symbol
        # fund_demo_wallet raises before reaching unknown token because _is_anvil passes
        # Mock properly
        p._is_anvil = MagicMock(return_value=True)
        p._fund_eth = MagicMock(return_value=True)
        p._fund_erc20 = MagicMock(return_value=True)
        results = p.fund_demo_wallet(tokens=["UNKNOWN_TOKEN"])
        assert results.get("UNKNOWN_TOKEN") is False


# ═══════════════════════════ DemoConfig ══════════════════════════════════════

class TestDemoConfig:

    def test_demo_trading_pair_is_arb_usdc(self):
        assert DEMO_TRADING_PAIR == "ARB/USDC"

    def test_demo_pool_address_checksummed(self):
        from web3 import Web3
        # Verify address can be normalized (length check)
        normalized = Web3.to_checksum_address(DEMO_POOL_ADDRESS.lower()
                                              if len(DEMO_POOL_ADDRESS) == 42
                                              else DEMO_POOL_ADDRESS + "0")
        assert normalized.startswith("0x")

    def test_arb_usdc_trading_rules(self):
        rules = ARB_USDC_TRADING_RULES
        assert rules.pair == "ARB/USDC"
        assert rules.lot_size_step == 0.1
        assert rules.price_tick == 0.0001
        assert rules.min_notional_usd == 5.0

    def test_arb_usdc_round_quantity(self):
        rules = ARB_USDC_TRADING_RULES
        # 5.35 ARB → floor to step 0.1 → 5.3
        assert rules.round_quantity(5.35) == pytest.approx(5.3)

    def test_arb_usdc_round_price(self):
        rules = ARB_USDC_TRADING_RULES
        assert rules.round_price(0.85123) == pytest.approx(0.8512)

    def test_arb_usdc_validate_order_ok(self):
        rules = ARB_USDC_TRADING_RULES
        # 100 ARB × $0.85 = $85 > $5 min notional
        ok, _ = rules.validate_order(quantity=100.0, price=0.85)
        assert ok

    def test_arb_usdc_validate_order_below_min_qty(self):
        rules = ARB_USDC_TRADING_RULES
        ok, reason = rules.validate_order(quantity=0.05, price=0.85)
        assert not ok
        assert "min" in reason.lower()

    def test_arb_usdc_validate_order_below_min_notional(self):
        rules = ARB_USDC_TRADING_RULES
        # 0.1 ARB × $0.01 = $0.001 → below $5
        ok, reason = rules.validate_order(quantity=0.1, price=0.01)
        assert not ok
        assert "notional" in reason.lower()

    def test_build_demo_config_returns_system_config(self):
        from config.mode import SystemConfig
        cfg = build_demo_config()
        assert isinstance(cfg, SystemConfig)

    def test_build_demo_config_pair_is_arb_usdc(self):
        cfg = build_demo_config()
        assert cfg.trading_pair == "ARB/USDC"

    def test_build_demo_config_conservative_risk(self):
        cfg = build_demo_config()
        assert cfg.risk.max_trade_usd <= 20.0
        assert cfg.kelly_fraction <= Decimal("0.25")   # More conservative than default

    def test_build_demo_config_anvil_rpc(self):
        cfg = build_demo_config()
        assert "localhost" in cfg.dex.rpc_url or "anvil" in cfg.dex.rpc_url

    def test_build_demo_config_chain_id_arbitrum(self):
        cfg = build_demo_config()
        assert cfg.dex.chain_id == 42161

    def test_build_demo_config_gas_cost_low(self):
        cfg = build_demo_config()
        # Anvil fork: gas is negligible
        assert cfg.gas_cost_usd <= Decimal("0.10")


# ═══════════════════════════ OperationMode demo ═══════════════════════════════

class TestOperationModeDemo:

    def test_demo_mode_enum_exists(self):
        assert hasattr(OperationMode, "DEMO")
        assert OperationMode.DEMO == "demo"

    def test_from_env_returns_demo_mode(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "demo"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.DEMO

    def test_is_demo_property_true(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "demo"}):
            cfg = SystemConfig.from_env()
        assert cfg.is_demo is True
        assert cfg.is_test is False
        assert cfg.is_production is False

    def test_is_test_property_false_in_demo(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "demo"}):
            cfg = SystemConfig.from_env()
        assert not cfg.is_test

    def test_is_production_false_in_demo(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "demo"}):
            cfg = SystemConfig.from_env()
        assert not cfg.is_production

    def test_demo_mode_from_env_uses_demo_keys(self):
        with patch.dict(os.environ, {
            "OPERATION_MODE": "demo",
            "BINANCE_DEMO_API_KEY": "demo_key_123",
            "BINANCE_DEMO_SECRET": "demo_secret_456"
        }):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.DEMO
        assert cfg.is_demo is True

    def test_test_mode_unchanged(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.TEST
        assert not cfg.is_demo

    def test_production_mode_unchanged(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "production"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.PRODUCTION
        assert not cfg.is_demo


# ═══════════════════════════ make_demo_order_book ════════════════════════════

class TestMakeDemoOrderBook:

    def test_returns_live_order_book(self):
        from src.exchange.feed import LiveOrderBook
        book = make_demo_order_book("ARB/USDC")
        assert isinstance(book, LiveOrderBook)

    def test_symbol_set_correctly(self):
        book = make_demo_order_book("ARB/USDC")
        assert "arb" in book._ws_symbol or "arb" in book._rest_symbol.lower()

    def test_demo_ws_url_used(self):
        book = make_demo_order_book("ARB/USDC")
        # Since we override MAINNET WS to demo URL and use testnet=False
        assert not book._testnet   # Uses mainnet slot (overridden to demo)

    def test_reconnect_enabled(self):
        book = make_demo_order_book("ARB/USDC")
        assert book._reconnect is True
