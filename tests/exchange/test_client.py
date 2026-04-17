"""
Unit tests for BinanceClient.

All ccxt calls are mocked; no real network calls are made.
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.exchange.client import BinanceClient, _to_dec

_CFG = {"apiKey": "k", "secret": "s", "sandbox": True}


def _make_client():
    mock_ex = MagicMock()
    mock_ex.id = "binance"
    mock_ex.fetch_time.return_value = {}
    with patch("ccxt.binance", return_value=mock_ex):
        client = BinanceClient(_CFG)
    return client, mock_ex


def _raw_order(status="closed", filled=1.0, amount=1.0, price=2000.0, side="buy"):
    return {
        "id": "123", "symbol": "ETH/USDT", "side": side, "type": "limit",
        "timeInForce": "IOC", "status": status, "filled": filled,
        "amount": amount, "average": price, "price": price,
        "fee": {"cost": 0.002, "currency": "USDT"},
        "timestamp": 1700000000000, "info": {"timeInForce": "IOC"}
    }


class TestConstruction:
    def test_health_check_on_init(self):
        mock_ex = MagicMock()
        mock_ex.fetch_time.return_value = {}
        with patch("ccxt.binance", return_value=mock_ex):
            BinanceClient(_CFG)
        mock_ex.fetch_time.assert_called_once()

    def test_missing_ccxt_raises(self):
        with patch.dict("sys.modules", {"ccxt": None}):
            with pytest.raises(ImportError, match="ccxt"):
                BinanceClient({})

    def test_auth_error_propagates(self):
        import ccxt
        mock_ex = MagicMock()
        mock_ex.fetch_time.side_effect = ccxt.AuthenticationError("bad key")
        with patch("ccxt.binance", return_value=mock_ex):
            with pytest.raises(ccxt.AuthenticationError):
                BinanceClient(_CFG)


class TestOrderBook:
    def setup_method(self):
        self.client, self.mock_ex = _make_client()
        self.mock_ex.fetch_order_book.return_value = {
            "symbol": "ETH/USDT",
            "timestamp": 1700000000000,
            "bids": [[2001.0, 1.5], [2000.0, 2.0]],
            "asks": [[2002.0, 1.0], [2003.0, 0.8]]
        }

    def test_returns_required_fields(self):
        book = self.client.fetch_order_book("ETH/USDT")
        for key in ("symbol", "timestamp", "bids", "asks", "best_bid", "best_ask", "mid_price", "spread_bps"):
            assert key in book

    def test_spread_bps_is_decimal(self):
        book = self.client.fetch_order_book("ETH/USDT")
        assert isinstance(book["spread_bps"], Decimal)

    def test_mid_price_correct(self):
        book = self.client.fetch_order_book("ETH/USDT")
        expected = (Decimal("2001") + Decimal("2002")) / Decimal("2")
        assert book["mid_price"] == expected

    def test_empty_book_no_crash(self):
        self.mock_ex.fetch_order_book.return_value = {
            "symbol": "ETH/USDT", "timestamp": 0, "bids": [], "asks": []
        }
        book = self.client.fetch_order_book("ETH/USDT")
        assert book["spread_bps"] == Decimal("0")

    def test_bids_sorted_descending(self):
        book = self.client.fetch_order_book("ETH/USDT")
        prices = [b[0] for b in book["bids"]]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        book = self.client.fetch_order_book("ETH/USDT")
        prices = [a[0] for a in book["asks"]]
        assert prices == sorted(prices)


class TestFetchBalance:
    def setup_method(self):
        self.client, self.mock_ex = _make_client()
        self.mock_ex.fetch_balance.return_value = {
            "ETH": {"free": "10.5", "used": "0", "total": "10.5"},
            "USDT": {"free": "20000", "used": "500", "total": "20500"},
            "BNB": {"free": "0", "used": "0", "total": "0"},
            "info": "metadata"
        }

    def test_values_are_decimal(self):
        bal = self.client.fetch_balance()
        for info in bal.values():
            assert isinstance(info["free"], Decimal)
            assert isinstance(info["locked"], Decimal)

    def test_locked_maps_from_used(self):
        bal = self.client.fetch_balance()
        assert bal["USDT"]["locked"] == Decimal("500")

    def test_filters_zero_balance_assets(self):
        bal = self.client.fetch_balance()
        assert "BNB" not in bal

    def test_filters_non_dict_entries(self):
        bal = self.client.fetch_balance()
        assert "info" not in bal


class TestRateLimiter:
    def setup_method(self):
        self.client, self.mock_ex = _make_client()
        self.mock_ex.fetch_order_book.return_value = {
            "symbol": "X", "timestamp": 0, "bids": [], "asks": []
        }

    def test_weight_resets_after_window(self):
        self.client._weight_consumed = 500
        self.client._window_resets_at = time.monotonic() - 1
        self.client.fetch_order_book("ETH/USDT")
        assert self.client._weight_consumed < 500

    def test_weight_accumulates(self):
        before = self.client._weight_consumed
        self.client.fetch_order_book("ETH/USDT")
        assert self.client._weight_consumed > before

    def test_blocks_when_budget_exhausted(self):
        self.client._weight_consumed = 1079
        self.client._window_resets_at = time.monotonic() + 0.05
        with patch("time.sleep") as mock_sleep:
            self.client.fetch_order_book("ETH/USDT")
            mock_sleep.assert_called()


class TestOrders:
    def setup_method(self):
        self.client, self.mock_ex = _make_client()

    def test_limit_ioc_order_calls_exchange(self):
        self.mock_ex.create_order.return_value = _raw_order()
        self.client.create_limit_ioc_order("ETH/USDT", "buy", 1.0, 2000.0)
        self.mock_ex.create_order.assert_called_once_with(
            "ETH/USDT", "limit", "buy", 1.0, 2000.0, {"timeInForce": "IOC"}
        )

    def test_filled_status(self):
        self.mock_ex.create_order.return_value = _raw_order(status="closed", filled=1.0, amount=1.0)
        result = self.client.create_limit_ioc_order("ETH/USDT", "buy", 1.0, 2000.0)
        assert result["status"] == "filled"

    def test_partial_fill_status(self):
        self.mock_ex.create_order.return_value = _raw_order(status="closed", filled=0.5, amount=1.0)
        result = self.client.create_limit_ioc_order("ETH/USDT", "buy", 1.0, 2000.0)
        assert result["status"] == "partially_filled"

    def test_expired_status(self):
        self.mock_ex.create_order.return_value = _raw_order(status="canceled", filled=0.0, amount=1.0)
        result = self.client.create_limit_ioc_order("ETH/USDT", "buy", 1.0, 2000.0)
        assert result["status"] == "expired"

    def test_amount_filled_is_decimal(self):
        self.mock_ex.create_order.return_value = _raw_order(filled=0.75)
        result = self.client.create_limit_ioc_order("ETH/USDT", "buy", 1.0, 2000.0)
        assert isinstance(result["amount_filled"], Decimal)

    def test_cancel_calls_exchange(self):
        self.mock_ex.cancel_order.return_value = _raw_order(status="canceled")
        self.client.cancel_order("123", "ETH/USDT")
        self.mock_ex.cancel_order.assert_called_once_with("123", "ETH/USDT")


class TestToDecHelper:
    def test_none_returns_zero(self):
        assert _to_dec(None) == Decimal("0")

    def test_float_converts(self):
        assert _to_dec(1.5) == Decimal("1.5")

    def test_invalid_string_returns_zero(self):
        assert _to_dec("not_a_number") == Decimal("0")

    def test_integer_converts(self):
        assert _to_dec(42) == Decimal("42")
