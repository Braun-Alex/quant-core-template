"""
Unit tests for OrderBookAnalyser.

Uses synthetic order book data; no network calls.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from src.exchange.orderbook import OrderBookAnalyzer


def _snapshot(
    bids=None,
    asks=None,
    mid="2001.5",
    symbol="ETH/USDT"
):
    bids = bids or [
        (Decimal("2001"), Decimal("1.5")),
        (Decimal("2000"), Decimal("2.0")),
        (Decimal("1999"), Decimal("0.5"))
    ]
    asks = asks or [
        (Decimal("2002"), Decimal("1.0")),
        (Decimal("2003"), Decimal("0.8")),
        (Decimal("2004"), Decimal("2.0"))
    ]
    mid_p = Decimal(mid)
    return {
        "symbol": symbol,
        "timestamp": 1700000000000,
        "bids": bids,
        "asks": asks,
        "best_bid": bids[0],
        "best_ask": asks[0],
        "mid_price": mid_p,
        "spread_bps": (asks[0][0] - bids[0][0]) / mid_p * Decimal("10000")
    }


def _analyzer(**kw) -> OrderBookAnalyzer:
    return OrderBookAnalyzer(_snapshot(**kw))


class TestSimulateFill:
    def test_exact_fill_single_level_buy(self):
        a = _analyzer()
        result = a.simulate_fill("buy", 1.0)
        assert result["fully_filled"] is True
        assert result["avg_price"] == Decimal("2002")
        assert result["levels_consumed"] == 1

    def test_total_cost_equals_fill_costs(self):
        a = _analyzer()
        result = a.simulate_fill("buy", 2.5)
        assert result["total_cost"] == sum(f["cost"] for f in result["fills"])

    def test_slippage_zero_at_single_level(self):
        a = _analyzer()
        result = a.simulate_fill("buy", 0.5)
        assert result["slippage_bps"] == Decimal("0")

    def test_invalid_direction_raises(self):
        a = _analyzer()
        with pytest.raises(ValueError, match="direction"):
            a.simulate_fill("long", 1.0)

    def test_exact_fill_single_level_sell(self):
        a = _analyzer()
        result = a.simulate_fill("sell", 1.5)
        assert result["fully_filled"] is True
        assert result["avg_price"] == Decimal("2001")

    def test_multiple_levels_avg_price_correct(self):
        a = _analyzer()
        result = a.simulate_fill("buy", 1.8)
        assert result["fully_filled"] is True
        assert result["levels_consumed"] == 2
        expected = (Decimal("2002") * 1 + Decimal("2003") * Decimal("0.8")) / Decimal("1.8")
        assert abs(result["avg_price"] - expected) < Decimal("1e-10")

    def test_insufficient_liquidity_returns_false(self):
        a = _analyzer()
        result = a.simulate_fill("buy", 100.0)
        assert result["fully_filled"] is False

    def test_zero_qty_raises(self):
        a = _analyzer()
        with pytest.raises(ValueError, match="positive"):
            a.simulate_fill("buy", 0)

    def test_negative_qty_raises(self):
        a = _analyzer()
        with pytest.raises(ValueError, match="positive"):
            a.simulate_fill("sell", -1.0)


class TestLiquidityBand:
    def test_bid_depth_10_bps_correct(self):
        a = _analyzer()
        # Best bid = 2001, threshold = 2001 × 0.999 = 1998.999
        # All bids (2001, 2000, 1999) qualify
        depth = a.liquidity_band("bid", 10)
        assert depth == Decimal("1.5") + Decimal("2.0") + Decimal("0.5")

    def test_ask_depth_10_bps_correct(self):
        a = _analyzer()
        depth = a.liquidity_band("ask", 10)
        assert depth == Decimal("1.0") + Decimal("0.8") + Decimal("2.0")

    def test_tight_band_excludes_far_levels(self):
        a = _analyzer()
        narrow = a.liquidity_band("bid", 1)
        wide = a.liquidity_band("bid", 50)
        assert narrow <= wide

    def test_empty_book_returns_zero(self):
        a = OrderBookAnalyzer({"bids": [], "asks": [], "mid_price": Decimal("0")})
        assert a.liquidity_band("bid", 10) == Decimal("0")
        assert a.liquidity_band("ask", 10) == Decimal("0")

    def test_invalid_side_raises(self):
        a = _analyzer()
        with pytest.raises(ValueError, match="side"):
            a.liquidity_band("buy", 10)

    def test_returns_decimal(self):
        a = _analyzer()
        assert isinstance(a.liquidity_band("bid", 10), Decimal)


class TestPressureRatio:
    def test_range_always_valid(self):
        a = _analyzer()
        assert -1.0 <= a.pressure_ratio() <= 1.0

    def test_bid_heavy_positive(self):
        bids = [(Decimal("100"), Decimal("3"))]
        asks = [(Decimal("101"), Decimal("1"))]
        a = OrderBookAnalyzer({"bids": bids, "asks": asks, "mid_price": Decimal("100.5")})
        assert a.pressure_ratio() > 0.0

    def test_ask_heavy_negative(self):
        bids = [(Decimal("100"), Decimal("1"))]
        asks = [(Decimal("101"), Decimal("3"))]
        a = OrderBookAnalyzer({"bids": bids, "asks": asks, "mid_price": Decimal("100.5")})
        assert a.pressure_ratio() < 0.0

    def test_balanced_book_near_zero(self):
        bids = [(Decimal("100"), Decimal("1"))]
        asks = [(Decimal("101"), Decimal("1"))]
        a = OrderBookAnalyzer({"bids": bids, "asks": asks, "mid_price": Decimal("100.5")})
        assert a.pressure_ratio() == 0.0


class TestRoundTripCost:
    def test_increases_with_quantity(self):
        a = _analyzer()
        small = a.round_trip_cost(0.5)
        large = a.round_trip_cost(3.5)
        assert large >= small

    def test_greater_than_or_equal_to_quoted_spread(self):
        a = _analyzer()
        eff = a.round_trip_cost(2.0)
        quoted = a.quoted_spread_bps
        assert eff >= quoted

    def test_zero_when_empty(self):
        a = OrderBookAnalyzer({"bids": [], "asks": [], "mid_price": Decimal("0")})
        assert a.round_trip_cost(1.0) == Decimal("0")
