"""
Tests for PnLTracker, ArbTrade, TradeLeg.
"""

from __future__ import annotations

import csv
import os
import tempfile
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from src.inventory.pnl import ArbTrade, PnLTracker, TradeLeg
from src.inventory.tracker import Venue

_TS = datetime(2024, 1, 15, 14, 30, 0, tzinfo=UTC)


def _leg(side: str, price: str, amount: str = "1", fee: str = "0.40") -> TradeLeg:
    return TradeLeg(
        leg_id=f"{side}-leg",
        executed_at=_TS,
        venue=Venue.BINANCE if side == "sell" else Venue.WALLET,
        symbol="ETH/USDT",
        side=side,
        quantity=Decimal(amount),
        price=Decimal(price),
        fee=Decimal(fee),
        fee_currency="USDT"
    )


def _trade(
    buy_price="2000", sell_price="2001.25",
    amount="1", buy_fee="0.40", sell_fee="0.40", gas="0",
) -> ArbTrade:
    return ArbTrade(
        trade_id="arb-1",
        opened_at=_TS,
        buy_leg=_leg("buy", buy_price, amount, buy_fee),
        sell_leg=_leg("sell", sell_price, amount, sell_fee),
        gas_usd=Decimal(gas)
    )


class TestArbTradeProperties:
    def test_gross_pnl(self):
        t = _trade(buy_price="2000", sell_price="2001.25")
        assert t.gross_pnl == Decimal("1.25")

    def test_gross_pnl_negative_when_inverted(self):
        t = _trade(buy_price="2001", sell_price="2000")
        assert t.gross_pnl == Decimal("-1")

    def test_all_fees_sum(self):
        t = _trade(buy_fee="0.40", sell_fee="0.60", gas="0.10")
        assert t.all_fees == Decimal("1.10")

    def test_net_pnl_bps(self):
        t = _trade(buy_price="2000", sell_price="2002", amount="1", buy_fee="0.40", sell_fee="0.40", gas="0")
        assert t.net_pnl_bps == Decimal("6.0")

    def test_net_pnl_includes_all_fees(self):
        t = _trade(buy_price="2000", sell_price="2002", buy_fee="0.40", sell_fee="0.40", gas="0.20")
        assert t.net_pnl == Decimal("1.00")

    def test_net_pnl_bps_zero_notional(self):
        t = _trade(buy_price="0", sell_price="1")
        assert t.net_pnl_bps == Decimal("0")


class TestPnLTrackerRecord:
    def test_starts_empty(self):
        assert PnLTracker().trades == []

    def test_record_appends(self):
        tracker = PnLTracker()
        tracker.record(_trade())
        assert len(tracker.trades) == 1


class TestSummaryEmpty:
    def test_no_trades_returns_zeros(self):
        s = PnLTracker().summary()
        assert s["total_trades"] == 0
        assert s["total_pnl_usd"] == Decimal("0")
        assert s["win_rate"] == 0.0
        assert s["sharpe_estimate"] == 0.0
        assert s["pnl_by_hour"] == {}


class TestSummaryWithTrades:
    def setup_method(self):
        self.tracker = PnLTracker()
        for bp, sp in [("2000", "2001.25"), ("2000", "2001.90"), ("2002", "2001.40"), ("2000", "2001.80")]:
            self.tracker.record(_trade(buy_price=bp, sell_price=sp))
        self.s = self.tracker.summary()

    def test_total_trades(self):
        assert self.s["total_trades"] == 4

    def test_win_rate(self):
        assert self.s["win_rate"] == pytest.approx(75.0)   # 75%

    def test_total_fees_correct(self):
        # 4 * (0.40 + 0.40) = 3.20
        assert self.s["total_fees_usd"] == Decimal("3.20")

    def test_best_and_worst_trade(self):
        assert self.s["best_trade_pnl"] > self.s["worst_trade_pnl"]

    def test_pnl_by_hour_sum_equals_total(self):
        hour_sum = sum(self.s["pnl_by_hour"].values(), Decimal("0"))
        assert hour_sum == self.s["total_pnl_usd"]


class TestRecent:
    def setup_method(self):
        self.tracker = PnLTracker()
        for i in range(5):
            ts = _TS + timedelta(minutes=i)
            buy = TradeLeg("b", ts, Venue.WALLET, "ETH/USDT", "buy", Decimal("1"),
                           Decimal("2000"), Decimal("0.4"), "USDT")
            sell = TradeLeg("s", ts, Venue.BINANCE, "ETH/USDT", "sell", Decimal("1"),
                            Decimal("2001"), Decimal("0.4"), "USDT")
            self.tracker.record(ArbTrade(f"arb-{i}", ts, buy, sell))

    def test_empty_tracker_returns_empty(self):
        assert PnLTracker().recent() == []

    def test_returns_list(self):
        assert isinstance(self.tracker.recent(), list)

    def test_returns_n_items(self):
        assert len(self.tracker.recent(3)) == 3

    def test_fewer_than_n_returns_all(self):
        assert len(self.tracker.recent(100)) == 5

    def test_most_recent_first(self):
        r = self.tracker.recent(5)
        times = [item["opened_at"] for item in r]
        assert times == sorted(times, reverse=True)


class TestExportCSV:
    def setup_method(self):
        self.tracker = PnLTracker()
        self.tracker.record(_trade("2000", "2001.25", "1", "0.40", "0.40", "0.10"))
        self.tracker.record(_trade("2001", "2002.00", "1", "0.40", "0.40"))
        self.tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self.tmp.close()
        self.path = self.tmp.name

    def teardown_method(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_creates_file_with_rows(self):
        self.tracker.export_csv(self.path)
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 2

    def test_empty_tracker_writes_header_only(self):
        PnLTracker().export_csv(self.path)
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert rows == []

    def test_expected_columns_present(self):
        self.tracker.export_csv(self.path)
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        expected = {"trade_id", "opened_at", "symbol", "buy_venue", "sell_venue",
                    "buy_price", "sell_price", "quantity", "gross_pnl", "net_pnl", "gas_usd"}
        assert expected <= set(rows[0].keys())

    def test_net_pnl_correct(self):
        self.tracker.export_csv(self.path)
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        assert Decimal(rows[0]["net_pnl"]) == Decimal("0.35")
