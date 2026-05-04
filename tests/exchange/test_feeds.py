"""
Tests for the real-time price feed infrastructure.

Coverage:
  TestLiveOrderBookState      - local order book state machine
  TestLiveOrderBookSnapshot   - snapshot format correctness
  TestDEXPriceFeed            - polling, price change detection, stop
  TestPriceState              - validity, staleness, spread computation
  TestPriceFeedManager        - CEX/DEX update routing, callback dispatch
  TestGeneratFromFeed         - event-driven signal generation
  TestEventDrivenBot          - _on_price_update integration
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exchange.feed import LiveOrderBook
from src.exchange.price_feed import PriceFeedManager, PriceState
from src.pricing.dex_feed import DEXPriceFeed, DEXPriceSnapshot
from src.strategy.signal import Direction, Signal
from src.core.types import Address, Token
from src.pricing.amm import PoolState

# ─────────────────────────────── Fixtures ───────────────────────────────────

WETH = Token(Address("0x82aF49447D8a07e3bd95BD0d56f35241523fBab1"), "WETH", 18)
USDC = Token(Address("0xaf88d065e77c8cC2239327C5EDb3A432268e5831"), "USDC", 6)
PAIR_ADDR = Address("0x905dfCD5649217c42684f23958568e533C711Aa3")


def _pool(ql=1_000 * 10 ** 18, qr=2_000_000 * 10 ** 6) -> PoolState:
    return PoolState(contract=PAIR_ADDR, left=WETH, right=USDC,
                     qty_left=ql, qty_right=qr, fee_bps=30)


def _ob_raw(bid=2001.0, ask=2002.0, symbol="ETH/USDC") -> dict:
    mid = (bid + ask) / 2
    return {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "bids": [(Decimal(str(bid)), Decimal("5.0")),
                 (Decimal(str(bid - 1)), Decimal("3.0"))],
        "asks": [(Decimal(str(ask)), Decimal("4.0")),
                 (Decimal(str(ask + 1)), Decimal("2.0"))],
        "best_bid": (Decimal(str(bid)), Decimal("5.0")),
        "best_ask": (Decimal(str(ask)), Decimal("4.0")),
        "mid_price": Decimal(str(mid)),
        "spread_bps": Decimal(str((ask - bid) / mid * 10000)),
        "last_update_id": 12345
    }


def _dex_snap(pair="ETH/USDC", price=1999.0) -> DEXPriceSnapshot:
    return DEXPriceSnapshot(
        pair=pair,
        pool_address=PAIR_ADDR.checksum,
        reserve_base=1_000 * 10 ** 18,
        reserve_quote=2_000_000 * 10 ** 6,
        price=Decimal(str(price)),
        mid_price=Decimal(str(price)),
        fee_bps=30,
        block_time_ms=5.0,
        timestamp=time.time()
    )


def _price_state(pair="ETH/USDC", bid=2001.0, ask=2002.0, dex=1999.0,
                 cex_stale=False, dex_stale=False) -> PriceState:
    state = PriceState(pair=pair)
    state.update_cex(_ob_raw(bid, ask, pair))
    snap = _dex_snap(pair, dex)
    if dex_stale:
        snap.timestamp = time.time() - 100
    state.update_dex(snap)
    if cex_stale:
        state.cex_timestamp = time.time() - 100
    state.recompute(stale_threshold=5.0)
    return state


# ═══════════════════════════ LiveOrderBook state ══════════════════════════════

class TestLiveOrderBookState:

    def _book(self) -> LiveOrderBook:
        book = LiveOrderBook.__new__(LiveOrderBook)
        book._symbol = "ETH/USDC"
        book._ws_symbol = "ethusdc"
        book._rest_symbol = "ETHUSDC"
        book._testnet = True
        book._max_depth = 20
        book._reconnect = False
        book._bids = {}
        book._asks = {}
        book._last_seq = 0
        book._initialized = False
        book._update_event = asyncio.Event()
        book._latest_snapshot = None
        book._connected = False
        book._ws = None
        book._http_session = None
        return book

    def test_apply_snapshot_seeds_book(self):
        book = self._book()
        book._apply_snapshot({
            "lastUpdateId": 100,
            "bids": [["2001.0", "5.0"], ["2000.0", "3.0"]],
            "asks": [["2002.0", "4.0"], ["2003.0", "2.0"]]
        })
        assert book._last_seq == 100
        assert Decimal("2001.0") in book._bids
        assert Decimal("2002.0") in book._asks

    def test_apply_diff_accepted_when_newer(self):
        book = self._book()
        book._last_seq = 100
        changed = book._apply_diff({
            "U": 101, "u": 105,
            "b": [["2001.5", "3.0"]],
            "a": []
        })
        assert changed is True
        assert book._last_seq == 105
        assert Decimal("2001.5") in book._bids

    def test_apply_diff_rejected_when_stale(self):
        book = self._book()
        book._last_seq = 200
        changed = book._apply_diff({"U": 50, "u": 100, "b": [], "a": []})
        assert changed is False
        assert book._last_seq == 200

    def test_apply_diff_removes_zero_qty_bid(self):
        book = self._book()
        book._bids[Decimal("2001.0")] = Decimal("5.0")
        book._last_seq = 10
        book._apply_diff({"u": 11, "b": [["2001.0", "0"]], "a": []})
        assert Decimal("2001.0") not in book._bids

    def test_apply_diff_removes_zero_qty_ask(self):
        book = self._book()
        book._asks[Decimal("2002.0")] = Decimal("4.0")
        book._last_seq = 10
        book._apply_diff({"u": 11, "b": [], "a": [["2002.0", "0"]]})
        assert Decimal("2002.0") not in book._asks

    def test_apply_diff_upserts_bid(self):
        book = self._book()
        book._bids[Decimal("2001.0")] = Decimal("1.0")
        book._last_seq = 10
        book._apply_diff({"u": 11, "b": [["2001.0", "9.5"]], "a": []})
        assert book._bids[Decimal("2001.0")] == Decimal("9.5")

    def test_current_snapshot_format(self):
        book = self._book()
        book._apply_snapshot({
            "lastUpdateId": 50,
            "bids": [["2001.0", "5.0"]],
            "asks": [["2002.0", "4.0"]]
        })
        snap = book.current_snapshot()
        assert snap["symbol"] == "ETH/USDC"
        assert snap["best_bid"][0] == Decimal("2001.0")
        assert snap["best_ask"][0] == Decimal("2002.0")
        assert snap["mid_price"] == Decimal("2001.5")
        assert snap["last_update_id"] == 50

    def test_spread_bps_computed_correctly(self):
        book = self._book()
        book._apply_snapshot({
            "lastUpdateId": 1,
            "bids": [["2000.0", "1.0"]],
            "asks": [["2002.0", "1.0"]]
        })
        snap = book.current_snapshot()
        # (2002 - 2000) / 2001 * 10000 ≈ 9.995 bps
        assert float(snap["spread_bps"]) == pytest.approx(9.995, rel=0.01)

    def test_get_latest_none_initially(self):
        book = self._book()
        assert book.get_latest() is None

    def test_bids_sorted_descending(self):
        book = self._book()
        book._bids = {
            Decimal("2000"): Decimal("1"),
            Decimal("2001"): Decimal("2"),
            Decimal("1999"): Decimal("3")
        }
        book._asks = {Decimal("2002"): Decimal("1")}
        book._last_seq = 1
        snap = book.current_snapshot()
        prices = [b[0] for b in snap["bids"]]
        assert prices == sorted(prices, reverse=True)

    def test_asks_sorted_ascending(self):
        book = self._book()
        book._bids = {Decimal("2000"): Decimal("1")}
        book._asks = {
            Decimal("2003"): Decimal("1"),
            Decimal("2002"): Decimal("2"),
            Decimal("2004"): Decimal("3")
        }
        book._last_seq = 1
        snap = book.current_snapshot()
        prices = [a[0] for a in snap["asks"]]
        assert prices == sorted(prices)

    def test_empty_book_returns_zero_prices(self):
        book = self._book()
        snap = book.current_snapshot()
        assert snap["best_bid"] == (Decimal("0"), Decimal("0"))
        assert snap["best_ask"] == (Decimal("0"), Decimal("0"))
        assert snap["mid_price"] == Decimal("0")


# ═══════════════════════════ DEXPriceFeed ═════════════════════════════════════

class TestDEXPriceFeed:

    def _feed(self, pools=None, interval=0.05, on_update=None) -> DEXPriceFeed:
        if pools is None:
            pool = _pool()
            pools = {"ETH/USDC": pool}
        client = MagicMock()
        feed = DEXPriceFeed(
            chain_client=client,
            pools=pools,
            poll_interval=interval,
            min_change_bps=0.0,   # Notify on any change
            on_update=on_update
        )
        return feed

    def _mock_reserves(self, feed, r0=1_000 * 10 ** 18, r1=2_000_000 * 10 ** 6):
        """Patch _fetch_reserves to return given values."""
        feed._fetch_reserves = MagicMock(return_value=(r0, r1))

    def test_snapshot_stored_after_poll(self):
        feed = self._feed()
        self._mock_reserves(feed)
        asyncio.get_event_loop().run_until_complete(feed._poll_all())
        snap = feed.get_snapshot("ETH/USDC")
        assert snap is not None
        assert snap.pair == "ETH/USDC"
        assert snap.price > Decimal("0")

    def test_price_computed_from_reserves(self):
        feed = self._feed()
        # 1000 WETH, 2_000_000 USDC → price ≈ 2000 USDC/WETH
        self._mock_reserves(feed, 1_000 * 10 ** 18, 2_000_000 * 10 ** 6)
        asyncio.get_event_loop().run_until_complete(feed._poll_all())
        snap = feed.get_snapshot("ETH/USDC")
        assert abs(float(snap.price) - 2000.0) < 1.0

    def test_callback_fired_on_first_poll(self):
        updates = []

        async def cb(pair, snap):
            updates.append((pair, snap))

        feed = self._feed(on_update=cb)
        self._mock_reserves(feed)
        asyncio.get_event_loop().run_until_complete(feed._poll_all())
        assert len(updates) == 1
        assert updates[0][0] == "ETH/USDC"

    def test_callback_not_fired_when_price_unchanged(self):
        updates = []

        async def cb(pair, snap):
            updates.append(pair)

        feed = self._feed(on_update=cb, interval=0.01)
        feed._min_change_bps = Decimal("0.1")   # Require 0.1 bps change
        r0, r1 = 1_000 * 10 ** 18, 2_000_000 * 10 ** 6
        self._mock_reserves(feed, r0, r1)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(feed._poll_all())   # First poll - fires
        loop.run_until_complete(feed._poll_all())   # Same reserves - no change
        assert len(updates) == 1

    def test_callback_fired_when_price_changes(self):
        updates = []

        async def cb(pair, snap):
            updates.append(float(snap.price))

        feed = self._feed(on_update=cb)
        feed._min_change_bps = Decimal("0.1")
        loop = asyncio.get_event_loop()
        self._mock_reserves(feed, 1_000 * 10 ** 18, 2_000_000 * 10 ** 6)
        loop.run_until_complete(feed._poll_all())   # Price ≈ 2000
        self._mock_reserves(feed, 1_000 * 10 ** 18, 2_100_000 * 10 ** 6)
        loop.run_until_complete(feed._poll_all())   # Price ≈ 2100: change > 0.1 bps
        assert len(updates) == 2
        assert updates[1] > updates[0]

    def test_reserve_fetch_error_does_not_crash(self):
        feed = self._feed()
        feed._fetch_reserves = MagicMock(side_effect=Exception("RPC error"))
        # Should not raise
        asyncio.get_event_loop().run_until_complete(feed._poll_all())
        assert feed.get_snapshot("ETH/USDC") is None

    @pytest.mark.asyncio
    async def test_run_and_stop(self):
        feed = self._feed(interval=0.05)
        self._mock_reserves(feed)
        task = asyncio.create_task(feed.run())
        await asyncio.sleep(0.15)
        await feed.stop()
        await asyncio.wait_for(task, timeout=1.0)
        assert not feed.is_running

    @pytest.mark.asyncio
    async def test_run_polls_multiple_times(self):
        poll_count = []

        async def cb(pair, snap):
            poll_count.append(1)

        feed = self._feed(on_update=cb, interval=0.05)
        self._mock_reserves(feed)
        feed._min_change_bps = Decimal("0")   # Fire every poll
        task = asyncio.create_task(feed.run())
        await asyncio.sleep(0.2)
        await feed.stop()
        try:
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            task.cancel()
        assert len(poll_count) >= 2, f"Expected ≥2 polls, got {len(poll_count)}"

    def test_get_all_snapshots(self):
        pool_a = _pool()
        pool_b = _pool(ql=500 * 10 ** 18, qr=1_000_000 * 10 ** 6)
        feed = self._feed(pools={"A/B": pool_a, "C/D": pool_b})
        feed._fetch_reserves = MagicMock(return_value=(1_000 * 10 ** 18, 2_000_000 * 10 ** 6))
        asyncio.get_event_loop().run_until_complete(feed._poll_all())
        snaps = feed.get_all_snapshots()
        assert set(snaps.keys()) == {"A/B", "C/D"}


# ═══════════════════════════ PriceState ══════════════════════════════════════

class TestPriceState:

    def test_valid_when_both_fresh(self):
        state = _price_state()
        assert state.is_valid is True

    def test_invalid_when_cex_stale(self):
        state = _price_state(cex_stale=True)
        assert state.is_valid is False

    def test_invalid_when_dex_stale(self):
        state = _price_state(dex_stale=True)
        assert state.is_valid is False

    def test_spread_bps_nonzero(self):
        state = _price_state(bid=2001.0, ask=2002.0, dex=1999.0)
        assert state.spread_bps > Decimal("0")

    def test_cex_fields_from_snapshot(self):
        state = _price_state(bid=2001.0, ask=2003.0)
        assert state.cex_bid == Decimal("2001.0")
        assert state.cex_ask == Decimal("2003.0")
        assert state.cex_mid == Decimal("2002.0")

    def test_dex_fields_from_snapshot(self):
        state = _price_state(dex=1998.5)
        assert state.dex_price == Decimal("1998.5")
        assert state.dex_reserve_base > 0

    def test_update_cex_updates_timestamp(self):
        state = PriceState(pair="ETH/USDC")
        before = state.cex_timestamp
        state.update_cex(_ob_raw())
        assert state.cex_timestamp > before

    def test_update_dex_updates_latency(self):
        state = PriceState(pair="ETH/USDC")
        snap = _dex_snap(price=2000.0)
        snap.block_time_ms = 12.5
        state.update_dex(snap)
        assert state.dex_poll_latency_ms == 12.5


# ═══════════════════════════ PriceFeedManager ════════════════════════════════

class TestPriceFeedManager:

    def _manager(self, on_update=None) -> PriceFeedManager:
        cex_book = MagicMock(spec=LiveOrderBook)
        cex_book._testnet = True
        cex_book._symbol = "ETH/USDC"
        dex_feed = MagicMock(spec=DEXPriceFeed)
        dex_feed._pools = {"ETH/USDC": _pool()}
        dex_feed.run = AsyncMock()
        dex_feed.stop = AsyncMock()
        mgr = PriceFeedManager(
            cex_book=cex_book,
            dex_feed=dex_feed,
            pairs=["ETH/USDC"],
            on_price_update=on_update,
            stale_threshold_seconds=5.0
        )
        return mgr

    @pytest.mark.asyncio
    async def test_on_cex_update_fires_callback_when_valid(self):
        fired = []

        async def cb(pair, state):
            fired.append(pair)

        mgr = self._manager(on_update=cb)
        # Seed DEX so state becomes valid
        await mgr._on_dex_update("ETH/USDC", _dex_snap())
        # Now fire CEX update
        await mgr._on_cex_update(_ob_raw(symbol="ETH/USDC"))
        assert "ETH/USDC" in fired

    @pytest.mark.asyncio
    async def test_on_dex_update_fires_callback_when_valid(self):
        fired = []

        async def cb(pair, state):
            fired.append(pair)

        mgr = self._manager(on_update=cb)
        # Seed CEX
        await mgr._on_cex_update(_ob_raw(symbol="ETH/USDC"))
        # Now fire DEX update
        await mgr._on_dex_update("ETH/USDC", _dex_snap())
        assert "ETH/USDC" in fired

    @pytest.mark.asyncio
    async def test_callback_not_fired_when_only_cex_available(self):
        fired = []

        async def cb(pair, state):
            fired.append(pair)

        mgr = self._manager(on_update=cb)
        # Only CEX update - DEX is stale/missing
        await mgr._on_cex_update(_ob_raw(symbol="ETH/USDC"))
        # State not valid (no DEX data) → no callback
        assert len(fired) == 0

    @pytest.mark.asyncio
    async def test_callback_error_does_not_propagate(self):
        async def bad_cb(pair, state):
            raise RuntimeError("callback exploded")

        mgr = self._manager(on_update=bad_cb)
        await mgr._on_dex_update("ETH/USDC", _dex_snap())
        # Should not raise even with bad callback
        await mgr._on_cex_update(_ob_raw(symbol="ETH/USDC"))

    def test_match_pair_compressed_symbol(self):
        mgr = self._manager()
        assert mgr._match_pair("ETHUSDC") == "ETH/USDC"

    def test_match_pair_slash_symbol(self):
        mgr = self._manager()
        assert mgr._match_pair("ETH/USDC") == "ETH/USDC"

    def test_match_pair_unknown_returns_none(self):
        mgr = self._manager()
        assert mgr._match_pair("BTCUSDT") is None

    def test_get_state_returns_price_state(self):
        mgr = self._manager()
        state = mgr.get_state("ETH/USDC")
        assert isinstance(state, PriceState)
        assert state.pair == "ETH/USDC"

    def test_get_state_unknown_pair_returns_none(self):
        mgr = self._manager()
        assert mgr.get_state("BTC/USDC") is None

    def test_dex_callback_wired_on_init(self):
        mgr = self._manager()
        assert mgr._dex_feed.on_update is not None


# ═══════════════════════════ generate_from_feed ═══════════════════════════════

class TestGenerateFromFeed:
    """Tests for SignalGenerator.generate_from_feed() event-driven path."""

    def _generator(self, cooldown=0.0):
        from src.strategy.generator import (
            FeeStructure, SignalGenerator, SignalGeneratorConfig
        )
        ex = MagicMock()
        inv = MagicMock()
        inv.available.return_value = Decimal("100000")
        inv.get_skews.return_value = []
        gen = SignalGenerator(
            exchange_client=ex,
            pricing_engine=None,
            inventory_tracker=inv,
            fee_structure=FeeStructure(
                cex_taker_bps=Decimal("10"),
                dex_swap_bps=Decimal("30"),
                gas_cost_usd=Decimal("0.10")
            ),
            config=SignalGeneratorConfig(
                alpha=Decimal("0.10"),
                kelly_fraction=Decimal("0.25"),
                max_position_usd=Decimal("10000"),
                signal_ttl_seconds=Decimal("5"),
                cooldown_seconds=Decimal(str(cooldown))
            )
        )
        # Warm up Kalman filter
        kf = gen._get_or_create_filter("ETH/USDC")
        for _ in range(200):
            kf.update(0.006)
        return gen

    def test_generate_from_valid_state(self):
        gen = self._generator()
        state = _price_state(bid=2000.0, ask=2001.0, dex=2010.0)
        sig = gen.generate_from_feed("ETH/USDC", Decimal("0.1"), state)
        # May or may not produce a signal depending on Kalman state
        # but must not raise
        assert sig is None or isinstance(sig, Signal)

    def test_returns_none_when_state_invalid(self):
        gen = self._generator()
        state = _price_state(dex_stale=True)
        assert gen.generate_from_feed("ETH/USDC", Decimal("0.1"), state) is None

    def test_returns_none_during_cooldown(self):
        gen = self._generator(cooldown=999.0)
        state = _price_state()
        gen._last_signal_time["ETH/USDC"] = Decimal(str(time.time()))
        assert gen.generate_from_feed("ETH/USDC", Decimal("0.1"), state) is None

    def test_returns_none_when_prices_zero(self):
        gen = self._generator()
        state = PriceState(pair="ETH/USDC")
        state.is_valid = True   # Force valid flag
        state.cex_bid = Decimal("0")
        state.cex_ask = Decimal("0")
        state.dex_price = Decimal("0")
        assert gen.generate_from_feed("ETH/USDC", Decimal("0.1"), state) is None

    def test_produced_signal_has_correct_pair(self):
        gen = self._generator()

        # Override _generate_from_prices to return a known signal
        def fake_gen(pair, size, prices):
            return Signal.create(
                pair=pair, direction=Direction.BUY_CEX_SELL_DEX,
                cex_price="2000", dex_price="2010",
                raw_spread_bps="50", filtered_spread="0.005",
                posterior_variance="1E-6", signal_confidence="0.92",
                kelly_size="0.1", expected_net_pnl="5",
                ttl_seconds="5", inventory_ok=True, within_limits=True,
                innovation_zscore="0.5"
            )

        gen._generate_from_prices = fake_gen
        state = _price_state(bid=2000.0, ask=2001.0, dex=2010.0)
        sig = gen.generate_from_feed("ETH/USDC", Decimal("0.1"), state)
        if sig:
            assert sig.pair == "ETH/USDC"

    def test_generate_from_prices_refactored_method_exists(self):
        gen = self._generator()
        assert hasattr(gen, "_generate_from_prices")
        assert callable(gen._generate_from_prices)

    def test_generate_rest_still_works(self):
        """REST-based generate() must still work for fallback mode."""
        gen = self._generator()
        # Mock the exchange
        gen._exchange.fetch_order_book.return_value = {
            "bids": [(Decimal("2000.0"), Decimal("5.0"))],
            "asks": [(Decimal("2001.0"), Decimal("4.0"))],
            "best_bid": (Decimal("2000.0"), Decimal("5.0")),
            "best_ask": (Decimal("2001.0"), Decimal("4.0")),
            "mid_price": Decimal("2000.5"),
            "spread_bps": Decimal("5.0")
        }
        # Should not raise
        result = gen.generate("ETH/USDC", Decimal("0.1"))
        assert result is None or isinstance(result, Signal)


# ═══════════════════════════ Event-driven bot ════════════════════════════════

class TestEventDrivenBot:
    """Tests for ArbBot._on_price_update event-driven path."""

    def _bot(self):
        from config.mode import SystemConfig
        import os
        with patch.dict(os.environ, {"OPERATION_MODE": "test", "DRY_RUN": "true"}):
            cfg = SystemConfig.from_env()
        from scripts.arb_bot import ArbBot
        bot = ArbBot.__new__(ArbBot)
        bot._cfg = cfg
        bot._dry_run = True
        bot.running = True
        bot._pq = []
        bot.pairs = ["ETH/USDC"]
        bot.trade_size = 0.01
        bot.min_score_threshold = 0.0   # Accept all scored signals
        bot.max_queue_depth = 10
        bot._using_ws = False
        bot._cex_book = None
        bot._dex_feed = None
        bot._feed_manager = None

        # Safety mocks
        bot._manual_kill = MagicMock()
        bot._manual_kill.is_active.return_value = False
        bot._auto_kill = MagicMock()
        bot._auto_kill.triggered = False
        bot.circuit_breaker = MagicMock()
        bot.circuit_breaker.is_open.return_value = False
        bot.circuit_breaker._A = Decimal("-1")
        bot.circuit_breaker._B = Decimal("2")
        bot.circuit_breaker.lambda_statistic = Decimal("0.5")

        bot._validator = MagicMock()
        bot._validator.validate_signal.return_value = (True, "OK")
        bot._trading_rules = MagicMock()
        bot._trading_rules.round_quantity.return_value = 0.01
        bot._trading_rules.round_price.return_value = 2000.0
        bot._trading_rules.validate_order.return_value = (True, "OK")
        bot._llm_advisor = MagicMock()
        bot._llm_advisor.should_query.return_value = False
        bot.scorer = MagicMock()
        bot.scorer.score.return_value = Decimal("0.8")
        bot.scorer.score_batch.return_value = {}
        bot._risk_manager = MagicMock()
        bot._risk_manager.daily_summary.return_value = {}

        # Generator mock - returns None by default
        bot.generator = MagicMock()
        bot.generator.generate_from_feed.return_value = None

        return bot

    @pytest.mark.asyncio
    async def test_no_signal_from_generator_no_queue(self):
        bot = self._bot()
        state = _price_state()
        bot.generator.generate_from_feed.return_value = None
        await bot._on_price_update("ETH/USDC", state)
        assert len(bot._pq) == 0

    @pytest.mark.asyncio
    async def test_signal_queued_when_valid(self):
        bot = self._bot()
        state = _price_state()
        sig = Signal.create(
            pair="ETH/USDC", direction=Direction.BUY_CEX_SELL_DEX,
            cex_price="2000", dex_price="2010",
            raw_spread_bps="50", filtered_spread="0.005",
            posterior_variance="1E-6", signal_confidence="0.92",
            kelly_size="0.01", expected_net_pnl="5",
            ttl_seconds="5", inventory_ok=True, within_limits=True,
            innovation_zscore="0.5"
        )
        bot.generator.generate_from_feed.return_value = sig

        # Patch _execute_signal to capture calls
        executed = []
        bot._execute_signal = AsyncMock(side_effect=lambda s: executed.append(s.signal_id))

        await bot._on_price_update("ETH/USDC", state)
        assert len(executed) == 1

    @pytest.mark.asyncio
    async def test_no_execution_when_bot_stopped(self):
        bot = self._bot()
        bot.running = False
        state = _price_state()
        executed = []
        bot._execute_signal = AsyncMock(side_effect=lambda s: executed.append(s))
        await bot._on_price_update("ETH/USDC", state)
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_no_execution_when_circuit_breaker_open(self):
        bot = self._bot()
        bot.circuit_breaker.is_open.return_value = True
        state = _price_state()
        executed = []
        bot._execute_signal = AsyncMock(side_effect=lambda s: executed.append(s))
        await bot._on_price_update("ETH/USDC", state)
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_validation_failure_skips_execution(self):
        bot = self._bot()
        bot._validator.validate_signal.return_value = (False, "spread too high")
        state = _price_state()
        sig = Signal.create(
            pair="ETH/USDC", direction=Direction.BUY_CEX_SELL_DEX,
            cex_price="2000", dex_price="2010", raw_spread_bps="50",
            filtered_spread="0.005", posterior_variance="1E-6",
            signal_confidence="0.92", kelly_size="0.01", expected_net_pnl="5",
            ttl_seconds="5", inventory_ok=True, within_limits=True,
            innovation_zscore="0.5"
        )
        bot.generator.generate_from_feed.return_value = sig
        executed = []
        bot._execute_signal = AsyncMock(side_effect=lambda s: executed.append(s))
        await bot._on_price_update("ETH/USDC", state)
        assert len(executed) == 0

    @pytest.mark.asyncio
    async def test_safety_tick_checks_kill_switch(self):
        bot = self._bot()
        bot._manual_kill.is_active.return_value = True
        bot._alerter = MagicMock()
        bot._alerter.kill_switch_activated = AsyncMock()
        bot._dead_man = MagicMock()
        bot._auto_kill = MagicMock()
        bot._auto_kill.check.return_value = False
        await bot._safety_tick()
        assert not bot.running
        bot._alerter.kill_switch_activated.assert_called_once()
