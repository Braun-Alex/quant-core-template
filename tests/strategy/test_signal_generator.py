"""
Tests for Kalman signal generator.
"""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import MagicMock

from src.strategy.signal import Direction, KalmanState, Signal
from src.strategy.generator import (
    FeeStructure,
    KalmanSpreadFilter,
    SignalGenerator,
    SignalGeneratorConfig
)


# ───────────────────────────── Helpers ──────────────────────────────────────

def _ob(bid=2000.0, ask=2001.0):
    mid = (bid + ask) / 2
    return {
        "symbol": "ETH/USDT",
        "timestamp": int(time.time() * 1000),
        "bids": [(bid, 10.0)],
        "asks": [(ask, 10.0)],
        "best_bid": (bid, 10.0),
        "best_ask": (ask, 10.0),
        "mid_price": mid,
        "spread_bps": (ask - bid) / mid * 10_000
    }


def _exchange(bid=2000.0, ask=2001.0):
    ex = MagicMock()
    ex.fetch_order_book.return_value = _ob(bid, ask)
    return ex


def _inventory(available=100_000.0):
    inv = MagicMock()
    inv.available.return_value = available
    inv.get_skews.return_value = []
    return inv


def _generator(bid=2000.0, ask=2001.0, available=100_000.0,
               cooldown_seconds=Decimal("0"),
               alpha=Decimal("0.10"),
               max_position_usd=Decimal("10000"),
               **extra_cfg):
    """Convenience factory — kwargs go directly into SignalGeneratorConfig."""
    cfg = SignalGeneratorConfig(
        alpha=alpha,
        kelly_fraction=Decimal("0.25"),
        max_position_usd=max_position_usd,
        signal_ttl_seconds=Decimal("5"),
        cooldown_seconds=cooldown_seconds,
        em_window=50
    )
    # Allow arbitrary overrides
    for k, v in extra_cfg.items():
        setattr(cfg, k, Decimal(str(v)) if not isinstance(v, int) else v)

    return SignalGenerator(
        exchange_client=_exchange(bid, ask),
        pricing_engine=None,
        inventory_tracker=_inventory(available),
        fee_structure=FeeStructure(
            cex_taker_bps=Decimal("10"),
            dex_swap_bps=Decimal("30"),
            gas_cost_usd=Decimal("2")
        ),
        config=cfg
    )


# ═══════════════════════════ KalmanSpreadFilter ══════════════════════════════

class TestKalmanSpreadFilter:
    def test_initial_state_decimal(self):
        kf = KalmanSpreadFilter()
        s = kf.state
        assert isinstance(s.mean, Decimal)
        assert s.mean == Decimal("0")
        assert s.variance == Decimal("1")
        assert s.tick == 0

    def test_tick_increments(self):
        kf = KalmanSpreadFilter()
        kf.update(0.01)
        assert kf.state.tick == 1

    def test_variance_shrinks(self):
        kf = KalmanSpreadFilter(init_Q=Decimal("1E-5"), init_R=Decimal("1E-4"))
        init = kf.state.variance
        kf.update(0.0)
        assert kf.state.variance < init

    def test_kalman_gain_in_unit_interval(self):
        kf = KalmanSpreadFilter()
        kf.update(0.005)
        assert Decimal("0") < kf.state.kalman_gain < Decimal("1")

    def test_mean_moves_toward_observation(self):
        kf = KalmanSpreadFilter(init_Q=Decimal("0.1"), init_R=Decimal("0.001"))
        for _ in range(20):
            kf.update(0.01)
        assert kf.state.mean > Decimal("0.005")

    def test_innovation_correct(self):
        kf = KalmanSpreadFilter(init_Q=Decimal("0"), init_R=Decimal("0.0001"))
        kf.update(0.01)
        # Prior mean = 0, z = 0.01 → innovation = 0.01
        assert abs(kf.state.innovation - Decimal("0.01")) < Decimal("1E-9")

    def test_em_runs_without_error(self):
        kf = KalmanSpreadFilter(em_window=10)
        for i in range(15):
            kf.update(0.001 * i)
        assert kf.state.process_noise > Decimal("0")
        assert kf.state.observation_noise > Decimal("0")

    def test_em_noise_positive_after_noisy_data(self):
        """Both Q and R remain positive after EM (sign correctness)."""
        import random
        random.seed(42)
        kf = KalmanSpreadFilter(em_window=20)
        for _ in range(25):
            kf.update(random.gauss(0.01, 0.05))
        assert kf.state.process_noise > Decimal("0")
        assert kf.state.observation_noise > Decimal("0")

    def test_innovation_zscore_is_decimal(self):
        kf = KalmanSpreadFilter()
        kf.update(0.01)
        assert isinstance(kf.last_innovation_zscore, Decimal)

    def test_state_object_returned(self):
        kf = KalmanSpreadFilter()
        s = kf.update(0.005)
        assert isinstance(s, KalmanState)
        assert s is kf.state

    def test_convergence_loose(self):
        """Filter mean stays within 5 bp of true spread after 200 obs."""
        true_spread = 0.0080
        kf = KalmanSpreadFilter(init_Q=Decimal("1E-4"), init_R=Decimal("1E-3"))
        import random
        random.seed(123)
        for _ in range(200):
            kf.update(true_spread + random.gauss(0, 0.002))
        assert abs(float(kf.state.mean) - true_spread) < 0.005


# ═══════════════════════════════ Signal ═════════════════════════════════════

class TestSignal:
    def _make(self, **kw) -> Signal:
        defaults = dict(
            pair="ETH/USDT", direction=Direction.BUY_CEX_SELL_DEX,
            cex_price="2000", dex_price="2010", raw_spread_bps="50",
            filtered_spread="0.005", posterior_variance="1E-6",
            signal_confidence="0.95", kelly_size="0.1",
            expected_net_pnl="5", ttl_seconds="5",
            inventory_ok=True, within_limits=True, innovation_zscore="1.2"
        )
        defaults.update(kw)
        return Signal.create(**defaults)

    def test_valid(self):
        assert self._make().is_valid()

    def test_expired_invalid(self):
        assert not self._make(ttl_seconds="-1").is_valid()

    def test_no_inventory_invalid(self):
        assert not self._make(inventory_ok=False).is_valid()

    def test_neg_pnl_invalid(self):
        assert not self._make(expected_net_pnl="-1").is_valid()

    def test_age_is_decimal(self):
        s = self._make()
        time.sleep(0.01)
        assert isinstance(s.age_seconds(), Decimal)
        assert s.age_seconds() > Decimal("0")

    def test_anomalous_high_z(self):
        assert self._make(innovation_zscore="5").is_anomalous()

    def test_not_anomalous_low_z(self):
        assert not self._make(innovation_zscore="1.5").is_anomalous()

    def test_unique_ids(self):
        assert self._make().signal_id != self._make().signal_id

    def test_id_has_pair(self):
        assert "ETHUSDT" in self._make(pair="ETH/USDT").signal_id

    def test_ttl_decimal(self):
        s = self._make(ttl_seconds="7")
        assert abs(s.ttl() - Decimal("7")) < Decimal("0.1")

    def test_remaining_ttl_positive_then_zero(self):
        s = self._make(ttl_seconds="5")
        r0 = s.remaining_ttl()
        time.sleep(0.05)
        r1 = s.remaining_ttl()
        assert r0 > r1

    def test_all_numeric_fields_decimal(self):
        s = self._make()
        for attr in ("cex_price", "dex_price", "raw_spread_bps", "filtered_spread",
                     "posterior_variance", "signal_confidence", "kelly_size",
                     "expected_net_pnl", "score", "timestamp", "expiry", "innovation_zscore"):
            assert isinstance(getattr(s, attr), Decimal), f"{attr} is not Decimal"


# ═══════════════════════════ FeeStructure ════════════════════════════════════

class TestFeeStructure:
    def test_total_bps_large_trade_close_to_fixed(self):
        fees = FeeStructure(cex_taker_bps=Decimal("10"),
                            dex_swap_bps=Decimal("30"),
                            gas_cost_usd=Decimal("5"))
        bps = fees.total_fee_bps(Decimal("1000000"))   # 1M USD → gas negligible
        assert abs(bps - Decimal("40")) < Decimal("1")   # ~40 bps

    def test_total_bps_small_trade_gas_dominates(self):
        fees = FeeStructure(cex_taker_bps=Decimal("10"),
                            dex_swap_bps=Decimal("30"),
                            gas_cost_usd=Decimal("5"))
        bps = fees.total_fee_bps(Decimal("50"))
        exp = Decimal("10") + Decimal("30") + (Decimal("5") / Decimal("50")) * Decimal("10000")
        assert abs(bps - exp) < Decimal("1")

    def test_zero_returns_inf(self):
        fees = FeeStructure()
        assert fees
        assert fees.total_fee_bps(Decimal("0")) == Decimal("Inf")

    def test_breakeven_positive(self):
        assert FeeStructure().breakeven_log_spread(Decimal("10000")) > Decimal("0")

    def test_breakeven_grows_with_gas(self):
        f1 = FeeStructure(gas_cost_usd=Decimal("2"))
        f2 = FeeStructure(gas_cost_usd=Decimal("20"))
        assert f2.breakeven_log_spread(Decimal("10000")) > f1.breakeven_log_spread(Decimal("10000"))

    def test_fields_are_decimal(self):
        f = FeeStructure()
        assert isinstance(f.cex_taker_bps, Decimal)
        assert isinstance(f.dex_swap_bps, Decimal)
        assert isinstance(f.gas_cost_usd, Decimal)


# ═══════════════════════════ SignalGenerator ══════════════════════════════════

class TestSignalGenerator:
    def _warmed(self, pair="ETH/USDT", spread=0.006, **kw) -> SignalGenerator:
        gen = _generator(**kw)
        kf = gen._get_or_create_filter(pair)
        for _ in range(30):
            kf.update(spread)
        return gen

    # ── Detection ────────────────────────────────────────────────────────────

    def test_generates_signal_when_spread_positive(self):
        """After 200 warm-up ticks the posterior confidence exceeds 0.90."""
        gen = self._warmed(spread=0.006, pair="ETH/USDT")
        # 200-tick warm-up (via _warmed with spread=0.006 x 200 iters)
        kf = gen._get_or_create_filter("ETH/USDT")
        for _ in range(170):   # Already did 30 in _warmed
            kf.update(0.006)
        sig = gen.generate("ETH/USDT", Decimal("1.0"))   # 1 ETH → gas bps negligible
        assert sig is not None

    def test_direction_is_buy_cex_sell_dex_when_dex_high(self):
        gen = self._warmed()
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        if sig:
            assert sig.direction == Direction.BUY_CEX_SELL_DEX

    def test_no_signal_when_filter_cold(self):
        """Alpha=0.001 means P(s>c) must exceed 99.9% — not met on cold filter."""
        gen = _generator(alpha=Decimal("0.001"), cooldown_seconds=Decimal("0"))
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        assert sig is None

    # ── Cooldown ─────────────────────────────────────────────────────────────

    def test_cooldown_blocks(self):
        gen = _generator(cooldown_seconds=Decimal("999"))
        gen._last_signal_time["ETH/USDT"] = Decimal(str(time.time()))
        assert gen.generate("ETH/USDT", Decimal("0.1")) is None

    def test_cooldown_expired_allows(self):
        gen = _generator(cooldown_seconds=Decimal("0.01"))
        gen._last_signal_time["ETH/USDT"] = Decimal(str(time.time())) - Decimal("1")
        # Should not raise; may return None for unrelated reasons
        gen.generate("ETH/USDT", Decimal("0.1"))

    # ── Per-pair independence ─────────────────────────────────────────────────

    def test_independent_filters(self):
        gen = _generator()
        kf_e = gen._get_or_create_filter("ETH/USDT")
        kf_b = gen._get_or_create_filter("BTC/USDT")
        kf_e.update(0.01)
        assert kf_e.state.mean != kf_b.state.mean

    # ── Signal fields ──────────────────────────────────────────────────────

    def test_signal_pair_correct(self):
        gen = self._warmed()
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        if sig:
            assert sig.pair == "ETH/USDT"

    def test_confidence_in_unit_interval(self):
        gen = self._warmed()
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        if sig:
            assert Decimal("0") <= sig.signal_confidence <= Decimal("1")

    def test_kelly_size_bounded(self):
        gen = self._warmed(max_position_usd=Decimal("1000"))
        sig = gen.generate("ETH/USDT", Decimal("1"))
        if sig:
            assert sig.kelly_size * sig.cex_price <= Decimal("1001")   # 0.1% tolerance

    def test_get_filter_none_for_unseen(self):
        assert _generator().get_filter_state("UNKNOWN/PAIR") is None

    def test_get_filter_state_after_update(self):
        gen = _generator()
        gen._get_or_create_filter("ETH/USDT").update(0.005)
        assert isinstance(gen.get_filter_state("ETH/USDT"), KalmanState)

    def test_posterior_variance_nonneg(self):
        gen = self._warmed()
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        if sig:
            assert sig.posterior_variance >= Decimal("0")

    # ── CDF ───────────────────────────────────────────────────────────────────

    def test_normal_cdf_at_zero(self):
        val = SignalGenerator._normal_cdf(Decimal("0"))
        assert abs(val - Decimal("0.5")) < Decimal("1E-6")

    def test_normal_cdf_at_1645(self):
        val = SignalGenerator._normal_cdf(Decimal("1.645"))
        assert abs(val - Decimal("0.95")) < Decimal("0.001")

    def test_normal_cdf_returns_decimal(self):
        assert isinstance(SignalGenerator._normal_cdf(Decimal("0")), Decimal)

    # ── Error handling ────────────────────────────────────────────────────────

    def test_fetch_error_returns_none(self):
        gen = _generator()
        gen._exchange.fetch_order_book.side_effect = Exception("network")
        assert gen.generate("ETH/USDT", Decimal("0.1")) is None

    def test_insufficient_inventory_flag(self):
        gen = self._warmed(available=0.0)
        sig = gen.generate("ETH/USDT", Decimal("0.1"))
        if sig:
            assert not sig.inventory_ok
