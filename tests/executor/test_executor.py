"""
Tests for PFA Executor and SPRT Circuit Breaker.
"""

from __future__ import annotations

import asyncio
import time
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.strategy.signal import Direction, Signal
from src.executor.engine import (
    Executor, ExecutionContext, ExecutionRiskFilter,
    ExecutorConfig, ExecutorState, ReplayProtection
)
from src.executor.recovery import (
    SPRTCircuitBreaker, SPRTConfig, LLMAnomalyAdvisor
)


# ─────────────────────────────── Helpers ────────────────────────────────────

def _d(v):
    return Decimal(str(v))


def _signal(
        net_pnl="20", kelly="0.1",
        cex_price="2000", dex_price="2010",
        ttl="5", inventory_ok=True, within_limits=True,
        direction=Direction.BUY_CEX_SELL_DEX
) -> Signal:
    return Signal.create(
        pair="ETH/USDT", direction=direction,
        cex_price=cex_price, dex_price=dex_price,
        raw_spread_bps="50", filtered_spread="0.005",
        posterior_variance="1E-6", signal_confidence="0.92",
        kelly_size=kelly, expected_net_pnl=net_pnl,
        ttl_seconds=ttl, inventory_ok=inventory_ok,
        within_limits=within_limits, innovation_zscore="0.8"
    )


def _cb(tripped=False):
    cb = SPRTCircuitBreaker()
    if tripped:
        cb.trip()
    return cb


def _executor(cb=None, use_dex_first=True, sim=True, **kw):
    cfg_kw = {k: _d(v) if k not in ("use_dex_first", "simulation_mode", "max_recovery_attempts") else v
              for k, v in kw.items()}
    return Executor(
        exchange_client=MagicMock(),
        pricing_engine=None,
        inventory_tracker=MagicMock(),
        circuit_breaker=cb or SPRTCircuitBreaker(),
        config=ExecutorConfig(simulation_mode=sim, use_dex_first=use_dex_first, **cfg_kw)
    )


# ═══════════════════════════ ExecutionRiskFilter ══════════════════════════════

class TestExecutionRiskFilter:
    def test_approved_when_var_below_pnl(self):
        cfg = ExecutorConfig(vol_per_sqrt_second=_d("0.0001"), var_confidence=_d("0.95"))
        rf = ExecutionRiskFilter(cfg)
        sig = _signal(net_pnl="100", kelly="0.1")
        ok, var = rf.approve(sig, _d("1"))
        assert ok
        assert var > _d("0")
        assert isinstance(var, Decimal)

    def test_rejected_extreme_vol(self):
        cfg = ExecutorConfig(vol_per_sqrt_second=_d("1"), var_confidence=_d("0.99"))
        rf = ExecutionRiskFilter(cfg)
        sig = _signal(net_pnl="0.01", kelly="10")
        ok, _ = rf.approve(sig, _d("60"))
        assert not ok

    def test_var_zero_latency(self):
        cfg = ExecutorConfig(vol_per_sqrt_second=_d("0.01"), var_confidence=_d("0.95"))
        _, var = ExecutionRiskFilter(cfg).approve(_signal(net_pnl="10"), _d("0"))
        assert var == _d("0")

    def test_var_increases_with_vol(self):
        sig = _signal(net_pnl="1000", kelly="0.1")
        _, v1 = ExecutionRiskFilter(ExecutorConfig(vol_per_sqrt_second=_d("0.0001"))).approve(sig, _d("1"))
        _, v2 = ExecutionRiskFilter(ExecutorConfig(vol_per_sqrt_second=_d("0.01"))).approve(sig, _d("1"))
        assert v2 > v1

    def test_var_is_decimal(self):
        _, var = ExecutionRiskFilter(ExecutorConfig()).approve(_signal(), _d("1"))
        assert isinstance(var, Decimal)

    def test_normal_quantile_95(self):
        z = ExecutionRiskFilter._normal_quantile(0.95)
        assert abs(z - 1.645) < 0.01


# ═══════════════════════════ PFA Executor ═════════════════════════════════════

class TestExecutorStateMachine:

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_fails(self):
        ex = _executor(cb=_cb(tripped=True))
        ctx = await ex.execute(_signal())
        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "CB_OPEN"

    @pytest.mark.asyncio
    async def test_replay_protection_blocks(self):
        ex = _executor()
        sig = _signal()
        ex.replay_protection.mark_executed(sig)
        ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "REPLAY"

    @pytest.mark.asyncio
    async def test_expired_signal_fails_validating(self):
        ex = _executor()
        ctx = await ex.execute(_signal(ttl="-1"))
        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "INVALID"

    @pytest.mark.asyncio
    async def test_high_var_rejected(self):
        ex = _executor(vol_per_sqrt_second=_d("10"), leg2_timeout=_d("600"))
        sig = _signal(net_pnl="0.001", kelly="100")
        ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "VAR_REJECTED"

    # ── DEX-first ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_dex_first_happy_path(self):
        ctx = await _executor(use_dex_first=True).execute(_signal())
        assert ctx.state == ExecutorState.DONE
        assert isinstance(ctx.actual_net_pnl, Decimal)

    @pytest.mark.asyncio
    async def test_dex_first_state_sequence(self):
        ctx = await _executor(use_dex_first=True).execute(_signal())
        states = [s for s, _ in ctx.state_history]
        assert ExecutorState.LEG1_PENDING in states
        assert ExecutorState.LEG1_FILLED in states
        assert ExecutorState.LEG2_PENDING in states
        assert ExecutorState.DONE in states

    @pytest.mark.asyncio
    async def test_dex_first_leg1_timeout(self):
        ex = _executor(use_dex_first=True, leg2_timeout=_d("0.0001"))

        async def slow(*a, **kw):
            await asyncio.sleep(10)
            return {"success": True, "price": 2010.0, "filled": 0.1}

        with patch.object(ex, "_execute_dex_leg", slow):
            ctx = await ex.execute(_signal())
        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "L1_TIMEOUT"

    @pytest.mark.asyncio
    async def test_dex_first_leg2_timeout_unwinds(self):
        ex = _executor(use_dex_first=True, leg1_timeout=_d("0.0001"))

        async def ok_dex(*a, **kw):
            return {"success": True, "price": 2010.0, "filled": 0.1}

        async def slow_cex(*a, **kw):
            await asyncio.sleep(10)
            return {}

        unwound = []

        async def mock_unwind(c):
            unwound.append(True)

        with (patch.object(ex, "_execute_dex_leg", ok_dex),
              patch.object(ex, "_execute_cex_leg", slow_cex),
              patch.object(ex, "_unwind", mock_unwind)):
            ctx = await ex.execute(_signal())

        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "L2_TIMEOUT"
        assert unwound

    # ── CEX-first ─────────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_cex_first_happy_path(self):
        ctx = await _executor(use_dex_first=False).execute(_signal())
        assert ctx.state == ExecutorState.DONE

    @pytest.mark.asyncio
    async def test_cex_first_leg2_fail_unwinds(self):
        ex = _executor(use_dex_first=False)

        async def ok_cex(*a, **kw):
            return {"success": True, "price": 2000.0, "filled": 0.1}

        async def fail_dex(*a, **kw):
            return {"success": False, "error": "reverted"}

        unwound = []

        async def mock_unwind(c):
            unwound.append(True)

        with (patch.object(ex, "_execute_cex_leg", ok_cex),
              patch.object(ex, "_execute_dex_leg", fail_dex),
              patch.object(ex, "_unwind", mock_unwind)):
            ctx = await ex.execute(_signal())

        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "L2_FAIL"
        assert unwound

    @pytest.mark.asyncio
    async def test_partial_fill_below_min_ratio_aborts(self):
        """
        Phi = 0.5 < min_fill_ratio=0.90, and the adaptive rule aborts because
        reduced_pnl (net_pnl*phi) <= abort_cost. The VaR gate is mocked to
        always approve so this test isolates the partial-fill branch.
        """
        ex = _executor(use_dex_first=False, min_fill_ratio=_d("0.90"))
        sig = _signal(net_pnl="0.001", kelly="0.1")

        async def partial_cex(*a, **kw):
            return {"success": True, "price": 2000.0, "filled": 0.05}  # phi=0.5

        # Bypass VaR gate so we reach the fill-quality check
        from unittest.mock import patch as _patch
        with (
            _patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))),
            patch.object(ex, "_execute_cex_leg", partial_cex),
        ):
            ctx = await ex.execute(sig)

        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "PARTIAL_FILL"

    @pytest.mark.asyncio
    async def test_partial_fill_above_threshold_proceeds(self):
        ex = _executor(use_dex_first=False, min_fill_ratio=_d("0.50"))

        async def partial_cex(*a, **kw):
            return {"success": True, "price": 2000.0, "filled": 0.09}  # Phi = 0.9 > 0.5

        sig = _signal(net_pnl="100", kelly="0.1")
        with patch.object(ex, "_execute_cex_leg", partial_cex):
            ctx = await ex.execute(sig)

        assert ctx.state == ExecutorState.DONE

    # ── PnL & Decimal ─────────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_pnl_is_decimal(self):
        ctx = await _executor().execute(_signal())
        if ctx.state == ExecutorState.DONE:
            assert isinstance(ctx.actual_net_pnl, Decimal)

    @pytest.mark.asyncio
    async def test_fill_quality_is_decimal(self):
        ctx = await _executor().execute(_signal())
        if ctx.state == ExecutorState.DONE:
            assert isinstance(ctx.fill_quality, Decimal)
            assert Decimal("0") <= ctx.fill_quality <= Decimal("1")

    @pytest.mark.asyncio
    async def test_slippage_is_decimal(self):
        ctx = await _executor().execute(_signal())
        if ctx.state == ExecutorState.DONE:
            assert isinstance(ctx.leg1_slippage_bps, Decimal)
            assert isinstance(ctx.leg2_slippage_bps, Decimal)

    @pytest.mark.asyncio
    async def test_timestamps_are_decimal(self):
        ctx = await _executor().execute(_signal())
        assert isinstance(ctx.started_at, Decimal)
        assert isinstance(ctx.finished_at, Decimal)

    # ── State history ──────────────────────────────────────────────────────────

    def test_initial_state_history_empty(self):
        ctx = ExecutionContext(signal=_signal())
        assert ctx.state_history == []

    def test_transition_records(self):
        ctx = ExecutionContext(signal=_signal())
        ctx.transition(ExecutorState.VALIDATING)
        ctx.transition(ExecutorState.LEG1_PENDING)
        assert len(ctx.state_history) == 2
        state, ts = ctx.state_history[0]
        assert state == ExecutorState.VALIDATING
        assert isinstance(ts, Decimal)

    def test_duration_decimal(self):
        ctx = ExecutionContext(signal=_signal())
        assert isinstance(ctx.duration(), Decimal)


# ═══════════════════════════ ReplayProtection ════════════════════════════════

class TestReplayProtection:
    def test_new_not_duplicate(self):
        assert not ReplayProtection().is_duplicate(_signal())

    def test_executed_is_duplicate(self):
        rp = ReplayProtection()
        s = _signal()
        rp.mark_executed(s)
        assert rp.is_duplicate(s)

    def test_different_signal_allowed(self):
        rp = ReplayProtection()
        s1, s2 = _signal(), _signal()
        rp.mark_executed(s1)
        assert not rp.is_duplicate(s2)

    def test_expired_entry_cleared(self):
        rp = ReplayProtection(ttl_seconds=_d("0.01"))
        s = _signal()
        rp.mark_executed(s)
        time.sleep(0.05)
        assert not rp.is_duplicate(s)

    def test_cleanup_removes_old(self):
        rp = ReplayProtection(ttl_seconds=_d("0.01"))
        s = _signal()
        rp.mark_executed(s)
        time.sleep(0.05)
        rp._cleanup()
        assert len(rp._executed) == 0


# ═══════════════════════════ SPRT Circuit Breaker ════════════════════════════

class TestSPRTCircuitBreaker:
    def test_starts_closed(self):
        assert not SPRTCircuitBreaker().is_open()

    def test_many_failures_trip(self):
        cb = SPRTCircuitBreaker(SPRTConfig(
            p0=_d("0.1"), p1=_d("0.9"), alpha=_d("0.05"), beta=_d("0.10"), gamma=_d("1")
        ))
        for _ in range(30):
            cb.record_failure()
        assert cb.is_open()

    def test_successes_dont_trip(self):
        cb = SPRTCircuitBreaker()
        for _ in range(100):
            cb.record_success()
        assert not cb.is_open()

    def test_trip_increments_count(self):
        cb = SPRTCircuitBreaker()
        cb.trip()
        assert cb.trip_count == 1

    def test_reset_closes(self):
        cb = SPRTCircuitBreaker()
        cb.trip()
        cb.reset()
        assert not cb.is_open()

    def test_exponential_backoff(self):
        cfg = SPRTConfig(base_cooldown=_d("10"), max_cooldown=_d("1000"), gamma=_d("1"))
        cb = SPRTCircuitBreaker(cfg)
        cb._trip_count = 0
        c0 = cb._current_cooldown()  # 10*2^0 = 10
        cb._trip_count = 2
        c2 = cb._current_cooldown()  # 10*2^1 = 20
        cb._trip_count = 3
        c3 = cb._current_cooldown()  # 10*2^2 = 40
        assert c0 == pytest.approx(10.0, rel=0.01)
        assert c2 == pytest.approx(20.0, rel=0.01)
        assert c3 == pytest.approx(40.0, rel=0.01)

    def test_cooldown_capped(self):
        cfg = SPRTConfig(base_cooldown=_d("10"), max_cooldown=_d("100"), max_backoff_steps=10, gamma=_d("1"))
        cb = SPRTCircuitBreaker(cfg)
        cb._trip_count = 20
        assert cb._current_cooldown() <= _d("100")

    def test_lambda_increases_on_failure(self):
        cb = SPRTCircuitBreaker()
        l0 = cb.lambda_statistic
        cb.record_failure()
        assert cb.lambda_statistic > l0

    def test_lambda_decreases_on_success(self):
        cb = SPRTCircuitBreaker()
        cb._lambda = _d("0.5")
        l0 = cb.lambda_statistic
        cb.record_success()
        assert cb.lambda_statistic < l0

    def test_lambda_is_decimal(self):
        assert isinstance(SPRTCircuitBreaker().lambda_statistic, Decimal)

    def test_time_until_reset_zero_when_closed(self):
        assert SPRTCircuitBreaker().time_until_reset() == _d("0")

    def test_time_until_reset_positive_when_open(self):
        cb = SPRTCircuitBreaker(SPRTConfig(base_cooldown=_d("100")))
        cb.trip()
        assert cb.time_until_reset() > _d("0")

    def test_time_until_reset_is_decimal(self):
        cb = SPRTCircuitBreaker()
        cb.trip()
        assert isinstance(cb.time_until_reset(), Decimal)

    def test_thresholds(self):
        cb = SPRTCircuitBreaker(SPRTConfig(p0=_d("0.1"), p1=_d("0.4"), alpha=_d("0.05"), beta=_d("0.10")))
        assert cb._B > _d("0")
        assert cb._A < _d("0")

    def test_forgetting_discounts_old_failures(self):
        """
        With gamma < 1 previous observations fade: after N failures then M successes,
        the discounted CB recovers closer to zero than the non-discounted one.
        Use p1=0.4 (moderate) with alpha=0.10, beta=0.20 so the trip threshold
        B is not reached with only 3 failures, and call _update() directly to
        bypass auto-trip logic.
        """
        cfg_disc = SPRTConfig(gamma=_d("0.5"), p0=_d("0.1"), p1=_d("0.4"),
                              alpha=_d("0.10"), beta=_d("0.20"))
        cfg_full = SPRTConfig(gamma=_d("1.0"), p0=_d("0.1"), p1=_d("0.4"),
                              alpha=_d("0.10"), beta=_d("0.20"))
        cb_d = SPRTCircuitBreaker(cfg_disc)
        cb_f = SPRTCircuitBreaker(cfg_full)

        # 3 failures then 10 successes - below trip threshold
        for _ in range(3):
            cb_d._update(success=False)
            cb_f._update(success=False)
        for _ in range(10):
            cb_d._update(success=True)
            cb_f._update(success=True)

        # Discounted CB forgets previous failures faster → its lambda is strictly lower
        assert cb_d.lambda_statistic < cb_f.lambda_statistic


# ═══════════════════════════ LLMAnomalyAdvisor ═══════════════════════════════

class TestLLMAnomalyAdvisor:
    def test_disabled_no_key(self):
        assert not LLMAnomalyAdvisor(api_key="")._enabled

    def test_gate_normal_zscore_no_query(self):
        adv = LLMAnomalyAdvisor(api_key="fake", zscore_threshold=_d("3"))
        assert not adv.should_query(_d("1.5"), _d("0.5"), _d("-1"), _d("2"))

    def test_gate_outside_indeterminate(self):
        adv = LLMAnomalyAdvisor(api_key="fake", zscore_threshold=_d("3"))
        # Anomalous but lambda > B → not indeterminate
        assert not adv.should_query(_d("4"), _d("3"), _d("-1"), _d("2"))

    def test_gate_triggered_anomalous_and_indeterminate(self):
        adv = LLMAnomalyAdvisor(api_key="fake", zscore_threshold=_d("3"))
        assert adv.should_query(_d("4"), _d("0.5"), _d("-1"), _d("2"))

    @pytest.mark.asyncio
    async def test_advise_none_when_disabled(self):
        adv = LLMAnomalyAdvisor(api_key="")
        res = await adv.advise("ETH/USDT", _d("4"), _d("0.005"), _d("50"), {})
        assert res is None
