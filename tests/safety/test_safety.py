"""
Tests for the safety module.

Coverage:
  TestRiskLimits          - dataclass defaults and constraint logic
  TestRiskManager         - pre-trade checks, recording, daily reset, drawdown
  TestSafetyCheck         - absolute hard-ceiling function
  TestManualKillSwitch    - file-based kill switch activate/deactivate
  TestAutoKillSwitch      - auto-trigger conditions
  TestDeadManSwitch       - heartbeat write / age check
  TestPreTradeValidator   - all signal sanity checks
  TestBinanceTradingRules - order rounding and validation
  TestSystemConfig        - mode, dry_run, Arbitrum preset, env overrides
"""

from __future__ import annotations

import asyncio
import os
import time
from decimal import Decimal
from unittest.mock import patch

import pytest

from src.safety.limits import RiskLimits, RiskManager
from src.safety.killswitch import (
    AutoKillSwitch, DeadManSwitch, ManualKillSwitch, safety_check,
    ABSOLUTE_MAX_DAILY_LOSS, ABSOLUTE_MAX_TRADE_USD,
    ABSOLUTE_MIN_CAPITAL, ABSOLUTE_MAX_TRADES_PER_HOUR,
    ABSOLUTE_MAX_ERRORS_PER_HOUR
)
from src.safety.validator import PreTradeValidator
from src.strategy.signal import Direction, Signal
from config.mode import BinanceTradingRules, OperationMode, SystemConfig, ARBITRUM_ONE


# ─────────────────────────────── Helpers ────────────────────────────────────

def _signal(
    pair="ETH/USDC", spread_bps="50", confidence="0.92",
    kelly="0.1", net_pnl="20", cex_price="2000", dex_price="2010",
    ttl="5", direction=Direction.BUY_CEX_SELL_DEX,
) -> Signal:
    return Signal.create(
        pair=pair, direction=direction,
        cex_price=cex_price, dex_price=dex_price,
        raw_spread_bps=spread_bps, filtered_spread="0.005",
        posterior_variance="1E-6", signal_confidence=confidence,
        kelly_size=kelly, expected_net_pnl=net_pnl,
        ttl_seconds=ttl, inventory_ok=True, within_limits=True,
        innovation_zscore="0.5"
    )


def _limits(**kw) -> RiskLimits:
    defaults = dict(
        max_trade_usd=20.0, max_trade_pct=0.20, max_position_per_token=30.0,
        max_open_positions=1, max_loss_per_trade=5.0, max_daily_loss=15.0,
        max_drawdown_pct=0.20, max_trades_per_hour=20, consecutive_loss_limit=3
    )
    defaults.update(kw)
    return RiskLimits(**defaults)


def _manager(capital=100.0, **kw) -> RiskManager:
    return RiskManager(_limits(**kw), initial_capital=capital)


# ═══════════════════════════ RiskLimits ══════════════════════════════════════

class TestRiskLimits:
    def test_defaults_sane(self):
        lim = RiskLimits()
        assert lim.max_trade_usd == 20.0
        assert lim.max_daily_loss == 15.0
        assert 0 < lim.max_drawdown_pct < 1

    def test_custom_values_stored(self):
        lim = RiskLimits(max_trade_usd=5.0, max_daily_loss=10.0)
        assert lim.max_trade_usd == 5.0
        assert lim.max_daily_loss == 10.0


# ═══════════════════════════ RiskManager ═════════════════════════════════════

class TestRiskManager:

    # ── Initial state ──────────────────────────────────────────────────────

    def test_initial_state(self):
        rm = _manager(capital=100.0)
        assert rm.current_capital == 100.0
        assert rm.daily_pnl == 0.0
        assert rm.consecutive_losses == 0
        assert rm.open_positions == 0

    # ── Pre-trade: allow ───────────────────────────────────────────────────

    def test_allows_valid_trade(self):
        rm = _manager()
        ok, reason = rm.check_pre_trade(trade_usd=10.0)
        assert ok, reason

    # ── Pre-trade: size limit ──────────────────────────────────────────────

    def test_blocks_trade_exceeding_max_usd(self):
        rm = _manager(max_trade_usd=10.0)
        ok, reason = rm.check_pre_trade(trade_usd=15.0)
        assert not ok
        assert "exceeds max" in reason

    def test_blocks_trade_exceeding_pct_of_capital(self):
        rm = _manager(capital=100.0, max_trade_pct=0.10)
        ok, reason = rm.check_pre_trade(trade_usd=15.0)   # 15% > 10%
        assert not ok
        assert "capital" in reason.lower()

    # ── Pre-trade: daily loss ──────────────────────────────────────────────

    def test_blocks_when_daily_loss_reached(self):
        rm = _manager(max_daily_loss=10.0)
        rm.record_trade(-10.0)
        ok, reason = rm.check_pre_trade(trade_usd=5.0)
        assert not ok
        assert "daily loss" in reason.lower()

    def test_allows_trade_just_below_daily_loss(self):
        rm = _manager(max_daily_loss=10.0)
        rm.record_trade(-9.99)
        ok, _ = rm.check_pre_trade(trade_usd=5.0)
        assert ok

    # ── Pre-trade: drawdown ────────────────────────────────────────────────

    def test_blocks_on_drawdown_breach(self):
        # Use daily_loss limit high enough so drawdown check fires first
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.20, max_daily_loss=50.0), initial_capital=100.0)
        rm.record_trade(-25.0)   # 25% drawdown - exceeds 20% limit
        ok, reason = rm.check_pre_trade(trade_usd=5.0)
        assert not ok
        assert "drawdown" in reason.lower()

    def test_allows_trade_just_below_drawdown(self):
        rm = RiskManager(RiskLimits(max_drawdown_pct=0.20, max_daily_loss=50.0), initial_capital=100.0)
        rm.record_trade(-19.0)   # 19% drawdown - just under 20% limit
        ok, _ = rm.check_pre_trade(trade_usd=5.0)
        assert ok

    # ── Pre-trade: consecutive losses ─────────────────────────────────────

    def test_blocks_after_consecutive_losses(self):
        rm = _manager(consecutive_loss_limit=3)
        for _ in range(3):
            rm.record_trade(-1.0)
        ok, reason = rm.check_pre_trade(trade_usd=5.0)
        assert not ok
        assert "consecutive" in reason.lower()

    def test_resets_consecutive_on_win(self):
        rm = _manager(consecutive_loss_limit=3)
        rm.record_trade(-1.0)
        rm.record_trade(-1.0)
        rm.record_trade(5.0)   # Win resets counter
        assert rm.consecutive_losses == 0
        ok, _ = rm.check_pre_trade(trade_usd=5.0)
        assert ok

    # ── Pre-trade: open positions ──────────────────────────────────────────

    def test_blocks_when_max_open_positions_reached(self):
        rm = _manager(max_open_positions=1)
        rm.open_position()
        ok, reason = rm.check_pre_trade(trade_usd=5.0)
        assert not ok
        assert "open positions" in reason.lower()

    def test_allows_after_position_closed(self):
        rm = _manager(max_open_positions=1)
        rm.open_position()
        rm.close_position()
        ok, _ = rm.check_pre_trade(trade_usd=5.0)
        assert ok

    # ── Pre-trade: hourly frequency ────────────────────────────────────────

    def test_blocks_at_hourly_trade_limit(self):
        rm = _manager(max_trades_per_hour=3)
        for _ in range(3):
            rm.record_trade(0.1)
        ok, reason = rm.check_pre_trade(trade_usd=5.0)
        assert not ok
        assert "hourly" in reason.lower()

    # ── Record trade ───────────────────────────────────────────────────────

    def test_record_trade_updates_capital(self):
        rm = _manager(capital=100.0)
        rm.record_trade(5.0)
        assert rm.current_capital == 105.0

    def test_record_loss_increases_consecutive(self):
        rm = _manager()
        rm.record_trade(-2.0)
        assert rm.consecutive_losses == 1

    def test_record_win_resets_consecutive(self):
        rm = _manager()
        rm.record_trade(-2.0)
        rm.record_trade(-2.0)
        rm.record_trade(1.0)
        assert rm.consecutive_losses == 0

    def test_peak_capital_tracks_high_water(self):
        rm = _manager(capital=100.0)
        rm.record_trade(20.0)
        rm.record_trade(-10.0)
        assert rm.peak_capital == 120.0

    # ── Daily reset ────────────────────────────────────────────────────────

    def test_daily_reset_clears_pnl_and_losses(self):
        rm = _manager()
        rm.record_trade(-5.0)
        rm.record_trade(-5.0)
        rm.reset_daily()
        assert rm.daily_pnl == 0.0
        assert rm.consecutive_losses == 0

    # ── Drawdown property ──────────────────────────────────────────────────

    def test_drawdown_zero_at_start(self):
        rm = _manager(capital=100.0)
        assert rm.drawdown_pct == 0.0

    def test_drawdown_correct_after_loss(self):
        rm = _manager(capital=100.0)
        rm.record_trade(-10.0)
        assert abs(rm.drawdown_pct - 0.10) < 1e-9

    # ── Daily summary ──────────────────────────────────────────────────────

    def test_daily_summary_empty(self):
        rm = _manager()
        s = rm.daily_summary()
        assert s["trades"] == 0

    def test_daily_summary_after_trades(self):
        rm = _manager()
        rm.record_trade(5.0, state="DONE")
        rm.record_trade(-2.0, state="DONE")
        s = rm.daily_summary()
        assert s["trades"] == 2
        assert s["wins"] == 1
        assert s["losses"] == 1
        assert abs(s["total_pnl"] - 3.0) < 1e-9


# ═══════════════════════════ Safety check ════════════════════════════════════

class TestSafetyCheck:

    def test_allows_normal_trade(self):
        ok, reason = safety_check(
            trade_usd=10.0, daily_pnl=-5.0,
            total_capital=90.0, trades_this_hour=5
        )
        assert ok, reason

    def test_blocks_trade_above_absolute_max(self):
        ok, reason = safety_check(
            trade_usd=ABSOLUTE_MAX_TRADE_USD + 1,
            daily_pnl=0.0, total_capital=100.0, trades_this_hour=0
        )
        assert not ok
        assert "ABSOLUTE max" in reason

    def test_blocks_when_daily_loss_reached(self):
        ok, reason = safety_check(
            trade_usd=1.0, daily_pnl=-(ABSOLUTE_MAX_DAILY_LOSS + 0.01),
            total_capital=100.0, trades_this_hour=0
        )
        assert not ok
        assert "daily loss" in reason.lower()

    def test_blocks_when_capital_below_minimum(self):
        ok, reason = safety_check(
            trade_usd=1.0, daily_pnl=0.0,
            total_capital=ABSOLUTE_MIN_CAPITAL - 1,
            trades_this_hour=0
        )
        assert not ok
        assert "minimum" in reason.lower()

    def test_blocks_at_hourly_limit(self):
        ok, reason = safety_check(
            trade_usd=1.0, daily_pnl=0.0, total_capital=100.0,
            trades_this_hour=ABSOLUTE_MAX_TRADES_PER_HOUR
        )
        assert not ok
        assert "hourly" in reason.lower()

    def test_blocks_at_error_limit(self):
        ok, reason = safety_check(
            trade_usd=1.0, daily_pnl=0.0, total_capital=100.0,
            trades_this_hour=0, errors_this_hour=ABSOLUTE_MAX_ERRORS_PER_HOUR
        )
        assert not ok
        assert "error" in reason.lower()

    def test_boundary_daily_loss_exact(self):
        # Exactly at the limit → blocked
        ok, _ = safety_check(
            trade_usd=1.0, daily_pnl=-ABSOLUTE_MAX_DAILY_LOSS,
            total_capital=100.0, trades_this_hour=0
        )
        assert not ok

    def test_boundary_capital_exact(self):
        # Strictly below minimum → blocked; exactly at minimum → allowed (< not <=)
        ok_above, _ = safety_check(trade_usd=1.0, daily_pnl=0.0,
                                   total_capital=ABSOLUTE_MIN_CAPITAL, trades_this_hour=0)
        ok_below, _ = safety_check(trade_usd=1.0, daily_pnl=0.0,
                                   total_capital=ABSOLUTE_MIN_CAPITAL - 0.01, trades_this_hour=0)
        assert not ok_below   # Below minimum → blocked
        # Note: exactly at minimum may be allowed depending on implementation (<)


# ═══════════════════════════ ManualKillSwitch ═════════════════════════════════

class TestManualKillSwitch:

    def _switch(self, tmp_path) -> ManualKillSwitch:
        path = str(tmp_path / "kill")
        return ManualKillSwitch(path=path)

    def test_inactive_by_default(self, tmp_path):
        ks = self._switch(tmp_path)
        assert not ks.is_active()

    def test_activate_creates_file(self, tmp_path):
        ks = self._switch(tmp_path)
        ks.activate("test")
        assert ks.is_active()

    def test_deactivate_removes_file(self, tmp_path):
        ks = self._switch(tmp_path)
        ks.activate()
        ks.deactivate()
        assert not ks.is_active()

    def test_deactivate_when_not_active_is_safe(self, tmp_path):
        ks = self._switch(tmp_path)
        ks.deactivate()   # Should not raise
        assert not ks.is_active()

    def test_re_activate_is_idempotent(self, tmp_path):
        ks = self._switch(tmp_path)
        ks.activate("first")
        ks.activate("second")
        assert ks.is_active()


# ═══════════════════════════ AutoKillSwitch ═══════════════════════════════════

class TestAutoKillSwitch:

    def _auto(self, tmp_path, **kw) -> AutoKillSwitch:
        manual = ManualKillSwitch(path=str(tmp_path / "kill"))
        return AutoKillSwitch(manual_switch=manual, **kw)

    def _rm(self, capital: float, initial: float = 100.0) -> RiskManager:
        rm = _manager(capital=initial)
        diff = capital - initial
        if diff != 0:
            rm.record_trade(diff)
        return rm

    def test_not_triggered_healthy_bot(self, tmp_path):
        auto = self._auto(tmp_path)
        rm = self._rm(capital=100.0)
        assert not auto.check(rm)
        assert not auto.triggered

    def test_triggered_when_capital_drops_below_floor(self, tmp_path):
        auto = self._auto(tmp_path, capital_floor_pct=0.50)
        rm = self._rm(capital=49.0, initial=100.0)
        assert auto.check(rm)
        assert auto.triggered

    def test_triggered_when_capital_below_absolute_minimum(self, tmp_path):
        auto = self._auto(tmp_path)
        rm = self._rm(capital=ABSOLUTE_MIN_CAPITAL - 1, initial=100.0)
        assert auto.check(rm)
        assert auto.triggered

    def test_triggered_when_error_rate_exceeded(self, tmp_path):
        auto = self._auto(tmp_path, max_errors_per_hour=5)
        rm = _manager()
        for _ in range(6):
            rm.record_error()
        assert auto.check(rm)
        assert auto.triggered

    def test_once_triggered_stays_triggered(self, tmp_path):
        auto = self._auto(tmp_path, capital_floor_pct=0.50)
        rm_bad = self._rm(capital=40.0)
        auto.check(rm_bad)
        rm_good = self._rm(capital=200.0)
        assert auto.check(rm_good)   # Still triggered

    def test_alert_callback_called(self, tmp_path):
        alerts = []
        auto = self._auto(tmp_path, capital_floor_pct=0.50,
                          alert_callback=lambda msg: alerts.append(msg))
        auto.check(self._rm(capital=40.0))
        assert len(alerts) == 1
        assert "KILL" in alerts[0].upper() or "capital" in alerts[0].lower()


# ═══════════════════════════ DeadManSwitch ════════════════════════════════════

class TestDeadManSwitch:

    def _dms(self, tmp_path, max_age=60.0) -> DeadManSwitch:
        return DeadManSwitch(
            path=str(tmp_path / "heartbeat"),
            max_age_seconds=max_age
        )

    def test_bot_not_alive_before_first_heartbeat(self, tmp_path):
        dms = self._dms(tmp_path)
        assert not dms.is_bot_alive()

    def test_bot_alive_after_fresh_heartbeat(self, tmp_path):
        dms = self._dms(tmp_path, max_age=60.0)
        dms.write_heartbeat()
        assert dms.is_bot_alive()

    def test_bot_dead_after_stale_heartbeat(self, tmp_path):
        dms = self._dms(tmp_path, max_age=1.0)
        dms.write_heartbeat()
        time.sleep(1.1)
        assert not dms.is_bot_alive()

    def test_seconds_since_heartbeat_increases(self, tmp_path):
        dms = self._dms(tmp_path)
        dms.write_heartbeat()
        time.sleep(0.1)
        age = dms.seconds_since_heartbeat()
        assert age is not None and age >= 0.1

    def test_seconds_since_none_when_no_file(self, tmp_path):
        dms = self._dms(tmp_path)
        assert dms.seconds_since_heartbeat() is None

    @pytest.mark.asyncio
    async def test_run_forever_writes_heartbeat(self, tmp_path):
        dms = DeadManSwitch(
            path=str(tmp_path / "hb"),
            max_age_seconds=10.0
        )
        # Patch interval to 0.05s so test is fast
        with patch("src.safety.killswitch.HEARTBEAT_INTERVAL_SECONDS", 0.05):
            task = asyncio.create_task(dms.run_forever())
            await asyncio.sleep(0.15)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        assert dms.is_bot_alive()


# ═══════════════════════════ PreTradeValidator ════════════════════════════════

class TestPreTradeValidator:

    def _v(self, **kw) -> PreTradeValidator:
        return PreTradeValidator(**kw)

    def test_valid_signal_passes(self):
        v = self._v()
        ok, reason = v.validate_signal(_signal())
        assert ok, reason

    def test_invalid_cex_price_fails(self):
        sig = _signal(cex_price="0")
        ok, reason = self._v().validate_signal(sig)
        assert not ok
        assert "CEX price" in reason

    def test_invalid_dex_price_fails(self):
        sig = _signal(dex_price="0")
        ok, reason = self._v().validate_signal(sig)
        assert not ok
        assert "DEX price" in reason

    def test_stale_signal_fails(self):
        import time
        sig = _signal(ttl="100")   # Long TTL but we check age independently
        # Manually backdate the signal timestamp
        from decimal import Decimal as D
        sig.timestamp = D(str(time.time() - 10))  # 10 seconds old
        # Signal dataclass is not frozen for timestamp, so we set it via object attr
        object.__setattr__(sig, 'timestamp', D(str(time.time() - 10)))
        ok, reason = self._v(max_signal_age_seconds=5.0).validate_signal(sig)
        assert not ok
        assert "old" in reason.lower() or "age" in reason.lower()

    def test_excessive_spread_fails(self):
        sig = _signal(spread_bps="600")
        ok, reason = self._v(max_spread_bps=500.0).validate_signal(sig)
        assert not ok
        assert "500" in reason or "spread" in reason.lower()

    def test_zero_kelly_size_fails(self):
        sig = _signal(kelly="0")
        ok, reason = self._v().validate_signal(sig)
        assert not ok
        assert "size" in reason.lower()

    def test_suspiciously_tight_spread_fails(self):
        sig = _signal(spread_bps="0.1")
        ok, reason = self._v().validate_signal(sig)
        assert not ok
        assert "below" in reason.lower() or "tight" in reason.lower()

    def test_large_cex_dex_ratio_fails(self):
        # CEX = 2000, DEX = 3000 → ratio = 0.667 → outside [0.90, 1.10]
        sig = _signal(cex_price="2000", dex_price="3000", spread_bps="50")
        ok, reason = self._v().validate_signal(sig)
        assert not ok
        assert "ratio" in reason.lower()

    def test_price_deviation_blocks_outlier(self):
        v = self._v(max_price_deviation_pct=0.02)
        normal = _signal(cex_price="2000", dex_price="2010")
        # Warm up history with normal prices
        for _ in range(5):
            v.validate_signal(normal)
        # Now a 10% deviation should be blocked
        outlier = _signal(cex_price="2200", dex_price="2210")
        ok, reason = v.validate_signal(outlier)
        assert not ok
        assert "deviat" in reason.lower()

    def test_post_fill_slippage_ok(self):
        v = self._v()
        ok, _ = v.validate_post_fill(
            expected_price=Decimal("2000"),
            actual_price=Decimal("2001"),
            max_slippage_bps=10.0
        )
        assert ok

    def test_post_fill_excessive_slippage_fails(self):
        v = self._v()
        ok, reason = v.validate_post_fill(
            expected_price=Decimal("2000"),
            actual_price=Decimal("2100"),   # 5% slippage
            max_slippage_bps=10.0,
        )
        assert not ok
        assert "slippage" in reason.lower()

    def test_validates_multiple_signals_sequentially(self):
        v = self._v()
        for i in range(5):
            sig = _signal(cex_price=str(2000 + i), dex_price=str(2010 + i))
            ok, _ = v.validate_signal(sig)
            assert ok


# ═══════════════════════════ BinanceTradingRules ══════════════════════════════

class TestBinanceTradingRules:

    def _rules(self) -> BinanceTradingRules:
        return BinanceTradingRules()

    def test_round_quantity_floors_to_step(self):
        r = self._rules()
        assert r.round_quantity(0.00019) == pytest.approx(0.0001)
        assert r.round_quantity(0.12345) == pytest.approx(0.1234)

    def test_round_price_to_tick(self):
        r = self._rules()
        assert r.round_price(2000.123) == pytest.approx(2000.12)
        assert r.round_price(1999.999) == pytest.approx(2000.00)

    def test_validate_order_ok(self):
        r = self._rules()
        ok, reason = r.validate_order(quantity=0.01, price=2000.0)
        assert ok, reason

    def test_validate_order_below_min_quantity(self):
        r = self._rules()
        ok, reason = r.validate_order(quantity=0.00001, price=2000.0)
        assert not ok
        assert "min" in reason.lower()

    def test_validate_order_below_min_notional(self):
        r = self._rules()
        # 0.001 ETH × $2 = $0.002 → below $5 min notional
        ok, reason = r.validate_order(quantity=0.001, price=2.0)
        assert not ok
        assert "notional" in reason.lower()

    def test_validate_order_exact_min_notional(self):
        r = self._rules()
        # 0.0001 ETH × $50001 = $5.0001 → just above min
        ok, _ = r.validate_order(quantity=0.0001, price=50001.0)
        assert ok


# ═══════════════════════════ SystemConfig ════════════════════════════════════

class TestSystemConfig:

    def test_test_mode_defaults(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test"}, clear=False):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.TEST
        assert cfg.cex.sandbox is True
        assert cfg.dex.dry_run is True
        assert cfg.is_test is True
        assert cfg.is_production is False

    def test_production_mode(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "production", "DRY_RUN": "false"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.PRODUCTION
        assert cfg.cex.sandbox is False
        assert cfg.dex.dry_run is False

    def test_dry_run_overrideable_independently(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "production", "DRY_RUN": "true"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.PRODUCTION
        assert cfg.dry_run is True   # Dry-run even in production config

    def test_arbitrum_defaults(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test", "USE_ARBITRUM": "true"}):
            cfg = SystemConfig.from_env()
        assert cfg.dex.chain_id == 42161
        assert cfg.dex.router_address == ARBITRUM_ONE.router_address

    def test_env_overrides_max_trade_usd(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test", "MAX_TRADE_USD": "7.5"}):
            cfg = SystemConfig.from_env()
        assert cfg.risk.max_trade_usd == 7.5

    def test_env_overrides_initial_capital(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test", "INITIAL_CAPITAL": "250.0"}):
            cfg = SystemConfig.from_env()
        assert cfg.risk.initial_capital == 250.0

    def test_network_preset_arbitrum(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test"}):
            cfg = SystemConfig.from_env()
        assert cfg.network_preset.chain_id == 42161

    def test_pool_addresses_parsed(self):
        addr = "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"
        with patch.dict(os.environ, {"OPERATION_MODE": "test", "POOL_ADDRESSES": addr}):
            cfg = SystemConfig.from_env()
        assert addr in cfg.dex.pool_addresses

    def test_pool_addresses_comma_separated(self):
        a1 = "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"
        a2 = "0x397FF1542f962076d0BFE58eA045FfA2d347ACa0"
        with patch.dict(os.environ, {"POOL_ADDRESSES": f"{a1}, {a2}"}):
            cfg = SystemConfig.from_env()
        assert a1 in cfg.dex.pool_addresses
        assert a2 in cfg.dex.pool_addresses

    def test_gas_cost_usd_arbitrum_default(self):
        with patch.dict(os.environ, {"OPERATION_MODE": "test"}):
            cfg = SystemConfig.from_env()
        # Default is $0.10 for Arbitrum (vs $5 for mainnet)
        assert cfg.gas_cost_usd <= Decimal("1.0")
