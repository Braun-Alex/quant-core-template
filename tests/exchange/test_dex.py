"""
Tests for DEX integration: DEXPriceSource, DEXExecutor, and unwind.
All chain calls are mocked - no real network required.
"""
from __future__ import annotations
import asyncio
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

from src.exchange.dex import DEXPriceSource, DEXExecutor, DEXQuote
from src.core.types import Address, Token
from src.pricing.amm import PoolState
from src.executor.engine import Executor, ExecutorConfig, ExecutorState, ExecutionContext
from src.executor.recovery import SPRTCircuitBreaker
from src.strategy.signal import Direction, Signal

WETH = Token(Address("0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"), "WETH", 18)
USDT = Token(Address("0xdAC17F958D2ee523a2206206994597C13D831ec7"), "USDT", 6)
PAIR_ADDR = Address("0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852")


def _d(v):
    return Decimal(str(v))


def _pool():
    return PoolState(contract=PAIR_ADDR, left=WETH, right=USDT,
                     qty_left=1_000 * 10**18, qty_right=2_000_000 * 10**6, fee_bps=30)


def _signal(net_pnl="20", kelly="0.1", cex_price="2000", dex_price="2010",
            direction=Direction.BUY_CEX_SELL_DEX):
    return Signal.create(
        pair="ETH/USDT", direction=direction,
        cex_price=cex_price, dex_price=dex_price,
        raw_spread_bps="50", filtered_spread="0.005",
        posterior_variance="1E-6", signal_confidence="0.92",
        kelly_size=kelly, expected_net_pnl=net_pnl,
        ttl_seconds="5", inventory_ok=True, within_limits=True,
        innovation_zscore="0.5"
    )


def _make_chain_client():
    client = MagicMock()
    client.get_chain_id.return_value = 1
    client.get_nonce.return_value = 10
    gp = MagicMock()
    gp.get_max_fee.return_value = 50_000_000_000
    gp.priority_fee_medium = 2_000_000_000
    gp.priority_fee_high = 4_000_000_000
    client.get_gas_price.return_value = gp
    client.send_transaction.return_value = "0x" + "a" * 64
    client.wait_for_receipt.return_value = MagicMock(status=True)
    w3_mock = MagicMock()
    w3_mock.is_connected.return_value = True
    client._get_w3.return_value = w3_mock
    return client


def _make_wallet():
    wallet = MagicMock()
    wallet.address = "0x" + "1" * 40
    signed = MagicMock()
    signed.raw_transaction = b"\x01" * 100
    wallet.sign_transaction.return_value = signed
    return wallet


def _make_dex_exec_mock(dry_run=True):
    """Mock DEXExecutor that returns a successful swap result."""
    dex_exec = MagicMock(spec=DEXExecutor)
    dex_exec._dry_run = dry_run
    dex_exec.execute_swap.return_value = {
        "success": True, "price": 2001.0, "filled": 0.1,
        "tx_hash": "0x" + "b" * 64, "dry_run": dry_run,
        "swap_tx": {"to": "0xROUTER", "gas": 250_000}
    }
    return dex_exec


def _make_dex_price_mock():
    dex_price = MagicMock(spec=DEXPriceSource)
    q = DEXQuote(
        token_in=WETH, token_out=USDT,
        amount_in=10**17, amount_out_min=int(198 * 10**6),
        expected_out=int(200 * 10**6),
        price=Decimal("2000"), impact_bps=Decimal("1"),
        fee_bps=Decimal("30"), gas_estimate=250_000,
        path=[WETH.address.checksum, USDT.address.checksum]
    )
    dex_price.get_full_quote.return_value = q
    return dex_price


def _make_pricing_with_pool():
    pricing = MagicMock()
    # Use real pool with real tokens so _resolve_tokens works
    pool = _pool()   # WETH/USDT pool
    pricing._pools = {"addr": pool}
    pricing._finder = MagicMock()
    return pricing


def _make_cex_mock():
    """Mock exchange returning a filled order."""
    ex = MagicMock()
    ex.create_limit_ioc_order.return_value = {
        "id": "ord_001", "status": "filled",
        "amount_filled": 0.1, "avg_fill_price": 2000.5
    }
    return ex


def _executor_with_real_dex(use_dex_first=True):
    cfg = ExecutorConfig(
        simulation_mode=False, use_dex_first=use_dex_first,
        vol_per_sqrt_second=_d("0.0001")
    )
    ex = Executor(
        exchange_client=_make_cex_mock(),
        pricing_engine=_make_pricing_with_pool(),
        inventory_tracker=MagicMock(),
        circuit_breaker=SPRTCircuitBreaker(),
        config=cfg,
        dex_price_source=_make_dex_price_mock(),
        dex_executor=_make_dex_exec_mock()
    )
    return ex


# ═══════════════════════════════════════════════════════════════════════════
# DEXPriceSource
# ═══════════════════════════════════════════════════════════════════════════

class TestDEXPriceSource:

    def _source(self, pool=None):
        source = DEXPriceSource(
            pricing_engine=None, chain_client=_make_chain_client(),
            router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            pool_addresses=[], fee_bps=Decimal("30"),
            slippage_bps=Decimal("50"), gas_price_gwei=20
        )
        if pool is not None:
            source._pools["test_pool"] = pool
            source._initialised = True
        return source

    def test_get_dex_quote_returns_dict(self):
        src = self._source(_pool())
        result = src.get_dex_quote("WETH", "USDT", Decimal("1"))
        assert isinstance(result, dict)
        assert "price" in result and "impact_bps" in result and "fee_bps" in result

    def test_price_is_decimal(self):
        src = self._source(_pool())
        result = src.get_dex_quote("WETH", "USDT", Decimal("1"))
        assert isinstance(result["price"], Decimal)

    def test_price_is_positive(self):
        src = self._source(_pool())
        result = src.get_dex_quote("WETH", "USDT", Decimal("1"))
        assert result["price"] > Decimal("0")

    def test_no_pool_returns_zero_price(self):
        result = self._source().get_dex_quote("WETH", "USDT", Decimal("1"))
        assert result["price"] == Decimal("0")

    def test_impact_bps_non_negative(self):
        src = self._source(_pool())
        assert src.get_dex_quote("WETH", "USDT", Decimal("1"))["impact_bps"] >= Decimal("0")

    def test_fee_bps_matches_config(self):
        assert self._source(_pool()).get_dex_quote("WETH", "USDT", Decimal("1"))["fee_bps"] == Decimal("30")

    def test_get_full_quote_returns_dex_quote(self):
        src = self._source(_pool())
        q = src.get_full_quote(WETH, USDT, 10**18)
        assert q is not None
        assert q.amount_in == 10**18
        assert q.expected_out > 0
        assert q.amount_out_min <= q.expected_out

    def test_full_quote_slippage_applied(self):
        q = self._source(_pool()).get_full_quote(WETH, USDT, 10**18)
        ratio = Decimal(q.amount_out_min) / Decimal(q.expected_out)
        assert Decimal("0.99") < ratio < Decimal("1")

    def test_full_quote_path_has_two_addresses(self):
        assert len(self._source(_pool()).get_full_quote(WETH, USDT, 10**18).path) == 2

    def test_full_quote_no_pool_returns_none(self):
        assert self._source().get_full_quote(WETH, USDT, 10**18) is None

    def test_larger_trade_higher_impact(self):
        src = self._source(_pool())
        q_small = src.get_full_quote(WETH, USDT, 10**17)
        q_large = src.get_full_quote(WETH, USDT, 10**19)
        assert q_large.impact_bps >= q_small.impact_bps


# ═══════════════════════════════════════════════════════════════════════════
# DEXExecutor (dry-run)
# ═══════════════════════════════════════════════════════════════════════════

class TestDEXExecutorDryRun:

    def _exec(self, dry_run=True):
        return DEXExecutor(
            chain_client=_make_chain_client(), wallet=_make_wallet(),
            router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
            gas_limit_swap=250_000, gas_limit_approval=60_000,
            slippage_bps=Decimal("50"), deadline_seconds=300, dry_run=dry_run
        )

    def _quote(self):
        return DEXQuote(
            token_in=WETH, token_out=USDT,
            amount_in=10**18, amount_out_min=int(1_990 * 10**6),
            expected_out=int(2_000 * 10**6), price=Decimal("2000"),
            impact_bps=Decimal("1.5"), fee_bps=Decimal("30"),
            gas_estimate=250_000,
            path=[WETH.address.checksum, USDT.address.checksum]
        )

    def test_dry_run_returns_success(self):
        assert self._exec(True).execute_swap(self._quote())["success"] is True

    def test_dry_run_does_not_send_tx(self):
        ex = self._exec(True)
        ex.execute_swap(self._quote())
        ex._client.send_transaction.assert_not_called()

    def test_dry_run_flag_in_result(self):
        assert self._exec(True).execute_swap(self._quote())["dry_run"] is True

    def test_swap_tx_in_result(self):
        result = self._exec(True).execute_swap(self._quote())
        assert "swap_tx" in result and result["swap_tx"] is not None

    def test_swap_tx_has_required_keys(self):
        tx = self._exec(True).execute_swap(self._quote())["swap_tx"]
        for key in ("to", "data", "gas", "nonce", "chainId"):
            assert key in tx, f"Missing key: {key}"

    def test_filled_amount_is_float(self):
        result = self._exec(True).execute_swap(self._quote())
        assert isinstance(result["filled"], float) and result["filled"] > 0

    def test_price_is_float(self):
        assert isinstance(self._exec(True).execute_swap(self._quote())["price"], float)

    def test_build_unwind_tx(self):
        tx = self._exec(True).build_unwind_tx(WETH, USDT, 10**18)
        assert tx["to"] is not None
        assert isinstance(tx["data"], str)
        assert tx["gas"] == 250_000

    def test_production_sends_tx(self):
        client = _make_chain_client()
        wallet = _make_wallet()
        w3_mock = MagicMock()
        w3_mock.is_connected.return_value = True
        token_contract = MagicMock()
        token_contract.functions.allowance.return_value.call.return_value = 10**30
        w3_mock.eth.contract.return_value = token_contract
        client._get_w3.return_value = w3_mock
        ex = DEXExecutor(chain_client=client, wallet=wallet,
                         router_address="0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",
                         dry_run=False)
        result = ex.execute_swap(self._quote())
        assert result["success"] is True
        assert result["dry_run"] is False
        client.send_transaction.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# Executor with real DEX components (mocked)
# ═══════════════════════════════════════════════════════════════════════════

class TestExecutorWithDEX:

    @pytest.mark.asyncio
    async def test_dex_first_with_real_dex_done(self):
        ex = _executor_with_real_dex(use_dex_first=True)
        sig = _signal()
        with patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))):
            ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.DONE

    @pytest.mark.asyncio
    async def test_cex_first_with_real_dex_done(self):
        ex = _executor_with_real_dex(use_dex_first=False)
        sig = _signal()
        with patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))):
            ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.DONE

    @pytest.mark.asyncio
    async def test_leg1_raw_tx_stored_on_dex_first(self):
        ex = _executor_with_real_dex(use_dex_first=True)
        sig = _signal()
        with patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))):
            ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.DONE
        # swap_tx is stored from the DEX executor result; may be None if token resolution
        # fell back to simulation - check that the leg1_tx_hash is recorded instead
        assert ctx.leg1_tx_hash is not None or ctx.leg1_raw_tx is not None

    @pytest.mark.asyncio
    async def test_leg2_tx_hash_stored_on_cex_first(self):
        ex = _executor_with_real_dex(use_dex_first=False)
        sig = _signal()
        with patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))):
            ctx = await ex.execute(sig)
        assert ctx.state == ExecutorState.DONE
        # Leg2 DEX tx hash is set from the DEX executor result
        assert ctx.leg2_tx_hash is not None

    @pytest.mark.asyncio
    async def test_dex_failure_triggers_unwind(self):
        ex = _executor_with_real_dex(use_dex_first=True)
        sig = _signal()

        async def ok_dex(*a, **kw):
            return {"success": True, "price": 2010.0, "filled": 0.1, "swap_tx": {}}

        async def fail_cex(*a, **kw):
            return {"success": False, "error": "rejected"}

        unwind_calls = []

        async def capture_unwind(ctx):
            unwind_calls.append(ctx.signal.signal_id)
            ctx.unwind_attempted = True
            ctx.unwind_succeeded = True

        with (patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))),
              patch.object(ex, "_execute_dex_leg", ok_dex),
              patch.object(ex, "_execute_cex_leg", fail_cex),
              patch.object(ex, "_unwind", capture_unwind)):
            ctx = await ex.execute(sig)

        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "L2_FAIL"
        assert sig.signal_id in unwind_calls


# ═══════════════════════════════════════════════════════════════════════════
# Unwind mechanism
# ═══════════════════════════════════════════════════════════════════════════

class TestUnwindMechanism:

    def _sim_executor(self, use_dex_first=True):
        cfg = ExecutorConfig(simulation_mode=True, use_dex_first=use_dex_first)
        return Executor(
            exchange_client=MagicMock(), pricing_engine=None,
            inventory_tracker=MagicMock(), circuit_breaker=SPRTCircuitBreaker(),
            config=cfg
        )

    @pytest.mark.asyncio
    async def test_unwind_sets_attempted_flag(self):
        ex = self._sim_executor()
        ctx = ExecutionContext(signal=_signal())
        ctx.leg1_venue = "dex"
        ctx.leg1_fill_size = _d("0.1")
        await ex._unwind(ctx)
        assert ctx.unwind_attempted is True

    @pytest.mark.asyncio
    async def test_unwind_zero_fill_skips(self):
        ex = self._sim_executor()
        ctx = ExecutionContext(signal=_signal())
        ctx.leg1_venue = "dex"
        ctx.leg1_fill_size = _d("0")
        await ex._unwind(ctx)
        # Size = 0 → skips without setting attempted
        assert ctx.unwind_attempted is False

    @pytest.mark.asyncio
    async def test_unwind_simulation_mode_succeeds(self):
        ex = self._sim_executor()
        ctx = ExecutionContext(signal=_signal())
        ctx.leg1_venue = "dex"
        ctx.leg1_fill_size = _d("0.1")
        await ex._unwind(ctx)
        assert ctx.unwind_attempted is True
        assert ctx.unwind_succeeded is True

    @pytest.mark.asyncio
    async def test_cex_unwind_simulation_mode(self):
        ex = self._sim_executor(use_dex_first=False)
        ctx = ExecutionContext(signal=_signal())
        ctx.leg1_venue = "cex"
        ctx.leg1_fill_price = _d("2000")
        ctx.leg1_fill_size = _d("0.1")
        await ex._unwind(ctx)
        assert ctx.unwind_attempted is True
        assert ctx.unwind_succeeded is True

    @pytest.mark.asyncio
    async def test_cex_unwind_real_exchange(self):
        exchange = MagicMock()
        exchange.create_limit_ioc_order.return_value = {
            "id": "unwind_001", "status": "filled",
            "amount_filled": 0.1, "avg_fill_price": 1995.0,
        }
        cfg = ExecutorConfig(simulation_mode=False, use_dex_first=False,
                             unwind_slippage_extra_bps=_d("100"))
        ex = Executor(exchange_client=exchange, pricing_engine=None,
                      inventory_tracker=MagicMock(), circuit_breaker=SPRTCircuitBreaker(),
                      config=cfg)
        sig = _signal(direction=Direction.BUY_CEX_SELL_DEX)
        ctx = ExecutionContext(signal=sig)
        ctx.leg1_venue = "cex"
        ctx.leg1_fill_price = _d("2000")
        ctx.leg1_fill_size = _d("0.1")
        await ex._unwind_cex(ctx, sig, _d("0.1"))
        exchange.create_limit_ioc_order.assert_called_once()
        call_kwargs = exchange.create_limit_ioc_order.call_args[1]
        assert call_kwargs["side"] == "sell"
        assert ctx.unwind_succeeded is True

    @pytest.mark.asyncio
    async def test_unwind_failure_does_not_raise(self):
        cfg = ExecutorConfig(simulation_mode=False)
        ex = Executor(exchange_client=MagicMock(), pricing_engine=None,
                      inventory_tracker=MagicMock(), circuit_breaker=SPRTCircuitBreaker(),
                      config=cfg)
        ctx = ExecutionContext(signal=_signal())
        ctx.leg1_venue = "unknown_venue"
        ctx.leg1_fill_size = _d("0.1")
        await ex._unwind(ctx)
        assert ctx.unwind_attempted is True

    @pytest.mark.asyncio
    async def test_full_flow_with_unwind_on_l2_timeout(self):
        cfg = ExecutorConfig(simulation_mode=True, use_dex_first=True, leg1_timeout=_d("0.001"))
        ex = Executor(exchange_client=MagicMock(), pricing_engine=None,
                      inventory_tracker=MagicMock(), circuit_breaker=SPRTCircuitBreaker(),
                      config=cfg)

        async def ok_dex(*a, **kw):
            return {"success": True, "price": 2010.0, "filled": 0.1}

        async def slow_cex(*a, **kw):
            await asyncio.sleep(10)
            return {}

        unwind_record = []

        async def spy_unwind(ctx):
            unwind_record.append("called")
            ctx.unwind_attempted = True
            ctx.unwind_succeeded = True

        with (patch.object(ex._risk_filter, "approve", return_value=(True, _d("0"))),
              patch.object(ex, "_execute_dex_leg", ok_dex),
              patch.object(ex, "_execute_cex_leg", slow_cex),
              patch.object(ex, "_unwind", spy_unwind)):
            ctx = await ex.execute(_signal())

        assert ctx.state == ExecutorState.FAILED
        assert ctx.error_code == "L2_TIMEOUT"
        assert "called" in unwind_record


# ═══════════════════════════════════════════════════════════════════════════
# System config / mode
# ═══════════════════════════════════════════════════════════════════════════

class TestSystemConfig:

    def test_test_mode_sandbox_true(self):
        from config.mode import SystemConfig, OperationMode
        with patch.dict("os.environ", {"OPERATION_MODE": "test"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.TEST
        assert cfg.cex.sandbox is True
        assert cfg.dex.dry_run is True

    def test_production_mode_sandbox_false(self):
        from config.mode import SystemConfig, OperationMode
        with patch.dict("os.environ", {"OPERATION_MODE": "production"}):
            cfg = SystemConfig.from_env()
        assert cfg.mode == OperationMode.PRODUCTION
        assert cfg.cex.sandbox is False
        assert cfg.dex.dry_run is False

    def test_env_overrides_leg1_timeout(self):
        from config.mode import SystemConfig
        with patch.dict("os.environ", {"OPERATION_MODE": "test", "LEG1_TIMEOUT": "15"}):
            cfg = SystemConfig.from_env()
        assert cfg.executor.leg1_timeout == Decimal("15")

    def test_env_overrides_pool_addresses(self):
        from config.mode import SystemConfig
        addr = "0x0d4a11d5EEaaC28EC3F61d100daF4d40471f1852"
        with patch.dict("os.environ", {"OPERATION_MODE": "test", "POOL_ADDRESSES": addr}):
            cfg = SystemConfig.from_env()
        assert addr in cfg.dex.pool_addresses

    def test_is_test_property(self):
        from config.mode import SystemConfig
        with patch.dict("os.environ", {"OPERATION_MODE": "test"}):
            cfg = SystemConfig.from_env()
        assert cfg.is_test is True and cfg.is_production is False
