"""
Production-ready CEX-DEX arbitrage bot.

Architecture:
  tick(t)
    -> KalmanFilter.update(z_t)
    -> BayesianSignalDetector.test()
    -> EntropyCRITIC.score()
    -> MaxHeap(TOPSIS).push()
    -> PFA.execute()
    -> SPRT.observe()
    -> PnLEngine.record()

Safety layers (per tick):
  1. Dead man heartbeat
  2. Manual kill switch
  3. Auto kill switch (capital / error-rate)
  4. SPRT circuit breaker
  5. PreTradeValidator
  6. RiskManager soft limits
  7. safety_check() absolute ceiling
  8. BinanceTradingRules order filter
  9. PFA executor + VaR gate
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from web3 import Web3

from config.mode import SystemConfig
from src.chain.client import ChainClient
from src.core.wallet import WalletManager
from src.exchange.dex import DEXPriceSource, DEXExecutor
from src.pricing.engine import PricingEngine
from src.pricing.fork_simulator import ForkedChain, TradeSimulator
from src.inventory.tracker import VenueTracker, Venue
from src.inventory.pnl import ArbTrade, TradeLeg, PnLTracker

from src.strategy.generator import FeeStructure, SignalGenerator, SignalGeneratorConfig
from src.strategy.scorer import SignalScorer, ScorerConfig
from src.strategy.signal import Signal

from src.executor.engine import Executor, ExecutorConfig, ExecutorState
from src.executor.recovery import SPRTCircuitBreaker, SPRTConfig, LLMAnomalyAdvisor

from src.safety import (
    RiskLimits, RiskManager, PreTradeValidator,
    ManualKillSwitch, AutoKillSwitch, DeadManSwitch,
    safety_check, TelegramAlerter, BotMonitor,
    BalanceVerifier, configure_logging
)

log = logging.getLogger("arb_bot")


@dataclass(order=True)
class _PQEntry:
    neg_score: float
    insert_time: float
    signal: Signal = field(compare=False)


class ArbBot:
    """Production-ready arbitrage bot with full safety integration."""

    def __init__(self, system_config: SystemConfig) -> None:
        self._cfg = system_config
        self._dry_run = system_config.dry_run
        log.info("ArbBot init | mode=%s dry_run=%s chain_id=%d",
                 system_config.mode.value, self._dry_run, system_config.dex.chain_id)
        if self._dry_run:
            log.info("DRY-RUN: signals logged, NO trades executed")

        # Wallet
        try:
            self._wallet = WalletManager.from_env()
            log.info("Wallet: %s", self._wallet.address)
        except Exception as exc:
            log.warning("Wallet unavailable (%s) - simulation", exc)
            self._wallet = None

        # Chain / pricing
        self._chain_client = self._pricing_engine = self._dex_price = None
        rpc_url = system_config.dex.rpc_url
        if rpc_url:
            try:
                self._chain_client = ChainClient([rpc_url])
                cid = self._chain_client.get_chain_id()
                log.info("Chain connected | chain_id=%d (%s)", cid, system_config.network_preset.name)
                fork = ForkedChain(Web3(Web3.HTTPProvider(rpc_url)))
                self._pricing_engine = PricingEngine(
                    self._chain_client, TradeSimulator(fork), system_config.dex.ws_url)
                self._dex_price = DEXPriceSource(
                    pricing_engine=self._pricing_engine,
                    chain_client=self._chain_client,
                    router_address=system_config.dex.router_address,
                    pool_addresses=system_config.dex.pool_addresses,
                    fee_bps=system_config.dex_swap_bps,
                    slippage_bps=system_config.dex.slippage_tolerance_bps,
                    gas_price_gwei=20
                )
                if system_config.dex.pool_addresses:
                    self._dex_price.initialize()
                    log.info("DEX price source ready | pools=%d", len(system_config.dex.pool_addresses))
                else:
                    log.warning("No POOL_ADDRESSES - DEX uses fallback pricing")
            except Exception as exc:
                log.warning("Chain/DEX init failed: %s", exc)

        # DEX executor
        self._dex_exec = None
        if self._chain_client and self._wallet:
            try:
                self._dex_exec = DEXExecutor(
                    chain_client=self._chain_client, wallet=self._wallet,
                    router_address=system_config.dex.router_address,
                    gas_limit_swap=system_config.dex.gas_limit_swap,
                    gas_limit_approval=system_config.dex.gas_limit_approval,
                    slippage_bps=system_config.dex.slippage_tolerance_bps,
                    deadline_seconds=system_config.dex.deadline_seconds,
                    dry_run=self._dry_run
                )
                log.info("DEX executor ready | dry_run=%s", self._dry_run)
            except Exception as exc:
                log.warning("DEX executor failed: %s", exc)

        # CEX
        try:
            from src.exchange.client import BinanceClient
            self.exchange = BinanceClient({
                "apiKey": system_config.cex.api_key,
                "secret": system_config.cex.secret,
                "sandbox": system_config.cex.sandbox,
                "options": {"defaultType": system_config.cex.default_type},
                "enableRateLimit": system_config.cex.enable_rate_limit
            })
            log.info("CEX connected | sandbox=%s pair=%s",
                     system_config.cex.sandbox, system_config.trading_pair)
        except Exception as exc:
            log.warning("CEX unavailable: %s — stub active", exc)
            self.exchange = _StubExchange()

        self._trading_rules = system_config.cex.trading_rules
        self.inventory = VenueTracker([Venue.BINANCE, Venue.WALLET])
        self.pnl_engine = PnLTracker()

        # Safety
        rc = system_config.risk
        self._risk_manager = RiskManager(
            limits=RiskLimits(
                max_trade_usd=rc.max_trade_usd, max_trade_pct=rc.max_trade_pct,
                max_position_per_token=rc.max_position_per_token,
                max_open_positions=rc.max_open_positions,
                max_loss_per_trade=rc.max_loss_per_trade,
                max_daily_loss=rc.max_daily_loss,
                max_drawdown_pct=rc.max_drawdown_pct,
                max_trades_per_hour=rc.max_trades_per_hour,
                consecutive_loss_limit=rc.consecutive_loss_limit,
                max_spread_bps=rc.max_spread_bps,
                max_signal_age_seconds=rc.max_signal_age_seconds
            ),
            initial_capital=rc.initial_capital
        )
        self._validator = PreTradeValidator(
            max_spread_bps=rc.max_spread_bps,
            max_signal_age_seconds=rc.max_signal_age_seconds
        )
        self._manual_kill = ManualKillSwitch()
        self._alerter = TelegramAlerter(
            bot_token=system_config.telegram_bot_token,
            chat_id=system_config.telegram_chat_id
        )
        self._auto_kill = AutoKillSwitch(
            manual_switch=self._manual_kill, capital_floor_pct=0.50,
            alert_callback=lambda msg: asyncio.create_task(self._alerter.critical(msg))
        )
        self._dead_man = DeadManSwitch()
        self._balance_verifier = None
        if self._chain_client:
            self._balance_verifier = BalanceVerifier(
                exchange_client=self.exchange,
                chain_client=self._chain_client,
                wallet_address=self._wallet.address if self._wallet else "",
                inventory_tracker=self.inventory,
                stop_callback=self.stop,
                alert_callback=lambda msg: asyncio.create_task(self._alerter.critical(msg))
            )

        # Strategy
        self.fees = FeeStructure(
            cex_taker_bps=system_config.cex_taker_bps,
            dex_swap_bps=system_config.dex_swap_bps,
            gas_cost_usd=system_config.gas_cost_usd
        )
        self.generator = SignalGenerator(
            exchange_client=self.exchange,
            pricing_engine=self._pricing_engine,
            inventory_tracker=self.inventory,
            fee_structure=self.fees,
            config=SignalGeneratorConfig(
                alpha=Decimal("0.10"),
                kelly_fraction=system_config.kelly_fraction,
                max_position_usd=system_config.max_position_usd,
                signal_ttl_seconds=system_config.signal_ttl_seconds,
                cooldown_seconds=system_config.cooldown_seconds
            ),
            dex_price_source=self._dex_price
        )
        self.scorer = SignalScorer(ScorerConfig())
        self.circuit_breaker = SPRTCircuitBreaker(
            SPRTConfig(p0=Decimal("0.10"), p1=Decimal("0.40"), gamma=Decimal("0.95")))

        ec = system_config.executor
        self.executor = Executor(
            exchange_client=self.exchange,
            pricing_engine=self._pricing_engine,
            inventory_tracker=self.inventory,
            circuit_breaker=self.circuit_breaker,
            config=ExecutorConfig(
                leg1_timeout=ec.leg1_timeout, leg2_timeout=ec.leg2_timeout,
                min_fill_ratio=ec.min_fill_ratio, var_confidence=ec.var_confidence,
                vol_per_sqrt_second=ec.vol_per_sqrt_second,
                use_dex_first=ec.use_dex_first, simulation_mode=self._dry_run,
                max_recovery_attempts=ec.max_recovery_attempts,
                unwind_timeout=ec.unwind_timeout,
                unwind_slippage_extra_bps=ec.unwind_slippage_bps
            ),
            dex_price_source=self._dex_price,
            dex_executor=self._dex_exec
        )
        self._llm_advisor = LLMAnomalyAdvisor(api_key=os.getenv("OPENAI_API_KEY", ""))
        self._monitor = BotMonitor(
            risk_manager=self._risk_manager,
            circuit_breaker=self.circuit_breaker,
            kill_switch=self._manual_kill,
            dead_man_switch=self._dead_man,
            alerter=self._alerter
        )

        self.pairs: list[str] = [system_config.trading_pair]
        self.trade_size: float = 0.01
        self.min_score_threshold: float = 0.55
        self.max_queue_depth: int = 10
        self.running = False
        self._pq: list[_PQEntry] = []

    @classmethod
    def from_config(cls, cfg: SystemConfig | None = None) -> "ArbBot":
        return cls(cfg or SystemConfig.from_env())

    async def run(self) -> None:
        self.running = True
        log.info("Bot running | mode=%s dry_run=%s", self._cfg.mode.value, self._dry_run)
        await self._alerter.info(f"Bot started | mode={self._cfg.mode.value} dry_run={self._dry_run}")
        asyncio.create_task(self._monitor.run_health_loop())
        await self._sync_balances()
        while self.running:
            try:
                await self._tick()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Tick error: %s", exc)
                self._risk_manager.record_error()
                await asyncio.sleep(5.0)
        await self._shutdown()

    def stop(self) -> None:
        log.warning("Bot stop requested")
        self.running = False

    async def _tick(self) -> None:
        self._dead_man.write_heartbeat()

        if self._manual_kill.is_active():
            log.critical("MANUAL KILL SWITCH ACTIVE - stopping")
            await self._alerter.kill_switch_activated("manual kill switch")
            self.stop()
            return

        if self._auto_kill.check(self._risk_manager):
            log.critical("AUTO KILL SWITCH: %s", self._auto_kill.reason)
            self.stop()
            return

        if self.circuit_breaker.is_open():
            log.info("CB OPEN - reset in %.0fs", self.circuit_breaker.time_until_reset())
            return

        new_signals: list[Signal] = []
        for pair in self.pairs:
            sig = self.generator.generate(pair, self.trade_size)
            if sig is None:
                continue

            valid, v_reason = self._validator.validate_signal(sig)
            if not valid:
                log.warning("VALIDATION_FAIL | %s | %s", sig.signal_id, v_reason)
                continue

            qty = self._trading_rules.round_quantity(float(sig.kelly_size))
            price = self._trading_rules.round_price(float(sig.cex_price))
            order_ok, order_reason = self._trading_rules.validate_order(qty, price)
            if not order_ok:
                log.warning("ORDER_FILTER | %s | %s", sig.signal_id, order_reason)
                continue

            if sig.is_anomalous() and self._llm_advisor.should_query(
                sig.innovation_zscore, self.circuit_breaker.lambda_statistic,
                self.circuit_breaker._A, self.circuit_breaker._B,
            ):
                expl = await self._llm_advisor.advise(
                    pair=pair, innovation_zscore=sig.innovation_zscore,
                    filtered_spread=sig.filtered_spread, raw_spread_bps=sig.raw_spread_bps,
                    recent_pnl_summary=self._risk_manager.daily_summary()
                )
                if expl and expl.suppress_signal:
                    log.warning("LLM_SUPPRESS | %s | %s", sig.signal_id, expl.anomaly_type)
                    continue

            new_signals.append(sig)

        if not new_signals:
            return

        skews = self._get_skews()
        scores = self.scorer.score_batch(new_signals, skews)
        for sig in new_signals:
            score = scores.get(sig.signal_id, Decimal("0"))
            sig.score = score
            if float(score) >= self.min_score_threshold:
                heapq.heappush(self._pq, _PQEntry(-float(score), time.time(), sig))
                log.info("QUEUED | %s spread=%.1fbps score=%.4f",
                         sig.pair, sig.raw_spread_bps, score)

        while len(self._pq) > self.max_queue_depth:
            heapq.heappop(self._pq)

        while self._pq:
            entry = heapq.heappop(self._pq)
            sig = entry.signal
            if sig.is_expired():
                continue
            await self._execute_signal(sig)
            break

    async def _execute_signal(self, sig: Signal) -> None:
        trade_usd = float(sig.kelly_size * sig.cex_price)

        rm_ok, rm_reason = self._risk_manager.check_pre_trade(trade_usd=trade_usd)
        if not rm_ok:
            log.warning("RISK_BLOCK | %s | %s", sig.signal_id, rm_reason)
            return

        abs_ok, abs_reason = safety_check(
            trade_usd=trade_usd, daily_pnl=self._risk_manager.daily_pnl,
            total_capital=self._risk_manager.current_capital,
            trades_this_hour=self._risk_manager.trades_this_hour,
            errors_this_hour=self._risk_manager.errors_this_hour
        )
        if not abs_ok:
            log.critical("ABS_SAFETY_BLOCK | %s | %s", sig.signal_id, abs_reason)
            await self._alerter.critical(f"Safety block: {abs_reason}")
            return

        if self._dry_run:
            log.info(
                "DRY_RUN | %s %s size=%.4f spread=%.1fbps pnl=$%.4f score=%.4f",
                sig.pair, sig.direction.value, sig.kelly_size,
                sig.raw_spread_bps, sig.expected_net_pnl, sig.score
            )
            return

        log.info("EXECUTE | %s | %s kelly=%.4f", sig.pair, sig.direction.value, sig.kelly_size)
        self._risk_manager.open_position()
        try:
            ctx = await self.executor.execute(sig)
        finally:
            self._risk_manager.close_position()

        net_pnl = float(ctx.actual_net_pnl or Decimal("0"))
        if ctx.state == ExecutorState.DONE:
            self.scorer.record_result(sig.pair, success=True)
            self._risk_manager.record_trade(
                net_pnl=net_pnl, pair=sig.pair, direction=sig.direction.value,
                size_usd=trade_usd, gross_pnl=float(ctx.leg_gap_pnl or 0),
                spread_bps=float(sig.raw_spread_bps),
                signal_age_s=float(sig.age_seconds()), state=ctx.state.name
            )
            self._monitor.log_trade_metrics(ctx)
            log.info("SUCCESS | %s | pnl=$%.4f duration=%.2fs", sig.pair, net_pnl, float(ctx.duration()))
            if self._balance_verifier:
                await self._balance_verifier.verify()
            self._record_pnl(ctx)
        else:
            self.scorer.record_result(sig.pair, success=False)
            self._risk_manager.record_error()
            log.warning("FAILED | %s | code=%s unwind=%s",
                        sig.pair, ctx.error_code,
                        "OK" if ctx.unwind_succeeded else "FAILED" if ctx.unwind_attempted else "N/A")
            if ctx.unwind_attempted and not ctx.unwind_succeeded:
                await self._alerter.critical(f"Unwind FAILED {sig.pair} | {ctx.error}")

        await self._sync_balances()

    async def _sync_balances(self) -> None:
        if isinstance(self.exchange, _StubExchange):
            return
        try:
            self.inventory.update_from_cex(Venue.BINANCE, self.exchange.fetch_balance())
        except Exception as exc:
            log.warning("Balance sync failed: %s", exc)
            self._risk_manager.record_error()

    def _get_skews(self) -> list[dict]:
        try:
            return self.inventory.all_skews()
        except Exception:
            return []

    def _record_pnl(self, ctx) -> None:
        try:
            sig = ctx.signal
            self.pnl_engine.record(ArbTrade(
                trade_id=sig.signal_id,
                opened_at=datetime.fromtimestamp(float(ctx.started_at)),
                buy_leg=TradeLeg(
                    leg_id=f"{sig.signal_id}_buy",
                    executed_at=datetime.fromtimestamp(float(ctx.started_at)),
                    venue=Venue.BINANCE if ctx.leg1_venue == "cex" else Venue.WALLET,
                    symbol=sig.pair, side="buy",
                    quantity=Decimal(str(ctx.leg1_fill_size or 0)),
                    price=Decimal(str(ctx.leg1_fill_price or 0)),
                    fee=Decimal("0"), fee_currency=sig.pair.split("/")[1]
                ),
                sell_leg=TradeLeg(
                    leg_id=f"{sig.signal_id}_sell",
                    executed_at=datetime.fromtimestamp(float(ctx.finished_at or ctx.started_at)),
                    venue=Venue.WALLET if ctx.leg2_venue == "dex" else Venue.BINANCE,
                    symbol=sig.pair, side="sell",
                    quantity=Decimal(str(ctx.leg2_fill_size or 0)),
                    price=Decimal(str(ctx.leg2_fill_price or 0)),
                    fee=Decimal("0"), fee_currency=sig.pair.split("/")[1]
                ),
            ))
        except Exception as exc:
            log.warning("PnL record failed: %s", exc)

    async def _shutdown(self) -> None:
        summary = self._risk_manager.daily_summary()
        log.info("DAILY_SUMMARY | trades=%d pnl=$%.2f capital=$%.2f",
                 summary.get("trades", 0), summary.get("total_pnl", 0.0), summary.get("capital", 0.0))
        await self._alerter.daily_summary(summary)
        log.info("Bot stopped cleanly")


class _StubExchange:
    def fetch_order_book(self, pair: str) -> dict:
        import random
        mid = 2000.0 + random.uniform(-10, 10)
        return {
            "symbol": pair, "timestamp": int(time.time() * 1000),
            "bids": [(mid - 0.25, 10.0)], "asks": [(mid + 0.25, 10.0)],
            "best_bid": (mid - 0.25, 10.0), "best_ask": (mid + 0.25, 10.0),
            "mid_price": mid, "spread_bps": 2.5
        }

    def fetch_balance(self) -> dict:
        return {"ETH": {"free": "100", "locked": "0", "total": "100"},
                "USDC": {"free": "200000", "locked": "0", "total": "200000"}}

    def create_limit_ioc_order(self, **kw) -> dict:
        return {"id": "stub_001", "status": "filled",
                "amount_filled": kw.get("amount", 0.01), "avg_fill_price": kw.get("price", 2000.0)}


if __name__ == "__main__":
    cfg = SystemConfig.from_env()
    configure_logging(log_dir=cfg.log_dir)
    bot = ArbBot.from_config(cfg)
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
