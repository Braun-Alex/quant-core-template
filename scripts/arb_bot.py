"""
Event-driven CEX-DEX arbitrage bot.

Architecture:
  tick(t)
    -> KalmanFilter.update(z_t)
    -> BayesianSignalDetector.test()
    -> EntropyCRITIC.score()
    -> MaxHeap(TOPSIS).push()
    -> PFA.execute()
    -> SPRT.observe()
    -> PnLEngine.record()
"""

from __future__ import annotations

import asyncio
import heapq
import logging
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

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


# ---------------------------------------------------------------------------
# Priority queue entry (max-heap via negated score)
# ---------------------------------------------------------------------------

@dataclass(order=True)
class _PQEntry:
    neg_score: float
    insert_time: float
    signal: Signal = field(compare=False)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class ArbBot:
    """
    Event-driven CEX-DEX arbitrage bot.

    Supports two modes driven by SystemConfig:
      test       → Binance Testnet and Ethereum Mainnet
      production → Binance Mainnet and Ethereum Mainnet
    """

    def __init__(self, system_config: SystemConfig) -> None:
        self._sys_cfg = system_config
        log.info(
            "ArbBot init | mode=%s dry_run=%s",
            system_config.mode.value,
            system_config.dex.dry_run
        )

        # ── Wallet ─────────────────────────────────────────────────────
        try:
            self._wallet = WalletManager.from_env()
            log.info("Wallet: %s", self._wallet.address)
        except Exception as exc:
            log.warning("Wallet not available: %s - simulation mode active", exc)
            self._wallet = None

        # ── Chain client ────────────────────────────────────────────────
        self._chain_client: ChainClient | None = None
        rpc_url = system_config.dex.rpc_url
        if rpc_url:
            try:
                self._chain_client = ChainClient([rpc_url])
                cid = self._chain_client.get_chain_id()
                log.info("Chain client ready | chain_id=%d", cid)
            except Exception as exc:
                log.warning("Chain client unavailable: %s", exc)

        # ── Pricing engine and DEX price source ───────────────────────────
        self._pricing_engine: PricingEngine | None = None
        self._dex_price: DEXPriceSource | None = None
        if self._chain_client:
            try:
                fork = ForkedChain(Web3(Web3.HTTPProvider(rpc_url)))
                sim = TradeSimulator(fork)
                self._pricing_engine = PricingEngine(
                    self._chain_client, sim, system_config.dex.ws_url
                )

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
                    log.info("DEX price source ready")
                else:
                    log.warning("No POOL_ADDRESSES configured - DEX prices will use fallback")
            except Exception as exc:
                log.warning("Pricing engine / DEX price init failed: %s", exc)

        # ── DEX executor ────────────────────────────────────────────────
        self._dex_exec: DEXExecutor | None = None
        if self._chain_client and self._wallet:
            try:
                self._dex_exec = DEXExecutor(
                    chain_client=self._chain_client,
                    wallet=self._wallet,
                    router_address=system_config.dex.router_address,
                    gas_limit_swap=system_config.dex.gas_limit_swap,
                    gas_limit_approval=system_config.dex.gas_limit_approval,
                    slippage_bps=system_config.dex.slippage_tolerance_bps,
                    deadline_seconds=system_config.dex.deadline_seconds,
                    dry_run=system_config.dex.dry_run
                )
                log.info("DEX executor ready | dry_run=%s", system_config.dex.dry_run)
            except Exception as exc:
                log.warning("DEX executor init failed: %s", exc)

        # ── CEX client ──────────────────────────────────────────────────
        self.exchange = None
        try:
            from src.exchange.client import BinanceClient
            self.exchange = BinanceClient({
                "apiKey": system_config.cex.api_key,
                "secret": system_config.cex.secret,
                "sandbox": system_config.cex.sandbox,
                "options": {"defaultType": system_config.cex.default_type},
                "enableRateLimit": system_config.cex.enable_rate_limit
            })
            log.info(
                "CEX client ready | sandbox=%s", system_config.cex.sandbox
            )
        except Exception as exc:
            log.warning("CEX client unavailable: %s — stub active", exc)
            self.exchange = _StubExchange()

        # ── Inventory ───────────────────────────────────────────────────
        try:
            self.inventory = VenueTracker([Venue.BINANCE, Venue.WALLET])
            self._Venue = Venue
        except Exception as exc:
            log.warning("Inventory init failed: %s — stub active", exc)
            self.inventory = _StubInventory()
            self._Venue = None

        # ── PnL engine ──────────────────────────────────────────────────
        try:
            self.pnl_engine = PnLTracker()
        except Exception as exc:
            log.warning("PnL engine init failed: %s", exc)
            self.pnl_engine = None

        # ── Fee structure ────────────────────────────────────────────────
        self.fees = FeeStructure(
            cex_taker_bps=system_config.cex_taker_bps,
            dex_swap_bps=system_config.dex_swap_bps,
            gas_cost_usd=system_config.gas_cost_usd
        )

        # ── Signal generator ────────────────────────────────────────────
        gen_cfg = SignalGeneratorConfig(
            alpha=Decimal("0.10"),
            kelly_fraction=system_config.kelly_fraction,
            max_position_usd=system_config.max_position_usd,
            signal_ttl_seconds=system_config.signal_ttl_seconds,
            cooldown_seconds=system_config.cooldown_seconds
        )
        self.generator = SignalGenerator(
            exchange_client=self.exchange,
            pricing_engine=self._pricing_engine,
            inventory_tracker=self.inventory,
            fee_structure=self.fees,
            config=gen_cfg,
            dex_price_source=self._dex_price
        )

        # ── TOPSIS scorer ────────────────────────────────────────────────
        self.scorer = SignalScorer(ScorerConfig())

        # ── SPRT circuit breaker ─────────────────────────────────────────
        self.circuit_breaker = SPRTCircuitBreaker(
            SPRTConfig(
                p0=Decimal("0.10"),
                p1=Decimal("0.40"),
                gamma=Decimal("0.95")
            )
        )

        # ── PFA executor ─────────────────────────────────────────────────
        ec = system_config.executor
        exec_cfg = ExecutorConfig(
            leg1_timeout=ec.leg1_timeout,
            leg2_timeout=ec.leg2_timeout,
            min_fill_ratio=ec.min_fill_ratio,
            var_confidence=ec.var_confidence,
            vol_per_sqrt_second=ec.vol_per_sqrt_second,
            use_dex_first=ec.use_dex_first,
            simulation_mode=(self.exchange is None or isinstance(self.exchange, _StubExchange)),
            max_recovery_attempts=ec.max_recovery_attempts,
            unwind_timeout=ec.unwind_timeout,
            unwind_slippage_extra_bps=ec.unwind_slippage_bps
        )
        self.executor = Executor(
            exchange_client=self.exchange,
            pricing_engine=self._pricing_engine,
            inventory_tracker=self.inventory,
            circuit_breaker=self.circuit_breaker,
            config=exec_cfg,
            dex_price_source=self._dex_price,
            dex_executor=self._dex_exec
        )

        # ── LLM anomaly advisor ──────────────────────────────────────────
        self.llm_advisor = LLMAnomalyAdvisor(
            api_key=system_config.openai_api_key,
            model="gpt-5.4-nano"
        )

        # ── Bot parameters ────────────────────────────────────────────────
        self.pairs: list[str] = ["ETH/USDT"]
        self.trade_size: float = 0.1
        self.min_score_threshold: float = 0.55
        self.max_queue_depth: int = 10
        self.running = False
        self._pq: list[_PQEntry] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.running = True
        log.info(
            "Bot starting | mode=%s pairs=%s",
            self._sys_cfg.mode.value, self.pairs,
        )
        await self._sync_balances()

        while self.running:
            try:
                await self._tick()
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.error("Tick error: %s", exc)
                await asyncio.sleep(5.0)

    def stop(self) -> None:
        self.running = False

    # ------------------------------------------------------------------
    # Tick
    # ------------------------------------------------------------------

    async def _tick(self) -> None:
        if self.circuit_breaker.is_open():
            log.info(
                "Circuit breaker OPEN — reset in %.0fs",
                self.circuit_breaker.time_until_reset()
            )
            return

        new_signals: list[Signal] = []
        for pair in self.pairs:
            sig = self.generator.generate(pair, self.trade_size)
            if sig is None:
                continue

            if sig.is_anomalous() and self.llm_advisor.should_query(
                sig.innovation_zscore,
                self.circuit_breaker.lambda_statistic,
                self.circuit_breaker._A,
                self.circuit_breaker._B
            ):
                explanation = await self.llm_advisor.advise(
                    pair=pair,
                    innovation_zscore=sig.innovation_zscore,
                    filtered_spread=sig.filtered_spread,
                    raw_spread_bps=sig.raw_spread_bps,
                    recent_pnl_summary={}
                )
                if explanation and explanation.suppress_signal:
                    log.warning(
                        "LLM suppressed signal %s: %s [%s]",
                        sig.signal_id,
                        explanation.anomaly_type,
                        explanation.reasoning
                    )
                    continue

            new_signals.append(sig)

        if not new_signals:
            log.info("No new signal yet...")
            return

        skews = self._get_skews()
        scores = self.scorer.score_batch(new_signals, skews)

        for sig in new_signals:
            score = scores.get(sig.signal_id, Decimal("0"))
            sig.score = score
            if float(score) >= self.min_score_threshold:
                heapq.heappush(self._pq, _PQEntry(-float(score), time.time(), sig))
                log.info(
                    "Queued | %s | spread=%.1fbps conf=%.3f score=%.4f",
                    sig.pair, sig.raw_spread_bps, sig.signal_confidence, score,
                )

        while len(self._pq) > self.max_queue_depth:
            heapq.heappop(self._pq)

        while self._pq:
            entry = heapq.heappop(self._pq)
            sig = entry.signal
            if sig.is_expired():
                log.debug("Signal %s expired — skipping", sig.signal_id)
                continue
            await self._execute_signal(sig)
            break

    async def _execute_signal(self, sig: Signal) -> None:
        log.info(
            "Executing | %s | dir=%s kelly=%.4f net_pnl_est=$%.4f",
            sig.pair, sig.direction.value, sig.kelly_size, sig.expected_net_pnl
        )

        ctx = await self.executor.execute(sig)

        if ctx.state == ExecutorState.DONE:
            self.scorer.record_result(sig.pair, success=True)
            log.info(
                "SUCCESS | %s | PnL=$%.4f duration=%.2fs phi=%.3f%s",
                sig.pair,
                ctx.actual_net_pnl or 0.0,
                ctx.duration(),
                ctx.fill_quality or 1.0,
                " [dry-run]" if self._sys_cfg.dex.dry_run else ""
            )
            if self.pnl_engine:
                try:
                    record = self._build_arb_record(ctx)
                    if record:
                        self.pnl_engine.record(record)
                except Exception as exc:
                    log.warning("PnL record failed: %s", exc)
        else:
            self.scorer.record_result(sig.pair, success=False)
            unwind_info = ""
            if ctx.unwind_attempted:
                unwind_info = (
                    f" | unwind={'OK' if ctx.unwind_succeeded else 'FAILED'}"
                    f" tx={ctx.unwind_tx_hash}"
                )
            log.warning(
                "FAILED | %s | error=%s code=%s duration=%.2fs%s",
                sig.pair, ctx.error, ctx.error_code, ctx.duration(), unwind_info
            )

        await self._sync_balances()

    # ------------------------------------------------------------------
    # Balance sync
    # ------------------------------------------------------------------

    async def _sync_balances(self) -> None:
        if self.exchange is None or isinstance(self.exchange, _StubExchange):
            return
        if self._Venue is None:
            return
        try:
            cex_balances = self.exchange.fetch_balance()
            self.inventory.update_from_cex(self._Venue.BINANCE, cex_balances)
        except Exception as exc:
            log.warning("Balance sync failed: %s", exc)

    def _get_skews(self) -> list[dict]:
        try:
            return self.inventory.all_skews()
        except Exception:
            return []

    # ------------------------------------------------------------------
    # PnL bridge
    # ------------------------------------------------------------------

    def _build_arb_record(self, ctx):
        try:
            sig = ctx.signal
            buy_leg = TradeLeg(
                leg_id=f"{sig.signal_id}_buy",
                executed_at=datetime.fromtimestamp(float(ctx.started_at)),
                venue=Venue.BINANCE if ctx.leg1_venue == "cex" else Venue.WALLET,
                symbol=sig.pair,
                side="buy",
                quantity=Decimal(str(ctx.leg1_fill_size or 0)),
                price=Decimal(str(ctx.leg1_fill_price or 0)),
                fee=Decimal("0"),
                fee_currency=sig.pair.split("/")[1]
            )
            sell_leg = TradeLeg(
                leg_id=f"{sig.signal_id}_sell",
                executed_at=datetime.fromtimestamp(
                    float(ctx.finished_at or ctx.started_at)
                ),
                venue=Venue.WALLET if ctx.leg2_venue == "dex" else Venue.BINANCE,
                symbol=sig.pair,
                side="sell",
                quantity=Decimal(str(ctx.leg2_fill_size or 0)),
                price=Decimal(str(ctx.leg2_fill_price or 0)),
                fee=Decimal("0"),
                fee_currency=sig.pair.split("/")[1]
            )
            return ArbTrade(
                trade_id=sig.signal_id,
                opened_at=datetime.fromtimestamp(float(ctx.started_at)),
                buy_leg=buy_leg,
                sell_leg=sell_leg
            )
        except Exception as exc:
            log.warning("Building ArbRecord failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

class _StubExchange:
    def fetch_order_book(self, pair: str) -> dict:
        import random
        mid = 2000.0 + random.uniform(-10, 10)
        spread = 0.5
        return {
            "symbol": pair,
            "timestamp": int(time.time() * 1000),
            "bids": [(mid - spread / 2, 10.0)],
            "asks": [(mid + spread / 2, 10.0)],
            "best_bid": (mid - spread / 2, 10.0),
            "best_ask": (mid + spread / 2, 10.0),
            "mid_price": mid,
            "spread_bps": spread / mid * 10_000
        }

    def fetch_balance(self) -> dict:
        return {
            "ETH": {"free": "100", "locked": "0", "total": "100"},
            "USDT": {"free": "200000", "locked": "0", "total": "200000"}
        }

    def create_limit_ioc_order(self, **kwargs) -> dict:
        return {
            "id": "stub_001",
            "status": "filled",
            "amount_filled": kwargs.get("amount", 0.1),
            "avg_fill_price": kwargs.get("price", 2000.0)
        }


class _StubInventory:
    def available(self, venue, asset: str) -> float:
        return 100_000.0

    def get_skews(self) -> list:
        return []

    def all_skews(self) -> list:
        return []

    def update_from_cex(self, *args) -> None:
        pass

    def update_from_wallet(self, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys_cfg = SystemConfig.from_env()
    bot = ArbBot(sys_cfg)
    asyncio.run(bot.run())
