"""
Event-driven arbitrage bot with priority-queue signal dispatch.

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
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from web3 import Web3

from src.chain.client import ChainClient
from src.pricing.engine import PricingEngine
from src.pricing.fork_simulator import TradeSimulator, ForkedChain
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
    neg_score: float   # Score for max-heap
    insert_time: float
    signal: Signal = field(compare=False)


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

class ArbBot:
    """
    Event-driven CEX-DEX arbitrage bot.

    Signal pipeline:
      1. Signal Generator (Kalman) produces candidates.
      2. Opportunity Scorer (TOPSIS) ranks them.
      3. Priority queue dispatches highest-ranked signal.
      4. PFA Executor executes it.
      5. SPRT Circuit Breaker monitors health.
    """

    def __init__(self, config: dict) -> None:
        # Exchange client
        try:
            from config.config import BINANCE_CONFIG
            from src.exchange.client import BinanceClient
            self.exchange = BinanceClient(BINANCE_CONFIG)
        except Exception as exc:
            log.warning(f"Real mode is not active: {exc}. Simulation mode activated")
            self.exchange = None   # Simulation mode

        self.pricing_engine = None
        self.chain_client = None
        if config.get("rpc_url"):
            try:
                self.chain_client = ChainClient([config["rpc_url"]])
                fork = ForkedChain(Web3(Web3.HTTPProvider(config["rpc_url"])))
                sim = TradeSimulator(fork)
                self.pricing_engine = PricingEngine(
                    self.chain_client, sim, config.get("ws_url", "")
                )
            except Exception as exc:
                log.warning("Pricing engine unavailable: %s", exc)

        try:
            self.inventory = VenueTracker([Venue.BINANCE, Venue.WALLET])
            self._Venue = Venue
        except Exception as exc:
            log.warning(f"Real inventory unavailable: {exc}. Stub inventory will be used")
            self.inventory = _StubInventory()
            self._Venue = None

        # PnL engine
        try:
            self.pnl_engine = PnLTracker()
        except Exception as exc:
            log.warning(f"PnL engine is not active: {exc}")
            self.pnl_engine = None

        # Fees
        self.fees = FeeStructure(
            cex_taker_bps=config.get("cex_taker_bps", Decimal(10.0)),
            dex_swap_bps=config.get("dex_swap_bps", Decimal(30.0)),
            gas_cost_usd=config.get("gas_cost_usd", Decimal(3.0))
        )

        # Kalman signal generator
        gen_cfg = SignalGeneratorConfig(**config.get("signal_config", {}))
        self.generator = SignalGenerator(
            exchange_client=self.exchange or _StubExchange(),
            pricing_engine=self.pricing_engine,
            inventory_tracker=self.inventory,
            fee_structure=self.fees,
            config=gen_cfg
        )

        # TOPSIS scorer
        self.scorer = SignalScorer(ScorerConfig(**config.get("scorer_config", {})))

        # SPRT circuit breaker
        self.circuit_breaker = SPRTCircuitBreaker(
            SPRTConfig(**config.get("sprt_config", {}))
        )

        # PFA Executor
        exec_cfg = ExecutorConfig(**config.get("executor_config", {}))
        self.executor = Executor(
            exchange_client=self.exchange or _StubExchange(),
            pricing_engine=self.pricing_engine,
            inventory_tracker=self.inventory,
            circuit_breaker=self.circuit_breaker,
            config=exec_cfg
        )

        # LLM anomaly advisor
        self.llm_advisor = LLMAnomalyAdvisor(
            api_key=config.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
            model=config.get("llm_model", "gpt-5.4-nano")
        )

        # Bot parameters
        self.pairs: list[str] = config.get("pairs", ["ETH/USDT"])
        self.trade_size: float = config.get("trade_size", 0.1)
        self.min_score_threshold: float = config.get("min_score_threshold", 0.55)
        self.max_queue_depth: int = config.get("max_queue_depth", 10)
        self.running = False

        # Priority queue (max-heap via negated score)
        self._pq: list[_PQEntry] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> None:
        self.running = True
        log.info("Bot starting - pairs=%s", self.pairs)
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
                "Circuit breaker OPEN - reset in %.0fs",
                self.circuit_breaker.time_until_reset(),
            )
            return

        # --- Generate signals for all pairs ---
        new_signals: list[Signal] = []
        for pair in self.pairs:
            sig = self.generator.generate(pair, self.trade_size)
            if sig is None:
                continue

            # LLM anomaly check
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
                        sig.signal_id, explanation.anomaly_type, explanation.reasoning
                    )
                    continue

            new_signals.append(sig)

        if not new_signals:
            log.info("No new signal found yet...")
            return

        # --- Score all new signals in a TOPSIS batch ---
        skews = self._get_skews()
        scores = self.scorer.score_batch(new_signals, skews)

        for sig in new_signals:
            score = scores.get(sig.signal_id, 0.0)
            sig.score = score
            if score >= self.min_score_threshold:
                heapq.heappush(self._pq, _PQEntry(-score, time.time(), sig))
                log.info(
                    "Queued | %s | spread=%.1fbps conf=%.3f score=%.4f",
                    sig.pair, sig.raw_spread_bps, sig.signal_confidence, score
                )

        # Trim queue to max depth
        while len(self._pq) > self.max_queue_depth:
            heapq.heappop(self._pq)

        # --- Execute highest-priority valid signal ---
        while self._pq:
            entry = heapq.heappop(self._pq)
            sig = entry.signal
            if sig.is_expired():
                log.debug("Signal %s expired, skipping.", sig.signal_id)
                continue
            await self._execute_signal(sig)
            break   # One execution per tick

    async def _execute_signal(self, sig: Signal) -> None:
        log.info(
            "Executing | %s | dir=%s kelly=%.4f net_pnl_est=$%.4f",
            sig.pair, sig.direction.value, sig.kelly_size, sig.expected_net_pnl
        )

        ctx = await self.executor.execute(sig)

        if ctx.state == ExecutorState.DONE:
            self.scorer.record_result(sig.pair, success=True)
            log.info(
                "SUCCESS | %s | PnL=$%.4f duration=%.2fs phi=%.3f",
                sig.pair,
                ctx.actual_net_pnl or 0.0,
                ctx.duration(),
                ctx.fill_quality or 1.0
            )
            # Record in PnL engine
            if self.pnl_engine:
                try:
                    record = self._build_arb_record(ctx)
                    if record:
                        self.pnl_engine.record(record)
                except Exception as exc:
                    log.warning("PnL record failed: %s", exc)
        else:
            self.scorer.record_result(sig.pair, success=False)
            log.warning(
                "FAILED | %s | error=%s code=%s duration=%.2fs",
                sig.pair, ctx.error, ctx.error_code, ctx.duration()
            )

        await self._sync_balances()

    # ------------------------------------------------------------------
    # Balance sync
    # ------------------------------------------------------------------

    async def _sync_balances(self) -> None:
        if self.exchange is None or self._Venue is None:
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
        """Bridge ExecutionContext -> ArbRecord."""
        try:
            sig = ctx.signal
            buy_leg = TradeLeg(
                leg_id=f"{sig.signal_id}_buy",
                executed_at=datetime.fromtimestamp(ctx.started_at),
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
                executed_at=datetime.fromtimestamp(ctx.finished_at or ctx.started_at),
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
                opened_at=datetime.fromtimestamp(ctx.started_at),
                buy_leg=buy_leg,
                sell_leg=sell_leg
            )
        except Exception as exc:
            log.warning("Building ArbRecord failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Stub helpers for pure simulation
# ---------------------------------------------------------------------------

class _StubExchange:
    """Minimal stub exchange for simulation."""

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
            "id": "stub_001", "status": "filled",
            "amount_filled": kwargs.get("amount", 0.1),
            "avg_fill_price": kwargs.get("price", 2000.0)
        }


class _StubInventory:
    """Minimal inventory stub for pure simulation."""

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
    config = {
        "apiKey": os.getenv("BINANCE_TESTNET_API_KEY", ""),
        "secret": os.getenv("BINANCE_TESTNET_SECRET", ""),
        "sandbox": True,
        "rpc_url": os.getenv("ETH_RPC_URL", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "pairs": ["ETH/USDT"],
        "trade_size": 0.1,
        "min_score_threshold": 0.55,
        "signal_config": {
            "alpha": 0.5,
            "kelly_fraction": 0.25,
            "cooldown_seconds": 2.0,
            "signal_ttl_seconds": 5.0
        },
        "executor_config": {
            "simulation_mode": True,
            "use_dex_first": True
        },
        "sprt_config": {
            "p0": 0.10,
            "p1": 0.40,
            "gamma": 0.95
        },
    }
    bot = ArbBot(config)
    asyncio.run(bot.run())
