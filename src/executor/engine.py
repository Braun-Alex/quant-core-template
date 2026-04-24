"""
Probabilistic Finite Automaton Executor.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, auto
from typing import Optional

from src.strategy.signal import Direction, Signal

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

log = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")
_EPS = Decimal("1E-12")


def _d(v) -> Decimal:
    return Decimal(str(v))


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class ExecutorState(Enum):
    IDLE = auto()
    VALIDATING = auto()
    LEG1_PENDING = auto()
    LEG1_FILLED = auto()
    LEG2_PENDING = auto()
    DONE = auto()
    FAILED = auto()
    UNWINDING = auto()
    RECOVERING = auto()


_TERMINAL = {ExecutorState.DONE, ExecutorState.FAILED}


# ---------------------------------------------------------------------------
# Execution context
# ---------------------------------------------------------------------------

@dataclass
class ExecutionContext:
    signal: Signal
    state: ExecutorState = ExecutorState.IDLE

    leg1_venue: str = ""
    leg1_order_id: Optional[str] = None
    leg1_fill_price: Optional[Decimal] = None
    leg1_fill_size: Optional[Decimal] = None
    leg1_slippage_bps: Optional[Decimal] = None

    leg2_venue: str = ""
    leg2_tx_hash: Optional[str] = None
    leg2_fill_price: Optional[Decimal] = None
    leg2_fill_size: Optional[Decimal] = None
    leg2_slippage_bps: Optional[Decimal] = None

    fill_quality: Optional[Decimal] = None
    var_alpha: Optional[Decimal] = None
    leg_gap_pnl: Optional[Decimal] = None

    started_at: Decimal = field(default_factory=lambda: _d(time.time()))
    finished_at: Optional[Decimal] = None
    actual_net_pnl: Optional[Decimal] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    state_history: list[tuple[ExecutorState, Decimal]] = field(default_factory=list)

    def transition(self, new_state: ExecutorState) -> None:
        self.state_history.append((new_state, _d(time.time())))
        self.state = new_state

    def duration(self) -> Decimal:
        end = self.finished_at or _d(time.time())
        return end - self.started_at

    def is_terminal(self) -> bool:
        return self.state in _TERMINAL


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExecutorConfig:
    leg1_timeout: Decimal = Decimal("5")
    leg2_timeout: Decimal = Decimal("60")
    min_fill_ratio: Decimal = Decimal("0.80")
    var_confidence: Decimal = Decimal("0.95")
    vol_per_sqrt_second: Decimal = Decimal("0.0002")
    use_dex_first: bool = True
    simulation_mode: bool = True
    max_recovery_attempts: int = 2

    def __post_init__(self) -> None:
        for f in ("leg1_timeout", "leg2_timeout", "min_fill_ratio",
                  "var_confidence", "vol_per_sqrt_second"):
            setattr(self, f, _d(getattr(self, f)))


# ---------------------------------------------------------------------------
# Risk filter
# ---------------------------------------------------------------------------

class ExecutionRiskFilter:
    """VaR gate: VaR_alpha = q * price * sigma * z_alpha — all Decimal."""

    def __init__(self, config: ExecutorConfig) -> None:
        self._cfg = config
        # Rational approximation
        self._z_alpha = _d(self._normal_quantile(float(config.var_confidence)))

    def approve(
            self, signal: Signal, expected_leg2_latency: Decimal
    ) -> tuple[bool, Decimal]:
        q = signal.kelly_size
        tau = max(expected_leg2_latency, _ZERO)
        sigma = self._cfg.vol_per_sqrt_second * _d(math.sqrt(float(tau)))
        var_alpha = q * signal.cex_price * sigma * self._z_alpha
        approved = var_alpha < signal.expected_net_pnl
        log.debug("RiskFilter VaR=%.6f pnl=%.6f ok=%s",
                  float(var_alpha), float(signal.expected_net_pnl), approved)
        return approved, var_alpha

    @staticmethod
    def _normal_quantile(p: float) -> float:
        """
        Standard-normal quantile via Abramowitz & Stegun 26.2.17.
        Error < 4.5e-4 for all p in (0, 1).
        """
        import math
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"p must be in (0, 1), got {p}")
        sign = 1.0 if p >= 0.5 else -1.0
        q = p if p >= 0.5 else 1.0 - p
        t = math.sqrt(-2.0 * math.log(1.0 - q))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        z = t - (c[0] + c[1] * t + c[2] * t * t) / (1.0 + d[0] * t + d[1] * t * t + d[2] * t * t * t)
        return sign * z


# ---------------------------------------------------------------------------
# Replay protection
# ---------------------------------------------------------------------------

class ReplayProtection:
    def __init__(self, ttl_seconds: Decimal = Decimal("60")) -> None:
        self._executed: dict[str, Decimal] = {}
        self._ttl = _d(ttl_seconds)

    def is_duplicate(self, signal: Signal) -> bool:
        self._cleanup()
        return signal.signal_id in self._executed

    def mark_executed(self, signal: Signal) -> None:
        self._executed[signal.signal_id] = _d(time.time())

    def _cleanup(self) -> None:
        cutoff = _d(time.time()) - self._ttl
        self._executed = {k: v for k, v in self._executed.items() if v > cutoff}


# ---------------------------------------------------------------------------
# PFA Executor
# ---------------------------------------------------------------------------

_FEE_FRACTION = Decimal("0.004")  # ~40 bps total round-trip fee


class Executor:
    """
    Probabilistic Finite Automaton Executor.
    """

    def __init__(
            self,
            exchange_client,
            pricing_engine,
            inventory_tracker,
            circuit_breaker,
            config: Optional[ExecutorConfig] = None
    ) -> None:
        self._exchange = exchange_client
        self._pricing = pricing_engine
        self._inventory = inventory_tracker
        self.circuit_breaker = circuit_breaker
        self._cfg = config or ExecutorConfig()
        self._risk_filter = ExecutionRiskFilter(self._cfg)
        self.replay_protection = ReplayProtection()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def execute(self, signal: Signal) -> ExecutionContext:
        ctx = ExecutionContext(signal=signal)

        if self.circuit_breaker.is_open():
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "circuit_breaker_open"
            ctx.error_code = "CB_OPEN"
            ctx.finished_at = _d(time.time())
            return ctx

        if self.replay_protection.is_duplicate(signal):
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "duplicate_signal"
            ctx.error_code = "REPLAY"
            ctx.finished_at = _d(time.time())
            return ctx

        ctx.transition(ExecutorState.VALIDATING)
        if not signal.is_valid():
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "signal_invalid_or_expired"
            ctx.error_code = "INVALID"
            ctx.finished_at = _d(time.time())
            return ctx

        expected_latency = self._cfg.leg2_timeout / Decimal("2")
        approved, var_alpha = self._risk_filter.approve(signal, expected_latency)
        ctx.var_alpha = var_alpha
        if not approved:
            ctx.transition(ExecutorState.FAILED)
            ctx.error = (f"var_exceeds_pnl: VaR={float(var_alpha):.4f} "
                         f"> pnl={float(signal.expected_net_pnl):.4f}")
            ctx.error_code = "VAR_REJECTED"
            ctx.finished_at = _d(time.time())
            return ctx

        if self._cfg.use_dex_first:
            ctx = await self._execute_dex_first(ctx)
        else:
            ctx = await self._execute_cex_first(ctx)

        self.replay_protection.mark_executed(signal)
        if ctx.state == ExecutorState.DONE:
            self.circuit_breaker.record_success()
        else:
            self.circuit_breaker.record_failure()

        ctx.finished_at = _d(time.time())
        return ctx

    # ------------------------------------------------------------------
    # CEX-first
    # ------------------------------------------------------------------

    async def _execute_cex_first(self, ctx: ExecutionContext) -> ExecutionContext:
        sig = ctx.signal

        ctx.transition(ExecutorState.LEG1_PENDING)
        ctx.leg1_venue = "cex"
        try:
            leg1 = await asyncio.wait_for(
                self._execute_cex_leg(sig, sig.kelly_size),
                timeout=float(self._cfg.leg1_timeout)
            )
        except asyncio.TimeoutError:
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "leg1_cex_timeout"
            ctx.error_code = "L1_TIMEOUT"
            return ctx

        if not leg1["success"]:
            ctx.transition(ExecutorState.FAILED)
            ctx.error = f"leg1_rejected: {leg1.get('error', 'unknown')}"
            ctx.error_code = "L1_REJECT"
            return ctx

        phi = _d(leg1["filled"]) / max(_d(sig.kelly_size), _EPS)
        ctx.fill_quality = phi

        if phi < self._cfg.min_fill_ratio:
            abort_cost = self._estimate_abort_cost(_d(leg1["filled"]), sig)
            reduced_pnl = sig.expected_net_pnl * phi
            if reduced_pnl <= abort_cost:
                ctx.transition(ExecutorState.FAILED)
                ctx.error = f"fill_quality_below_threshold: phi={float(phi):.3f}"
                ctx.error_code = "PARTIAL_FILL"
                return ctx

        ctx.leg1_fill_price = _d(leg1["price"])
        ctx.leg1_fill_size = _d(leg1["filled"])
        ctx.leg1_slippage_bps = abs(ctx.leg1_fill_price - sig.cex_price) / sig.cex_price * Decimal("10000")
        ctx.transition(ExecutorState.LEG1_FILLED)

        ctx.transition(ExecutorState.LEG2_PENDING)
        ctx.leg2_venue = "dex"
        try:
            leg2 = await asyncio.wait_for(
                self._execute_dex_leg(sig, ctx.leg1_fill_size),
                timeout=float(self._cfg.leg2_timeout)
            )
        except asyncio.TimeoutError:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "leg2_dex_timeout_unwound"
            ctx.error_code = "L2_TIMEOUT"
            return ctx

        if not leg2["success"]:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED)
            ctx.error = f"leg2_dex_failed_unwound: {leg2.get('error')}"
            ctx.error_code = "L2_FAIL"
            return ctx

        ctx.leg2_fill_price = _d(leg2["price"])
        ctx.leg2_fill_size = _d(leg2["filled"])
        ctx.leg2_slippage_bps = abs(ctx.leg2_fill_price - sig.dex_price) / sig.dex_price * Decimal("10000")
        ctx.actual_net_pnl = self._calculate_pnl(ctx)
        ctx.transition(ExecutorState.DONE)
        return ctx

    # ------------------------------------------------------------------
    # DEX-first
    # ------------------------------------------------------------------

    async def _execute_dex_first(self, ctx: ExecutionContext) -> ExecutionContext:
        sig = ctx.signal

        ctx.transition(ExecutorState.LEG1_PENDING)
        ctx.leg1_venue = "dex"
        try:
            leg1 = await asyncio.wait_for(
                self._execute_dex_leg(sig, sig.kelly_size),
                timeout=float(self._cfg.leg2_timeout)
            )
        except asyncio.TimeoutError:
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "leg1_dex_timeout"
            ctx.error_code = "L1_TIMEOUT"
            return ctx

        if not leg1["success"]:
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "leg1_dex_failed_no_cost"
            ctx.error_code = "L1_FAIL"
            return ctx

        phi = _d(leg1["filled"]) / max(sig.kelly_size, _EPS)
        ctx.fill_quality = phi
        ctx.leg1_fill_price = _d(leg1["price"])
        ctx.leg1_fill_size = _d(leg1["filled"])
        ctx.leg1_slippage_bps = abs(ctx.leg1_fill_price - sig.dex_price) / sig.dex_price * Decimal("10000")
        ctx.transition(ExecutorState.LEG1_FILLED)

        ctx.transition(ExecutorState.LEG2_PENDING)
        ctx.leg2_venue = "cex"
        try:
            leg2 = await asyncio.wait_for(
                self._execute_cex_leg(sig, ctx.leg1_fill_size),
                timeout=float(self._cfg.leg1_timeout)
            )
        except asyncio.TimeoutError:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED)
            ctx.error = "leg2_cex_timeout_after_dex_unwound"
            ctx.error_code = "L2_TIMEOUT"
            return ctx

        if not leg2["success"]:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            ctx.transition(ExecutorState.FAILED)
            ctx.error = f"leg2_cex_failed_after_dex: {leg2.get('error')}"
            ctx.error_code = "L2_FAIL"
            return ctx

        ctx.leg2_fill_price = _d(leg2["price"])
        ctx.leg2_fill_size = _d(leg2["filled"])
        ctx.leg2_slippage_bps = abs(ctx.leg2_fill_price - sig.cex_price) / sig.cex_price * Decimal("10000")
        ctx.actual_net_pnl = self._calculate_pnl(ctx)
        ctx.transition(ExecutorState.DONE)
        return ctx

    # ------------------------------------------------------------------
    # Leg implementations
    # ------------------------------------------------------------------

    async def _execute_cex_leg(self, signal: Signal, size: Decimal) -> dict:
        if self._cfg.simulation_mode:
            await asyncio.sleep(0.05)
            slippage = Decimal("0.0001")
            fill_price = (signal.cex_price * (_ONE + slippage)
                          if signal.direction == Direction.BUY_CEX_SELL_DEX
                          else signal.cex_price * (_ONE - slippage))
            return {"success": True, "price": float(fill_price),
                    "filled": float(size), "order_id": "sim_cex_001"}
        side = "buy" if signal.direction == Direction.BUY_CEX_SELL_DEX else "sell"
        limit = signal.cex_price * Decimal("1.001") if side == "buy" \
            else signal.cex_price * Decimal("0.999")
        result = self._exchange.create_limit_ioc_order(
            symbol=signal.pair, side=side,
            amount=float(size), price=float(limit)
        )
        filled = float(result["amount_filled"])
        avg_price = float(result["avg_fill_price"]) if filled > 0 else float(signal.cex_price)
        return {"success": result["status"] == "filled",
                "price": avg_price, "filled": filled,
                "order_id": result["id"], "error": result.get("status")}

    async def _execute_dex_leg(self, signal: Signal, size: Decimal) -> dict:
        if self._cfg.simulation_mode:
            await asyncio.sleep(0.3)
            slippage = Decimal("0.0002")
            fill_price = (signal.dex_price * (_ONE - slippage)
                          if signal.direction == Direction.BUY_CEX_SELL_DEX
                          else signal.dex_price * (_ONE + slippage))
            return {"success": True, "price": float(fill_price),
                    "filled": float(size), "tx_hash": "0xsim"}
        raise NotImplementedError("Real DEX execution integration in progress")

    async def _unwind(self, ctx: ExecutionContext) -> None:
        if self._cfg.simulation_mode:
            await asyncio.sleep(0.05)
            log.warning("SIMULATED UNWIND for %s", ctx.signal.signal_id)
            return
        raise NotImplementedError("Real unwind not implemented")

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    def _calculate_pnl(self, ctx: ExecutionContext) -> Decimal:
        sig = ctx.signal
        q = min(ctx.leg1_fill_size or _ZERO, ctx.leg2_fill_size or _ZERO)
        if q <= _ZERO:
            return _ZERO
        p1 = ctx.leg1_fill_price or _ZERO
        p2 = ctx.leg2_fill_price or _ZERO
        if sig.direction == Direction.BUY_CEX_SELL_DEX:
            gross = (p2 - p1) * q
        else:
            gross = (p1 - p2) * q
        fees = q * p1 * _FEE_FRACTION
        ctx.leg_gap_pnl = gross
        return gross - fees

    def _estimate_abort_cost(self, filled: Decimal, signal: Signal) -> Decimal:
        impact_bps = Decimal("10")
        return filled * signal.cex_price * impact_bps / Decimal("10000")
