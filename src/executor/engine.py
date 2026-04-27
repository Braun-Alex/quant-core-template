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
    leg1_tx_hash: Optional[str] = None

    leg2_venue: str = ""
    leg2_tx_hash: Optional[str] = None
    leg2_fill_price: Optional[Decimal] = None
    leg2_fill_size: Optional[Decimal] = None
    leg2_slippage_bps: Optional[Decimal] = None

    # Unwind tracking
    unwind_tx_hash: Optional[str] = None
    unwind_attempted: bool = False
    unwind_succeeded: bool = False

    fill_quality: Optional[Decimal] = None
    var_alpha: Optional[Decimal] = None
    leg_gap_pnl: Optional[Decimal] = None

    started_at: Decimal = field(default_factory=lambda: _d(time.time()))
    finished_at: Optional[Decimal] = None
    actual_net_pnl: Optional[Decimal] = None
    error: Optional[str] = None
    error_code: Optional[str] = None

    state_history: list[tuple[ExecutorState, Decimal]] = field(default_factory=list)

    # Raw built transactions (for inspection / logging in dry-run mode)
    leg1_raw_tx: Optional[dict] = None
    leg2_raw_tx: Optional[dict] = None
    unwind_raw_tx: Optional[dict] = None

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
    """All numeric executor parameters."""
    leg1_timeout: Decimal = Decimal("5")
    leg2_timeout: Decimal = Decimal("60")
    min_fill_ratio: Decimal = Decimal("0.80")
    var_confidence: Decimal = Decimal("0.95")
    vol_per_sqrt_second: Decimal = Decimal("0.0002")
    use_dex_first: bool = True
    simulation_mode: bool = True
    max_recovery_attempts: int = 2
    # Unwind knobs
    unwind_timeout: Decimal = Decimal("30")
    unwind_slippage_extra_bps: Decimal = Decimal("100")   # 1 % extra for unwind fills
    abort_cost_impact_bps: Decimal = Decimal("10")   # Assumed market-impact cost to abort

    def __post_init__(self) -> None:
        for f in (
            "leg1_timeout", "leg2_timeout", "min_fill_ratio",
            "var_confidence", "vol_per_sqrt_second",
            "unwind_timeout", "unwind_slippage_extra_bps",
            "abort_cost_impact_bps"
        ):
            setattr(self, f, _d(getattr(self, f)))


# ---------------------------------------------------------------------------
# Risk filter
# ---------------------------------------------------------------------------

class ExecutionRiskFilter:
    """VaR gate."""

    def __init__(self, config: ExecutorConfig) -> None:
        self._cfg = config
        self._z_alpha = _d(self._normal_quantile(float(config.var_confidence)))

    def approve(
        self, signal: Signal, expected_leg2_latency: Decimal
    ) -> tuple[bool, Decimal]:
        q = signal.kelly_size
        tau = max(expected_leg2_latency, _ZERO)
        sigma = self._cfg.vol_per_sqrt_second * _d(math.sqrt(float(tau)))
        var_alpha = q * signal.cex_price * sigma * self._z_alpha
        approved = var_alpha < signal.expected_net_pnl
        log.debug(
            "RiskFilter VaR=%.6f pnl=%.6f ok=%s",
            float(var_alpha), float(signal.expected_net_pnl), approved
        )
        return approved, var_alpha

    @staticmethod
    def _normal_quantile(p: float) -> float:
        import math
        if p <= 0.0 or p >= 1.0:
            raise ValueError(f"p must be in (0, 1), got {p}")
        sign = 1.0 if p >= 0.5 else -1.0
        q = p if p >= 0.5 else 1.0 - p
        t = math.sqrt(-2.0 * math.log(1.0 - q))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        z = t - (c[0] + c[1] * t + c[2] * t**2) / (
            1.0 + d[0] * t + d[1] * t**2 + d[2] * t**3
        )
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
# Fee constant
# ---------------------------------------------------------------------------

_FEE_FRACTION = Decimal("0.004")   # ~40 bps total round-trip


# ---------------------------------------------------------------------------
# PFA Executor
# ---------------------------------------------------------------------------

class Executor:
    """
    Probabilistic Finite Automaton Executor.
    """

    def __init__(
        self,
        exchange_client,   # BinanceClient or stub
        pricing_engine,   # PricingEngine or None
        inventory_tracker,
        circuit_breaker,
        config: Optional[ExecutorConfig] = None,
        dex_price_source=None,   # DEXPriceSource instance (optional)
        dex_executor=None   # DEXExecutor instance (optional)
    ) -> None:
        self._exchange = exchange_client
        self._pricing = pricing_engine
        self._inventory = inventory_tracker
        self.circuit_breaker = circuit_breaker
        self._cfg = config or ExecutorConfig()
        self._risk_filter = ExecutionRiskFilter(self._cfg)
        self.replay_protection = ReplayProtection()
        self._dex_price = dex_price_source
        self._dex_exec = dex_executor

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def execute(self, signal: Signal) -> ExecutionContext:
        ctx = ExecutionContext(signal=signal)

        if self.circuit_breaker.is_open():
            return self._fail(ctx, "circuit_breaker_open", "CB_OPEN")

        if self.replay_protection.is_duplicate(signal):
            return self._fail(ctx, "duplicate_signal", "REPLAY")

        ctx.transition(ExecutorState.VALIDATING)
        if not signal.is_valid():
            return self._fail(ctx, "signal_invalid_or_expired", "INVALID")

        expected_latency = self._cfg.leg2_timeout / Decimal("2")
        approved, var_alpha = self._risk_filter.approve(signal, expected_latency)
        ctx.var_alpha = var_alpha
        if not approved:
            return self._fail(
                ctx,
                f"var_exceeds_pnl: VaR={float(var_alpha):.4f} "
                f"> pnl={float(signal.expected_net_pnl):.4f}",
                "VAR_REJECTED"
            )

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
    # CEX-first flow
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
            return self._fail(ctx, "leg1_cex_timeout", "L1_TIMEOUT")

        if not leg1["success"]:
            return self._fail(ctx, f"leg1_rejected: {leg1.get('error', 'unknown')}", "L1_REJECT")

        phi = _d(leg1["filled"]) / max(_d(sig.kelly_size), _EPS)
        ctx.fill_quality = phi

        if phi < self._cfg.min_fill_ratio:
            abort_cost = self._estimate_abort_cost(_d(leg1["filled"]), sig)
            reduced_pnl = sig.expected_net_pnl * phi
            if reduced_pnl <= abort_cost:
                return self._fail(
                    ctx,
                    f"fill_quality_below_threshold: phi={float(phi):.3f}",
                    "PARTIAL_FILL"
                )

        ctx.leg1_fill_price = _d(leg1["price"])
        ctx.leg1_fill_size = _d(leg1["filled"])
        ctx.leg1_slippage_bps = (
            abs(ctx.leg1_fill_price - sig.cex_price) / sig.cex_price * Decimal("10000")
        )
        ctx.transition(ExecutorState.LEG1_FILLED)

        # --- Leg 2: DEX ---
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
            return self._fail(ctx, "leg2_dex_timeout_unwound", "L2_TIMEOUT")

        if not leg2["success"]:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            return self._fail(ctx, f"leg2_dex_failed_unwound: {leg2.get('error')}", "L2_FAIL")

        ctx.leg2_raw_tx = leg2.get("swap_tx")
        ctx.leg2_tx_hash = leg2.get("tx_hash")
        ctx.leg2_fill_price = _d(leg2["price"])
        ctx.leg2_fill_size = _d(leg2["filled"])
        ctx.leg2_slippage_bps = (
            abs(ctx.leg2_fill_price - sig.dex_price) / sig.dex_price * Decimal("10000")
        )
        ctx.actual_net_pnl = self._calculate_pnl(ctx)
        ctx.transition(ExecutorState.DONE)
        return ctx

    # ------------------------------------------------------------------
    # DEX-first flow
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
            return self._fail(ctx, "leg1_dex_timeout", "L1_TIMEOUT")

        if not leg1["success"]:
            return self._fail(ctx, "leg1_dex_failed_no_cost", "L1_FAIL")

        phi = _d(leg1["filled"]) / max(sig.kelly_size, _EPS)
        ctx.fill_quality = phi
        ctx.leg1_raw_tx = leg1.get("swap_tx")
        ctx.leg1_tx_hash = leg1.get("tx_hash")
        ctx.leg1_fill_price = _d(leg1["price"])
        ctx.leg1_fill_size = _d(leg1["filled"])
        ctx.leg1_slippage_bps = (
            abs(ctx.leg1_fill_price - sig.dex_price) / sig.dex_price * Decimal("10000")
        )
        ctx.transition(ExecutorState.LEG1_FILLED)

        # --- Leg 2: CEX ---
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
            return self._fail(ctx, "leg2_cex_timeout_after_dex_unwound", "L2_TIMEOUT")

        if not leg2["success"]:
            ctx.transition(ExecutorState.UNWINDING)
            await self._unwind(ctx)
            return self._fail(ctx, f"leg2_cex_failed_after_dex: {leg2.get('error')}", "L2_FAIL")

        ctx.leg2_fill_price = _d(leg2["price"])
        ctx.leg2_fill_size = _d(leg2["filled"])
        ctx.leg2_slippage_bps = (
            abs(ctx.leg2_fill_price - sig.cex_price) / sig.cex_price * Decimal("10000")
        )
        ctx.actual_net_pnl = self._calculate_pnl(ctx)
        ctx.transition(ExecutorState.DONE)
        return ctx

    # ------------------------------------------------------------------
    # Leg implementations
    # ------------------------------------------------------------------

    async def _execute_cex_leg(self, signal: Signal, size: Decimal) -> dict:
        """Execute the CEX leg."""
        if self._cfg.simulation_mode or self._exchange is None:
            await asyncio.sleep(0.05)
            slippage = Decimal("0.0001")
            fill_price = (
                signal.cex_price * (_ONE + slippage)
                if signal.direction == Direction.BUY_CEX_SELL_DEX
                else signal.cex_price * (_ONE - slippage)
            )
            return {
                "success": True,
                "price": float(fill_price),
                "filled": float(size),
                "order_id": "sim_cex_001"
            }

        side = "buy" if signal.direction == Direction.BUY_CEX_SELL_DEX else "sell"
        limit = (
            signal.cex_price * Decimal("1.001")
            if side == "buy"
            else signal.cex_price * Decimal("0.999")
        )
        try:
            result = self._exchange.create_limit_ioc_order(
                symbol=signal.pair,
                side=side,
                amount=float(size),
                price=float(limit)
            )
            filled = float(result["amount_filled"])
            avg_price = float(result["avg_fill_price"]) if filled > 0 else float(signal.cex_price)
            return {
                "success": result["status"] == "filled",
                "price": avg_price,
                "filled": filled,
                "order_id": result["id"],
                "error": result.get("status")
            }
        except Exception as exc:
            log.error("CEX leg failed: %s", exc)
            return {"success": False, "price": 0.0, "filled": 0.0, "error": str(exc)}

    async def _execute_dex_leg(self, signal: Signal, size: Decimal) -> dict:
        """
        Execute the DEX leg.
        """
        # ── Real DEX execution ──────────────────────────────────────────
        if self._dex_exec is not None and self._dex_price is not None:
            try:
                return await self._execute_dex_real(signal, size)
            except Exception as exc:
                log.error("Real DEX leg failed: %s", exc)
                return {"success": False, "price": 0.0, "filled": 0.0, "error": str(exc)}

        # ── Simulation fallback ─────────────────────────────────────────
        if self._cfg.simulation_mode:
            await asyncio.sleep(0.3)
            slippage = Decimal("0.0002")
            fill_price = (
                signal.dex_price * (_ONE - slippage)
                if signal.direction == Direction.BUY_CEX_SELL_DEX
                else signal.dex_price * (_ONE + slippage)
            )
            return {
                "success": True,
                "price": float(fill_price),
                "filled": float(size),
                "tx_hash": "0xsim"
            }

        raise NotImplementedError(
            "Real DEX execution requires dex_executor and dex_price_source to be provided."
        )

    async def _execute_dex_real(self, signal: Signal, size: Decimal) -> dict:
        """Build & execute a real Uniswap V2 swap."""
        # Determine token direction
        base_sym, quote_sym = signal.pair.split("/")
        if signal.direction == Direction.BUY_CEX_SELL_DEX:
            # Selling base on DEX → token_in=base, token_out=quote
            selling_sym, buying_sym = base_sym, quote_sym
        else:
            # Buying base on DEX → token_in=quote, token_out=base
            selling_sym, buying_sym = quote_sym, base_sym

        # Find token objects from the pricing engine's pool registry
        token_in, token_out = self._resolve_tokens(selling_sym, buying_sym)
        if token_in is None or token_out is None:
            # Fall back to simulation if tokens not registered
            log.warning(
                "Token resolution failed for %s/%s - using simulation fallback",
                selling_sym, buying_sym
            )
            await asyncio.sleep(0.1)
            slippage = Decimal("0.0002")
            fill_price = (
                signal.dex_price * (_ONE - slippage)
                if signal.direction == Direction.BUY_CEX_SELL_DEX
                else signal.dex_price * (_ONE + slippage)
            )
            return {
                "success": True,
                "price": float(fill_price),
                "filled": float(size),
                "tx_hash": "0xsim_token_fallback",
                "swap_tx": None
            }

        amount_in_raw = int(size * Decimal(10 ** token_in.decimals))
        quote = self._dex_price.get_full_quote(token_in, token_out, amount_in_raw)
        if quote is None:
            return {
                "success": False,
                "price": 0.0,
                "filled": 0.0,
                "error": "DEX quote unavailable"
            }

        # Execute (or dry-run) via DEXExecutor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._dex_exec.execute_swap(quote)
        )
        return result

    # ------------------------------------------------------------------
    # Unwind implementation
    # ------------------------------------------------------------------

    async def _unwind(self, ctx: ExecutionContext) -> None:
        """
        Unwind strategy after a failed leg.

        Leg1 venue | Action
        -----------+--------------------------------------------------
        dex        | Place a reverse swap on DEX (sell back what we bought)
        cex        | Place a reverse limit-IOC order on CEX
        """
        sig = ctx.signal
        size_to_unwind = ctx.leg1_fill_size or _ZERO

        if size_to_unwind <= _ZERO:
            log.warning("Unwind skipped - no leg1 fill size recorded")
            return

        ctx.unwind_attempted = True
        log.warning(
            "UNWIND triggered | signal=%s leg1_venue=%s size=%s",
            sig.signal_id, ctx.leg1_venue, size_to_unwind
        )

        try:
            if ctx.leg1_venue == "dex":
                await self._unwind_dex(ctx, sig, size_to_unwind)
            elif ctx.leg1_venue == "cex":
                await self._unwind_cex(ctx, sig, size_to_unwind)
            else:
                log.warning("Unknown leg1 venue for unwind: %s", ctx.leg1_venue)
        except Exception as exc:
            log.error("Unwind execution error: %s", exc)

    async def _unwind_dex(
        self, ctx: ExecutionContext, sig: Signal, size: Decimal
    ) -> None:
        """Reverse the DEX leg: sell back the token we bought."""
        if self._cfg.simulation_mode or self._dex_exec is None:
            await asyncio.sleep(0.05)
            log.warning("[SIM] DEX unwind — %s (NOT broadcast)", sig.signal_id)
            ctx.unwind_succeeded = True
            ctx.unwind_tx_hash = "0xunwind_sim"
            return

        try:
            base_sym, quote_sym = sig.pair.split("/")
            # If DEX-first bought base with quote → unwind sells base for quote
            if sig.direction == Direction.BUY_DEX_SELL_CEX:
                selling_sym, buying_sym = base_sym, quote_sym
            else:
                selling_sym, buying_sym = quote_sym, base_sym

            token_in, token_out = self._resolve_tokens(selling_sym, buying_sym)
            if token_in is None or token_out is None:
                log.error("Unwind: cannot resolve tokens %s/%s", selling_sym, buying_sym)
                return

            amount_in_raw = int(size * Decimal(10 ** token_in.decimals))
            loop = asyncio.get_event_loop()
            unwind_tx = await loop.run_in_executor(
                None,
                lambda: self._dex_exec.build_unwind_tx(
                    token_in, token_out, amount_in_raw
                )
            )
            ctx.unwind_raw_tx = unwind_tx

            if not self._dex_exec._dry_run:
                tx_hash = await loop.run_in_executor(
                    None, lambda: self._dex_exec._broadcast(unwind_tx)
                )
                ctx.unwind_tx_hash = tx_hash
                log.warning("DEX unwind broadcast: %s", tx_hash)
            else:
                ctx.unwind_tx_hash = "0x" + "d" * 64
                log.warning("[DRY-RUN] DEX unwind tx built but not broadcast")

            ctx.unwind_succeeded = True

        except Exception as exc:
            log.error("DEX unwind failed: %s", exc)

    async def _unwind_cex(
        self, ctx: ExecutionContext, sig: Signal, size: Decimal
    ) -> None:
        """Reverse the CEX leg: place opposite order on CEX."""
        if self._cfg.simulation_mode or self._exchange is None:
            await asyncio.sleep(0.05)
            log.warning("[SIM] CEX unwind - %s (NOT broadcast)", sig.signal_id)
            ctx.unwind_succeeded = True
            ctx.unwind_tx_hash = "0xunwind_cex_sim"
            return

        # Opposite side
        reverse_side = (
            "sell"
            if sig.direction == Direction.BUY_CEX_SELL_DEX
            else "buy"
        )
        # Use current fill price ± extra slippage for unwind limit
        fill_price = ctx.leg1_fill_price or sig.cex_price
        extra_bps = self._cfg.unwind_slippage_extra_bps / Decimal("10000")
        if reverse_side == "sell":
            limit_price = fill_price * (_ONE - extra_bps)
        else:
            limit_price = fill_price * (_ONE + extra_bps)

        try:
            result = self._exchange.create_limit_ioc_order(
                symbol=sig.pair,
                side=reverse_side,
                amount=float(size),
                price=float(limit_price)
            )
            ctx.unwind_succeeded = result.get("status") in ("filled", "partially_filled")
            ctx.unwind_tx_hash = result.get("id", "unknown")
            log.warning(
                "CEX unwind %s  status=%s  id=%s",
                reverse_side, result.get("status"), result.get("id"),
            )
        except Exception as exc:
            log.error("CEX unwind order failed: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_tokens(self, sym_in: str, sym_out: str):
        """Look up Token objects from the pricing engine's pool registry."""
        if self._pricing is None:
            return None, None
        for pool in self._pricing._pools.values():
            syms = {pool.left.symbol.upper(), pool.right.symbol.upper()}
            if {sym_in.upper(), sym_out.upper()} <= syms:
                t_in = pool.left if pool.left.symbol.upper() == sym_in.upper() else pool.right
                t_out = pool.right if t_in == pool.left else pool.left
                return t_in, t_out
        return None, None

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
        return (
            filled
            * signal.cex_price
            * self._cfg.abort_cost_impact_bps
            / Decimal("10000")
        )

    @staticmethod
    def _fail(ctx: ExecutionContext, error: str, code: str) -> ExecutionContext:
        ctx.transition(ExecutorState.FAILED)
        ctx.error = error
        ctx.error_code = code
        ctx.finished_at = _d(time.time())
        return ctx
