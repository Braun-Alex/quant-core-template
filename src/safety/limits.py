"""
Risk management: RiskLimits and RiskManager.

RiskLimits  - configurable soft limits (can be tightened, never loosened past hardcoded ceilings).
RiskManager - stateful enforcer; call check_pre_trade() before every execution and
              record_trade() after every settlement.

Absolute hard ceilings are imported from killswitch.py and are checked last - they
cannot be overridden by configuration.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configurable risk limits (soft; overridable via env / config, but capped by
# the absolute constants in killswitch.py)
# ---------------------------------------------------------------------------


@dataclass
class RiskLimits:
    """
    Soft limits for a single trading session.

    All monetary values are in USD.
    These can be set conservatively at bot startup; the hard ceilings
    in killswitch.ABSOLUTE_* always take precedence.
    """

    # ── Per-trade limits ───────────────────────────────────────────────────
    max_trade_usd: float = 20.0   # Never trade more than $20 per leg
    max_trade_pct: float = 0.20   # Never trade more than 20% of capital

    # ── Position limits ────────────────────────────────────────────────────
    max_position_per_token: float = 30.0   # Never hold >$30 of any single token
    max_open_positions: int = 1   # Only one open arb at a time

    # ── Loss limits ────────────────────────────────────────────────────────
    max_loss_per_trade: float = 5.0   # Per-trade stop-loss
    max_daily_loss: float = 15.0   # Stop trading after $15 daily loss
    max_drawdown_pct: float = 0.20   # Stop at 20% drawdown from equity peak

    # ── Frequency limits ───────────────────────────────────────────────────
    max_trades_per_hour: int = 20   # Prevent runaway execution loops
    consecutive_loss_limit: int = 3   # Pause after 3 losses in a row

    # ── Spread sanity ──────────────────────────────────────────────────────
    max_spread_bps: float = 500.0   # >500 bps is almost certainly bad data
    max_signal_age_seconds: float = 5.0   # Reject stale signals

    # ── Slippage tolerance ─────────────────────────────────────────────────
    max_slippage_bps: float = 100.0   # Abort if actual slippage exceeds this


# ---------------------------------------------------------------------------
# Trade record (kept in memory for the session)
# ---------------------------------------------------------------------------

@dataclass
class TradeRecord:
    """Lightweight record written after every settled trade."""
    timestamp: float
    pair: str
    direction: str
    size_usd: float
    gross_pnl: float
    fees: float
    net_pnl: float
    spread_bps: float
    signal_age_s: float
    state: str   # "DONE" | "FAILED" | "UNWOUND"


# ---------------------------------------------------------------------------
# RiskManager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Stateful risk enforcer.

    Instantiate once at bot startup and share across ticks.
    All public methods are thread-safe for single-event-loop usage
    (asyncio and single thread).
    """

    def __init__(self, limits: RiskLimits, initial_capital: float) -> None:
        self.limits = limits
        self.initial_capital = float(initial_capital)
        self.peak_capital = float(initial_capital)
        self.current_capital = float(initial_capital)

        # Session counters
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.consecutive_losses: int = 0
        self.open_positions: int = 0

        # Rolling 1-hour trade window
        self._trade_times: deque[float] = deque()

        # Full trade history (in-memory; export via daily summary)
        self.trade_history: list[TradeRecord] = []

        # Error counter (reset hourly)
        self._error_times: deque[float] = deque()

        log.info(
            "RiskManager initialized | capital=$%.2f max_trade=$%.2f max_daily_loss=$%.2f",
            initial_capital, limits.max_trade_usd, limits.max_daily_loss
        )

    # ------------------------------------------------------------------
    # Pre-trade gate
    # ------------------------------------------------------------------

    def check_pre_trade(
        self,
        trade_usd: float
    ) -> tuple[bool, str]:
        """
        Run all soft risk checks before allowing a trade.
        Returns (allowed: bool, reason: str).

        The caller should also run killswitch.safety_check() after this.
        """
        self._prune_hour_window()

        # ── Trade size ─────────────────────────────────────────────────
        if trade_usd > self.limits.max_trade_usd:
            return False, (
                f"Trade ${trade_usd:.2f} exceeds max ${self.limits.max_trade_usd:.2f}"
            )

        capital_limit = self.current_capital * self.limits.max_trade_pct
        if trade_usd > capital_limit:
            return False, (
                f"Trade ${trade_usd:.2f} exceeds {self.limits.max_trade_pct:.0%} "
                f"of capital ${self.current_capital:.2f}"
            )

        # ── Open positions ─────────────────────────────────────────────
        if self.open_positions >= self.limits.max_open_positions:
            return False, (
                f"Max open positions ({self.limits.max_open_positions}) already active"
            )

        # ── Daily loss ─────────────────────────────────────────────────
        if self.daily_pnl <= -self.limits.max_daily_loss:
            return False, (
                f"Daily loss limit reached: ${self.daily_pnl:.2f} "
                f"(max ${self.limits.max_daily_loss:.2f})"
            )

        # ── Drawdown ───────────────────────────────────────────────────
        if self.peak_capital > 0:
            drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
            if drawdown >= self.limits.max_drawdown_pct:
                return False, (
                    f"Drawdown {drawdown:.1%} exceeds limit "
                    f"{self.limits.max_drawdown_pct:.1%}"
                )

        # ── Consecutive losses ─────────────────────────────────────────
        if self.consecutive_losses >= self.limits.consecutive_loss_limit:
            return False, (
                f"Consecutive loss limit ({self.consecutive_losses}) reached"
            )

        # ── Hourly frequency ───────────────────────────────────────────
        trades_this_hour = len(self._trade_times)
        if trades_this_hour >= self.limits.max_trades_per_hour:
            return False, (
                f"Hourly trade limit ({self.limits.max_trades_per_hour}) reached "
                f"({trades_this_hour} this hour)"
            )

        return True, "OK"

    # ------------------------------------------------------------------
    # Post-trade accounting
    # ------------------------------------------------------------------

    def open_position(self) -> None:
        """Call immediately before dispatching an execution."""
        self.open_positions = min(
            self.open_positions + 1, self.limits.max_open_positions + 1
        )

    def close_position(self) -> None:
        """Call after the execution settles (success or failure)."""
        self.open_positions = max(self.open_positions - 1, 0)

    def record_trade(
        self,
        net_pnl: float,
        *,
        pair: str = "?",
        direction: str = "?",
        size_usd: float = 0.0,
        gross_pnl: float = 0.0,
        fees: float = 0.0,
        spread_bps: float = 0.0,
        signal_age_s: float = 0.0,
        state: str = "DONE"
    ) -> None:
        """Update all risk state after a trade is settled."""
        self.daily_pnl += net_pnl
        self.total_pnl += net_pnl
        self.current_capital += net_pnl
        self.peak_capital = max(self.peak_capital, self.current_capital)
        self._trade_times.append(time.monotonic())

        if net_pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        record = TradeRecord(
            timestamp=time.time(),
            pair=pair,
            direction=direction,
            size_usd=size_usd,
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            spread_bps=spread_bps,
            signal_age_s=signal_age_s,
            state=state
        )
        self.trade_history.append(record)

        log.info(
            "TRADE_RECORD | pair=%s dir=%s net_pnl=$%.4f capital=$%.2f "
            "daily_pnl=$%.2f consecutive_losses=%d",
            pair, direction, net_pnl, self.current_capital,
            self.daily_pnl, self.consecutive_losses
        )

    def record_error(self) -> None:
        """Call whenever a recoverable error occurs (for hourly error-rate tracking)."""
        self._error_times.append(time.monotonic())

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def reset_daily(self) -> None:
        """Call at the start of each trading day (UTC midnight)."""
        log.info(
            "DAILY_RESET | daily_pnl=$%.2f capital=$%.2f total_trades=%d",
            self.daily_pnl, self.current_capital, len(self.trade_history)
        )
        self.daily_pnl = 0.0
        self.consecutive_losses = 0
        self._trade_times.clear()
        self._error_times.clear()

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @property
    def trades_this_hour(self) -> int:
        self._prune_hour_window()
        return len(self._trade_times)

    @property
    def errors_this_hour(self) -> int:
        cutoff = time.monotonic() - 3600
        while self._error_times and self._error_times[0] < cutoff:
            self._error_times.popleft()
        return len(self._error_times)

    @property
    def drawdown_pct(self) -> float:
        if self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital

    @property
    def win_rate(self) -> Optional[float]:
        if not self.trade_history:
            return None
        wins = sum(1 for t in self.trade_history if t.net_pnl > 0)
        return wins / len(self.trade_history)

    def daily_summary(self) -> dict:
        """Return a snapshot of today's performance."""
        today = [t for t in self.trade_history
                 if time.time() - t.timestamp < 86_400]
        if not today:
            return {"trades": 0, "pnl": 0.0, "win_rate": None}
        wins = sum(1 for t in today if t.net_pnl > 0)
        return {
            "trades": len(today),
            "wins": wins,
            "losses": len(today) - wins,
            "win_rate": wins / len(today),
            "total_pnl": sum(t.net_pnl for t in today),
            "best_trade": max(t.net_pnl for t in today),
            "worst_trade": min(t.net_pnl for t in today),
            "avg_spread_bps": sum(t.spread_bps for t in today) / len(today),
            "capital": self.current_capital,
            "drawdown_pct": self.drawdown_pct
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _prune_hour_window(self) -> None:
        cutoff = time.monotonic() - 3600
        while self._trade_times and self._trade_times[0] < cutoff:
            self._trade_times.popleft()
