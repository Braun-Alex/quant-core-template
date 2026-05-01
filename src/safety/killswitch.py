"""
Kill switches and absolute (non-configurable) safety constants.

Three layers of protection:
  1. ABSOLUTE_* constants   - hardcoded ceilings, never overridable at runtime
  2. safety_check()         - final gate run after all soft checks
  3. ManualKillSwitch       - file-based, operator-activated
  4. AutoKillSwitch         - triggered by capital or error thresholds
  5. DeadManSwitch          - heartbeat-based watchdog (for external cron use)

Usage pattern in the bot loop:
    # 1. Soft risk limits
    allowed, reason = risk_manager.check_pre_trade(trade_usd)
    if not allowed: ...

    # 2. Hard absolute ceiling
    ok, reason = safety_check(trade_usd, risk_manager.daily_pnl,
                              risk_manager.current_capital, risk_manager.trades_this_hour)
    if not ok: ...

    # 3. Kill switches
    if manual_kill.is_active() or auto_kill.check(risk_manager): ...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ABSOLUTE SAFETY CONSTANTS
# These are NOT configurable at runtime and CANNOT be loosened by any env var.
# ---------------------------------------------------------------------------

ABSOLUTE_MAX_TRADE_USD: float = 25.0   # Hard ceiling on any single trade
ABSOLUTE_MAX_DAILY_LOSS: float = 20.0   # Hard ceiling on cumulative daily loss
ABSOLUTE_MIN_CAPITAL: float = 50.0   # Auto-stop if total capital falls below this
ABSOLUTE_MAX_TRADES_PER_HOUR: int = 30   # Prevent runaway execution loops
ABSOLUTE_MAX_SPREAD_BPS: float = 1_000.0   # Any spread above 10% is definitively bad data
ABSOLUTE_MAX_ERRORS_PER_HOUR: int = 50   # Too many errors → something is very wrong

# ---------------------------------------------------------------------------
# Final safety gate
# ---------------------------------------------------------------------------


def safety_check(
    trade_usd: float,
    daily_pnl: float,
    total_capital: float,
    trades_this_hour: int,
    errors_this_hour: int = 0
) -> tuple[bool, str]:
    """
    Absolute final gate - runs AFTER all soft checks.

    This function uses only the ABSOLUTE_* constants above.
    It must never reference configurable limits.

    Returns (allowed: bool, reason: str).
    """
    if trade_usd > ABSOLUTE_MAX_TRADE_USD:
        return False, (
            f"Trade ${trade_usd:.2f} exceeds ABSOLUTE max "
            f"${ABSOLUTE_MAX_TRADE_USD:.2f}"
        )
    if daily_pnl <= -ABSOLUTE_MAX_DAILY_LOSS:
        return False, (
            f"Absolute daily loss limit reached: "
            f"${daily_pnl:.2f} ≤ -${ABSOLUTE_MAX_DAILY_LOSS:.2f}"
        )
    if total_capital < ABSOLUTE_MIN_CAPITAL:
        return False, (
            f"Total capital ${total_capital:.2f} below "
            f"ABSOLUTE minimum ${ABSOLUTE_MIN_CAPITAL:.2f}"
        )
    if trades_this_hour >= ABSOLUTE_MAX_TRADES_PER_HOUR:
        return False, (
            f"ABSOLUTE hourly trade limit {ABSOLUTE_MAX_TRADES_PER_HOUR} reached "
            f"({trades_this_hour} this hour)"
        )
    if errors_this_hour >= ABSOLUTE_MAX_ERRORS_PER_HOUR:
        return False, (
            f"ABSOLUTE hourly error limit {ABSOLUTE_MAX_ERRORS_PER_HOUR} reached "
            f"({errors_this_hour} errors this hour)"
        )
    return True, "OK"


# ---------------------------------------------------------------------------
# Manual kill switch (file-based)
# ---------------------------------------------------------------------------

KILL_SWITCH_FILE: str = os.getenv(
    "KILL_SWITCH_FILE", "/tmp/arb_bot_kill"
)


class ManualKillSwitch:
    """
    File-based kill switch.

    Activate: touch /tmp/arb_bot_kill   (or set KILL_SWITCH_FILE env var)
    Deactivate: rm /tmp/arb_bot_kill

    The bot checks this every tick. The file just needs to exist - its content is ignored.
    """

    def __init__(self, path: str = KILL_SWITCH_FILE) -> None:
        self._path = path

    def is_active(self) -> bool:
        return os.path.exists(self._path)

    def activate(self, reason: str = "manual") -> None:
        """Create the kill-switch file (programmatic activation)."""
        try:
            with open(self._path, "w") as f:
                f.write(f"{time.time():.0f} reason={reason}\n")
            log.critical("MANUAL_KILL_SWITCH ACTIVATED | reason=%s | file=%s", reason, self._path)
        except OSError as exc:
            log.error("Could not write kill-switch file: %s", exc)

    def deactivate(self) -> None:
        """Remove the kill-switch file (programmatic deactivation)."""
        try:
            if os.path.exists(self._path):
                os.remove(self._path)
                log.info("MANUAL_KILL_SWITCH deactivated | file=%s", self._path)
        except OSError as exc:
            log.error("Could not remove kill-switch file: %s", exc)


# ---------------------------------------------------------------------------
# Automatic kill switch (condition-based)
# ---------------------------------------------------------------------------

class AutoKillSwitch:
    """
    Triggers automatically when configurable thresholds are breached.

    Backed by the ManualKillSwitch (writes the same file) so the bot
    stops even if AutoKillSwitch is not polled in the same call path.
    """

    def __init__(
        self,
        manual_switch: Optional[ManualKillSwitch] = None,
        capital_floor_pct: float = 0.50,   # Kill if capital < 50% of start
        max_errors_per_hour: int = ABSOLUTE_MAX_ERRORS_PER_HOUR,
        alert_callback=None   # Async or sync callable(msg: str)
    ) -> None:
        self._manual = manual_switch or ManualKillSwitch()
        self._capital_floor_pct = capital_floor_pct
        self._max_errors = max_errors_per_hour
        self._alert = alert_callback
        self.triggered: bool = False
        self.reason: Optional[str] = None

    def check(self, risk_manager) -> bool:
        """
        Evaluate auto-kill conditions against the current risk state.
        Returns True if the kill switch was (or is already) triggered.
        """
        if self.triggered:
            return True

        # Capital floor
        if (risk_manager.initial_capital > 0 and
                risk_manager.current_capital < risk_manager.initial_capital * self._capital_floor_pct):
            self._trigger(
                f"Capital ${risk_manager.current_capital:.2f} fell below "
                f"{self._capital_floor_pct:.0%} of initial "
                f"${risk_manager.initial_capital:.2f}"
            )
            return True

        # Absolute minimum capital (hardcoded)
        if risk_manager.current_capital < ABSOLUTE_MIN_CAPITAL:
            self._trigger(
                f"Capital ${risk_manager.current_capital:.2f} below "
                f"ABSOLUTE minimum ${ABSOLUTE_MIN_CAPITAL:.2f}"
            )
            return True

        # Error rate
        if risk_manager.errors_this_hour >= self._max_errors:
            self._trigger(
                f"Error rate {risk_manager.errors_this_hour}/hr "
                f"exceeds limit {self._max_errors}/hr"
            )
            return True

        return False

    def _trigger(self, reason: str) -> None:
        self.triggered = True
        self.reason = reason
        log.critical("AUTO_KILL_SWITCH TRIGGERED | reason=%s", reason)
        self._manual.activate(reason=reason)
        if self._alert:
            try:
                self._alert(f"AUTO KILL SWITCH: {reason}")
            except Exception as exc:
                log.error("Alert callback failed: %s", exc)


# ---------------------------------------------------------------------------
# Dead man's switch (heartbeat for external watchdog)
# ---------------------------------------------------------------------------

HEARTBEAT_FILE: str = os.getenv(
    "HEARTBEAT_FILE", "/tmp/arb_bot_heartbeat"
)
HEARTBEAT_INTERVAL_SECONDS: float = 30.0
HEARTBEAT_MAX_AGE_SECONDS: float = 120.0   # 2 × interval - if stale, bot is inactive


class DeadManSwitch:
    """
    Bot-side heartbeat writer.

    Call write_heartbeat() from the async main loop every tick (or a
    dedicated background task) to keep the file fresh.

    A separate watchdog script (cron, systemd timer)
    should call is_bot_alive() periodically and take action when False.
    """

    def __init__(
        self,
        path: str = HEARTBEAT_FILE,
        max_age_seconds: float = HEARTBEAT_MAX_AGE_SECONDS
    ) -> None:
        self._path = path
        self._max_age = max_age_seconds
        self._last_write: float = 0.0

    def write_heartbeat(self) -> None:
        """Write current timestamp to the heartbeat file."""
        try:
            with open(self._path, "w") as f:
                f.write(f"{time.time():.3f}\n")
            self._last_write = time.time()
        except OSError as exc:
            log.error("Could not write heartbeat: %s", exc)

    def is_bot_alive(self) -> bool:
        """
        Read the heartbeat file and check its age.
        For use by an external watchdog - not the bot itself.
        """
        try:
            with open(self._path) as f:
                ts = float(f.read().strip())
            return (time.time() - ts) <= self._max_age
        except (OSError, ValueError):
            return False

    def seconds_since_heartbeat(self) -> Optional[float]:
        """Return age of the last heartbeat, or None if file is missing."""
        try:
            with open(self._path) as f:
                ts = float(f.read().strip())
            return time.time() - ts
        except (OSError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Async background task
    # ------------------------------------------------------------------

    async def run_forever(self) -> None:
        """
        Asyncio background task: writes the heartbeat every
        HEARTBEAT_INTERVAL_SECONDS seconds.

        Usage:
            asyncio.create_task(dead_man_switch.run_forever())
        """
        import asyncio
        while True:
            self.write_heartbeat()
            await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
