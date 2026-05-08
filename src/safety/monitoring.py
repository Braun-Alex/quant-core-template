"""
Monitoring, alerting, and observability.

Components:
  BotMonitor        - collects BotHealth + TradeMetrics, logs them structured
  BalanceVerifier   - compares expected vs actual balances; stops on mismatch
  configure_logging - sets up rotating file + stdout logging
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Health dataclass
# ---------------------------------------------------------------------------

@dataclass
class BotHealth:
    """Snapshot of the bot's operational health. Logged every minute."""
    timestamp: float = field(default_factory=time.time)
    is_running: bool = False
    last_heartbeat: float = 0.0
    last_trade_time: float = 0.0

    cex_last_response_ms: float = 0.0
    dex_last_response_ms: float = 0.0

    current_capital: float = 0.0
    daily_pnl: float = 0.0
    drawdown_pct: float = 0.0

    error_count_1h: int = 0
    trades_count_1h: int = 0
    circuit_breaker_open: bool = False
    kill_switch_active: bool = False

    def log_health(self) -> None:
        log.info(
            "BOT_HEALTH | running=%s capital=$%.2f daily_pnl=$%.2f "
            "drawdown=%.1f%% errors_1h=%d trades_1h=%d "
            "cb_open=%s kill=%s",
            self.is_running, self.current_capital, self.daily_pnl,
            self.drawdown_pct * 100,
            self.error_count_1h, self.trades_count_1h,
            self.circuit_breaker_open, self.kill_switch_active
        )


@dataclass
class TradeMetrics:
    """Per-trade execution quality metrics. Logged after every settlement."""
    signal_id: str
    pair: str
    direction: str

    expected_spread_bps: float = 0.0
    actual_spread_bps: float = 0.0
    leg1_slippage_bps: float = 0.0
    leg2_slippage_bps: float = 0.0

    signal_to_fill_ms: float = 0.0
    leg1_time_ms: float = 0.0
    leg2_time_ms: float = 0.0

    gross_pnl: float = 0.0
    fees_paid: float = 0.0
    net_pnl: float = 0.0
    state: str = "UNKNOWN"

    def log_trade(self) -> None:
        log.info(
            "TRADE | id=%s pair=%s dir=%s spread_exp=%.1fbps spread_act=%.1fbps "
            "slip_l1=%.1fbps slip_l2=%.1fbps ttf=%.0fms "
            "pnl=$%.4f fees=$%.4f net=$%.4f state=%s",
            self.signal_id, self.pair, self.direction,
            self.expected_spread_bps, self.actual_spread_bps,
            self.leg1_slippage_bps, self.leg2_slippage_bps,
            self.signal_to_fill_ms,
            self.gross_pnl, self.fees_paid, self.net_pnl,
            self.state
        )


# ---------------------------------------------------------------------------
# Balance verifier
# ---------------------------------------------------------------------------

class BalanceVerifier:
    """
    Compares expected (inventory-tracked) balances against actual on-chain
    and CEX balances after every trade settlement.

    A mismatch above the tolerance triggers an immediate bot stop.
    """

    def __init__(
        self,
        exchange_client,
        chain_client,
        wallet_address: str,
        inventory_tracker,
        tolerance: float = 0.001,
        stop_callback=None,
        alert_callback=None   # Async callable(msg: str)
    ) -> None:
        self._exchange = exchange_client
        self._chain = chain_client
        self._wallet = wallet_address
        self._inventory = inventory_tracker
        self._tolerance = tolerance
        self._stop = stop_callback
        self._alert = alert_callback

    async def verify(self, assets: list[str] = None) -> bool:
        assets = assets or ["ETH", "USDC", "USDT"]
        mismatches: list[str] = []

        try:
            cex_actual = self._exchange.fetch_balance()
        except Exception as exc:
            log.error("Balance verify: could not fetch CEX balance: %s", exc)
            return False

        for asset in assets:
            actual_free = float(
                cex_actual.get(asset, {}).get("free", Decimal("0"))
            )
            try:
                from src.inventory.tracker import Venue
                expected = float(self._inventory.available(Venue.BINANCE, asset))
            except Exception:
                continue

            diff = abs(actual_free - expected)
            if expected > 0 and diff / max(expected, 1e-9) > self._tolerance:
                mismatches.append(
                    f"CEX {asset}: expected={expected:.6f} actual={actual_free:.6f} "
                    f"diff={diff:.6f}"
                )

        try:
            await self._check_onchain_balances(assets, mismatches)
        except Exception as exc:
            log.warning("Balance verify: on-chain check failed: %s", exc)

        if mismatches:
            msg = "BALANCE_MISMATCH | " + " | ".join(mismatches)
            log.critical(msg)
            if self._alert:
                try:
                    await self._alert(f"🚨 {msg}")
                except Exception:
                    pass
            if self._stop:
                self._stop()
            return False

        log.debug("Balance verification OK | assets=%s", assets)
        return True

    async def _check_onchain_balances(
        self, assets: list[str], mismatches: list[str]
    ) -> None:
        from src.core.types import Address
        from src.inventory.tracker import Venue

        for asset in assets:
            try:
                expected = float(self._inventory.available(Venue.WALLET, asset))
            except Exception:
                continue
            if expected <= 0:
                continue

            if asset in ("ETH", "WETH"):
                try:
                    bal = self._chain.get_balance(Address.from_string(self._wallet))
                    actual = float(bal.human)
                    diff = abs(actual - expected)
                    if expected > 0 and diff / max(expected, 1e-9) > self._tolerance:
                        mismatches.append(
                            f"WALLET {asset}: expected={expected:.6f} "
                            f"actual={actual:.6f} diff={diff:.6f}"
                        )
                except Exception as exc:
                    log.debug("On-chain balance check for %s failed: %s", asset, exc)


# ---------------------------------------------------------------------------
# Bot monitor
# ---------------------------------------------------------------------------

class BotMonitor:
    """
    Collects and logs health metrics every minute.
    Sends alerts via the supplied alerter.
    """

    def __init__(
        self,
        risk_manager,
        circuit_breaker,
        kill_switch,
        dead_man_switch=None,
        alerter=None,
        health_interval_seconds: float = 60.0
    ) -> None:
        self._risk = risk_manager
        self._cb = circuit_breaker
        self._kill = kill_switch
        self._dms = dead_man_switch
        self._alerter = alerter
        self._interval = health_interval_seconds
        self._start_time = time.time()

    async def run_health_loop(self) -> None:
        heartbeat_interval = min(30.0, self._interval / 2)
        last_health_log = 0.0

        while True:
            now = time.time()

            if self._dms:
                self._dms.write_heartbeat()

            if now - last_health_log >= self._interval:
                health = self._build_health()
                health.log_health()
                last_health_log = now

                if self._alerter:
                    if health.circuit_breaker_open:
                        await self._alerter.warning("Circuit breaker is OPEN")
                    if health.kill_switch_active:
                        await self._alerter.critical("Kill switch is ACTIVE")
                    if health.error_count_1h > 10:
                        await self._alerter.warning(
                            f"High error rate: {health.error_count_1h} errors/hr"
                        )

            await asyncio.sleep(heartbeat_interval)

    def _build_health(self) -> BotHealth:
        return BotHealth(
            is_running=True,
            last_heartbeat=time.time(),
            current_capital=self._risk.current_capital,
            daily_pnl=self._risk.daily_pnl,
            drawdown_pct=self._risk.drawdown_pct,
            error_count_1h=self._risk.errors_this_hour,
            trades_count_1h=self._risk.trades_this_hour,
            circuit_breaker_open=self._cb.is_open(),
            kill_switch_active=self._kill.is_active()
        )

    def log_trade_metrics(self, ctx) -> None:
        try:
            sig = ctx.signal
            m = TradeMetrics(
                signal_id=sig.signal_id,
                pair=sig.pair,
                direction=sig.direction.value,
                expected_spread_bps=float(sig.raw_spread_bps),
                actual_spread_bps=float(
                    (ctx.leg1_slippage_bps or Decimal("0"))
                    + (ctx.leg2_slippage_bps or Decimal("0"))
                ),
                leg1_slippage_bps=float(ctx.leg1_slippage_bps or 0),
                leg2_slippage_bps=float(ctx.leg2_slippage_bps or 0),
                signal_to_fill_ms=float(ctx.duration()) * 1000,
                gross_pnl=float(ctx.leg_gap_pnl or 0),
                fees_paid=float(
                    (ctx.actual_net_pnl or Decimal("0"))
                    - (ctx.leg_gap_pnl or Decimal("0"))
                ),
                net_pnl=float(ctx.actual_net_pnl or 0),
                state=ctx.state.name
            )
            m.log_trade()
        except Exception as exc:
            log.warning("Could not log trade metrics: %s", exc)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    os.makedirs(log_dir, exist_ok=True)

    date_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"bot_{date_str}.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = [
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    for noisy in ("web3", "urllib3", "asyncio", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("arb_bot").setLevel(level)
    log.info(
        "Logging initialized | file=%s level=%s",
        log_file, logging.getLevelName(level)
    )
