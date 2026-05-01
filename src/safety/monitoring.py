"""
Monitoring, alerting, and observability.

Components:
  BotMonitor        - collects BotHealth + TradeMetrics, logs them structured
  TelegramAlerter   - sends critical alerts to a Telegram chat
  BalanceVerifier   - compares expected vs actual balances; stops on mismatch
  DailySummary      - generates and optionally sends end-of-day report
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

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

    cex_connected: bool = False
    cex_last_response_ms: float = 0.0

    dex_connected: bool = False
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
            "drawdown=%.1f%% cex=%s dex=%s errors_1h=%d trades_1h=%d "
            "cb_open=%s kill=%s",
            self.is_running, self.current_capital, self.daily_pnl,
            self.drawdown_pct * 100,
            "OK" if self.cex_connected else "DOWN",
            "OK" if self.dex_connected else "DOWN",
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
# Telegram alerter
# ---------------------------------------------------------------------------

class TelegramAlerter:
    """
    Sends messages to a Telegram chat via the Bot API.

    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the environment.
    If either is missing, alerts are logged but not sent (graceful no-op).
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout_seconds: float = 5.0
    ) -> None:
        self._token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._timeout = timeout_seconds
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            log.info(
                "TelegramAlerter: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set - "
                "alerts will be logged only"
            )

    # ------------------------------------------------------------------
    # Public send methods
    # ------------------------------------------------------------------

    async def critical(self, message: str) -> None:
        """Send a 🚨 critical alert."""
        await self._send(f"🚨 CRITICAL: {message}")

    async def warning(self, message: str) -> None:
        """Send a ⚠️ warning."""
        await self._send(f"⚠️ WARNING: {message}")

    async def info(self, message: str) -> None:
        """Send an ℹ️ informational alert."""
        await self._send(f"{message}")

    async def trade_done(self, metrics: TradeMetrics) -> None:
        sign = "+" if metrics.net_pnl >= 0 else ""
        emoji = "✅" if metrics.net_pnl >= 0 else "❌"
        await self._send(
            f"{emoji} TRADE {metrics.pair} {metrics.direction}\n"
            f"Net PnL: {sign}${metrics.net_pnl:.4f}\n"
            f"Spread: {metrics.actual_spread_bps:.1f}bps | "
            f"TTF: {metrics.signal_to_fill_ms:.0f}ms"
        )

    async def kill_switch_activated(self, reason: str) -> None:
        await self._send(f"🔴 BOT KILLED: {reason}")

    async def daily_summary(self, summary: dict) -> None:
        n = summary.get("trades", 0)
        if n == 0:
            await self._send("📊 Daily Summary: No trades today")
            return
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        pnl = summary.get("total_pnl", 0.0)
        sign = "+" if pnl >= 0 else ""
        wr = summary.get("win_rate", 0.0) * 100
        cap = summary.get("capital", 0.0)
        dd = summary.get("drawdown_pct", 0.0) * 100
        best = summary.get("best_trade", 0.0)
        worst = summary.get("worst_trade", 0.0)
        await self._send(
            f"📊 <b>Daily Summary</b>\n\n"
            f"Trades: {n} ({wins}W / {losses}L)\n"
            f"Win Rate: {wr:.0f}%\n\n"
            f"💰 PnL: <b>{sign}${pnl:.2f}</b>\n"
            f"Best: +${best:.2f}   Worst: ${worst:.2f}\n\n"
            f"Capital: ${cap:.2f}\n"
            f"Drawdown: {dd:.1f}%"
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _send(self, text: str) -> None:
        log.info("ALERT | %s", text)
        if not self._enabled:
            return
        try:
            import aiohttp
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML"
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload, timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        log.warning("Telegram send failed (HTTP %d): %s", resp.status, body[:200])
        except Exception as exc:
            log.warning("Telegram alert failed: %s", exc)


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
        tolerance: float = 0.001,   # 0.1% - allow for rounding
        stop_callback=None,   # callable() to halt the bot
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
        """
        Fetch real balances and compare to inventory expectations.
        Returns True if all within tolerance; False (and stops bot) if not.
        """
        assets = assets or ["ETH", "USDC", "USDT"]
        mismatches: list[str] = []

        # ── CEX balances ───────────────────────────────────────────────
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
                expected = float(
                    self._inventory.available(Venue.BINANCE, asset)
                )
            except Exception:
                continue

            diff = abs(actual_free - expected)
            if expected > 0 and diff / max(expected, 1e-9) > self._tolerance:
                mismatches.append(
                    f"CEX {asset}: expected={expected:.6f} actual={actual_free:.6f} "
                    f"diff={diff:.6f}"
                )

        # ── On-chain (DEX) balances ────────────────────────────────────
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
        """Check on-chain ERC-20 balances. Non-fatal if chain is unreachable."""
        from src.core.types import Address
        from src.inventory.tracker import Venue

        for asset in assets:
            try:
                expected = float(self._inventory.available(Venue.WALLET, asset))
            except Exception:
                continue
            if expected <= 0:
                continue

            # ETH native balance
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
# Bot monitor (wraps health logging and heartbeat)
# ---------------------------------------------------------------------------

class BotMonitor:
    """
    Collects and logs health metrics every minute.
    Also writes the heartbeat file for the dead-man switch.

    Wire into the bot:
        monitor = BotMonitor(risk_manager, circuit_breaker, kill_switch, alerter)
        asyncio.create_task(monitor.run_health_loop())
    """

    def __init__(
        self,
        risk_manager,
        circuit_breaker,
        kill_switch,
        dead_man_switch=None,
        alerter: Optional[TelegramAlerter] = None,
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
        """Background task: log health every minute, write heartbeat every 30s."""
        heartbeat_interval = min(30.0, self._interval / 2)
        last_health_log = 0.0

        while True:
            now = time.time()

            # Heartbeat
            if self._dms:
                self._dms.write_heartbeat()

            # Health snapshot
            if now - last_health_log >= self._interval:
                health = self._build_health()
                health.log_health()
                last_health_log = now

                # Alert on degraded state
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
        """Extract TradeMetrics from an ExecutionContext and log them."""
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
                fees_paid=float((ctx.actual_net_pnl or Decimal("0")) - (ctx.leg_gap_pnl or Decimal("0"))),
                net_pnl=float(ctx.actual_net_pnl or 0),
                state=ctx.state.name
            )
            m.log_trade()
        except Exception as exc:
            log.warning("Could not log trade metrics: %s", exc)


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------

def configure_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Configure structured logging to both a rotating daily file and stdout.
    Call once at bot startup before any other code runs.
    """
    import os
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

    # Silence noisy third-party loggers
    for noisy in ("web3", "urllib3", "asyncio", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger("arb_bot").setLevel(level)
    log.info("Logging initialized | file=%s level=%s", log_file, logging.getLevelName(level))
