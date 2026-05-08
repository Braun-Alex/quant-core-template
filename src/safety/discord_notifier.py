"""
Discord Webhook Notifier for the arbitrage bot.

Sends structured embed messages to a Discord channel via webhook.

Environment variables
---------------------
DISCORD_WEBHOOK_URL   Full Discord webhook URL
                      e.g. https://discord.com/api/webhooks/<id>/<token>

Usage
-----
    alerter = DiscordAlerter.from_env()
    await alerter.info("Bot started in test mode")
    await alerter.critical("Kill switch activated!")
    await alerter.daily_summary(summary_dict)
"""

from __future__ import annotations

import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

# Discord embed color palette
_COLOUR_CRITICAL = 0xE74C3C   # Red
_COLOUR_WARNING = 0xF39C12   # Orange
_COLOUR_INFO = 0x3498DB   # Blue
_COLOUR_SUCCESS = 0x2ECC71   # Green
_COLOUR_NEUTRAL = 0x95A5A6   # Grey


class DiscordAlerter:
    """
    Sends Discord webhook messages as embeds.

    All send methods are async and swallow exceptions internally so
    a failed Discord notification never crashes the bot loop.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        timeout_seconds: float = 5.0,
        username: str = "ArbBot",
        avatar_url: str = ""
    ) -> None:
        self._url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL", "")
        self._timeout = timeout_seconds
        self._username = username
        self._avatar_url = avatar_url
        self._enabled = bool(self._url)

        if not self._enabled:
            log.info(
                "DiscordAlerter: DISCORD_WEBHOOK_URL not set - "
                "Discord alerts will be logged only"
            )

    @classmethod
    def from_env(cls) -> "DiscordAlerter":
        """Create from DISCORD_WEBHOOK_URL environment variable."""
        return cls(
            webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            username=os.getenv("DISCORD_BOT_USERNAME", "ArbBot"),
            avatar_url=os.getenv("DISCORD_AVATAR_URL", "")
        )

    # ------------------------------------------------------------------
    # Public alert methods
    # ------------------------------------------------------------------

    async def critical(self, message: str) -> None:
        """Send a 🚨 critical alert embed."""
        await self._send_embed(
            title="🚨 CRITICAL ALERT",
            description=message,
            colour=_COLOUR_CRITICAL
        )

    async def warning(self, message: str) -> None:
        """Send a ⚠️ warning embed."""
        await self._send_embed(
            title="⚠️ Warning",
            description=message,
            colour=_COLOUR_WARNING
        )

    async def info(self, message: str) -> None:
        """Send an ℹ️ informational embed."""
        await self._send_embed(
            title="ℹ️ Info",
            description=message,
            colour=_COLOUR_INFO
        )

    async def kill_switch_activated(self, reason: str) -> None:
        """Send a bot-killed embed."""
        await self._send_embed(
            title="🔴 BOT KILLED",
            description=f"Kill switch activated: {reason}",
            colour=_COLOUR_CRITICAL
        )

    async def trade_done(self, metrics) -> None:
        """
        Send a trade-execution summary embed.
        Accepts a TradeMetrics instance (or any object with the same attributes).
        """
        sign = "+" if metrics.net_pnl >= 0 else ""
        colour = _COLOUR_SUCCESS if metrics.net_pnl >= 0 else _COLOUR_CRITICAL
        emoji = "✅" if metrics.net_pnl >= 0 else "❌"

        fields = [
            {"name": "Pair", "value": metrics.pair, "inline": True},
            {"name": "Direction", "value": metrics.direction, "inline": True},
            {"name": "Net PnL", "value": f"{sign}${metrics.net_pnl:.4f}", "inline": True},
            {"name": "Spread", "value": f"{metrics.actual_spread_bps:.1f} bps", "inline": True},
            {"name": "TTF", "value": f"{metrics.signal_to_fill_ms:.0f} ms", "inline": True},
            {"name": "State", "value": metrics.state, "inline": True}
        ]

        await self._send_embed(
            title=f"{emoji} Trade Executed",
            colour=colour,
            fields=fields
        )

    async def daily_summary(self, summary: dict) -> None:
        """Send an end-of-day performance summary embed."""
        n = summary.get("trades", 0)
        if n == 0:
            await self._send_embed(
                title="📊 Daily Summary",
                description="No trades executed today.",
                colour=_COLOUR_NEUTRAL
            )
            return

        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        pnl = summary.get("total_pnl", 0.0)
        wr = summary.get("win_rate", 0.0) * 100
        cap = summary.get("capital", 0.0)
        dd = summary.get("drawdown_pct", 0.0) * 100
        best = summary.get("best_trade", 0.0)
        worst = summary.get("worst_trade", 0.0)
        sign = "+" if pnl >= 0 else ""
        colour = _COLOUR_SUCCESS if pnl >= 0 else _COLOUR_CRITICAL

        fields = [
            {"name": "Trades", "value": f"{n} ({wins}W / {losses}L)", "inline": True},
            {"name": "Win Rate", "value": f"{wr:.0f}%", "inline": True},
            {"name": "Net PnL", "value": f"{sign}${pnl:.2f}", "inline": True},
            {"name": "Best Trade", "value": f"+${best:.2f}", "inline": True},
            {"name": "Worst Trade", "value": f"${worst:.2f}", "inline": True},
            {"name": "Capital", "value": f"${cap:.2f}", "inline": True},
            {"name": "Drawdown", "value": f"{dd:.1f}%", "inline": True},
        ]

        await self._send_embed(
            title="📊 Daily Summary",
            colour=colour,
            fields=fields
        )

    # ------------------------------------------------------------------
    # Low-level webhook sender
    # ------------------------------------------------------------------

    async def _send_embed(
        self,
        title: str,
        description: str = "",
        colour: int = _COLOUR_INFO,
        fields: Optional[list[dict]] = None
    ) -> None:
        """Build and POST a Discord embed payload."""
        msg = f"[DISCORD] {title}"
        if description:
            msg += f" | {description}"
        log.info(msg)

        if not self._enabled:
            return

        embed: dict = {
            "title": title,
            "colour": colour,
            "timestamp": _iso_now()
        }
        if description:
            embed["description"] = description
        if fields:
            embed["fields"] = fields

        payload: dict = {
            "embeds": [embed]
        }
        if self._username:
            payload["username"] = self._username
        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self._timeout)
                ) as resp:
                    if resp.status not in (200, 204):
                        body = await resp.text()
                        log.warning(
                            "Discord webhook failed (HTTP %d): %s",
                            resp.status, body[:200]
                        )
        except Exception as exc:
            log.warning("Discord alert failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso_now() -> str:
    """Return current UTC time in ISO-8601 format for Discord embed timestamps."""
    import datetime
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
