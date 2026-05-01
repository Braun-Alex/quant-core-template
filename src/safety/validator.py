"""
Pre-trade signal validation.

PreTradeValidator runs fast, stateless sanity checks on every Signal
before it reaches the risk manager or executor. Catches bad data, stale
prices, and obviously absurd spreads before they can cause damage.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal

from src.strategy.signal import Signal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Price history (per-pair rolling window for deviation checks)
# ---------------------------------------------------------------------------

@dataclass
class PriceSnapshot:
    mid_price: float
    timestamp: float


class _PriceHistory:
    """Rolling 5-minute price history for deviation guard."""

    def __init__(self, window_seconds: float = 300.0, max_deviation_pct: float = 0.05):
        self._window = window_seconds
        self._max_dev = max_deviation_pct
        self._history: dict[str, deque[PriceSnapshot]] = {}

    def add(self, pair: str, price: float) -> None:
        if pair not in self._history:
            self._history[pair] = deque()
        self._history[pair].append(PriceSnapshot(mid_price=price, timestamp=time.time()))
        self._prune(pair)

    def check_deviation(self, pair: str, price: float) -> tuple[bool, str]:
        if pair not in self._history or not self._history[pair]:
            return True, "OK"   # No history yet - allow
        self._prune(pair)
        if not self._history[pair]:
            return True, "OK"
        avg = sum(s.mid_price for s in self._history[pair]) / len(self._history[pair])
        if avg <= 0:
            return True, "OK"
        deviation = abs(price - avg) / avg
        if deviation > self._max_dev:
            return False, (
                f"Price {price:.2f} deviates {deviation:.1%} from "
                f"{len(self._history[pair])}-sample avg {avg:.2f}"
            )
        return True, "OK"

    def _prune(self, pair: str) -> None:
        cutoff = time.time() - self._window
        while self._history.get(pair) and self._history[pair][0].timestamp < cutoff:
            self._history[pair].popleft()


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class PreTradeValidator:
    """
    Stateless and lightly stateful sanity checks for every Signal.

    Checks (in order):
      1. Price positivity
      2. Signal freshness
      3. Spread magnitude (catches bad price feeds)
      4. Trade size positivity
      5. Price deviation from recent history (5-min rolling avg)
      6. CEX / DEX price ratio sanity
    """

    def __init__(
        self,
        max_spread_bps: float = 500.0,
        max_signal_age_seconds: float = 5.0,
        max_price_deviation_pct: float = 0.05,   # 5% from 5-min avg
        min_spread_bps: float = 0.5   # Suspiciously tight
    ) -> None:
        self._max_spread_bps = max_spread_bps
        self._max_age = max_signal_age_seconds
        self._min_spread_bps = min_spread_bps
        self._price_history = _PriceHistory(
            window_seconds=300.0,
            max_deviation_pct=max_price_deviation_pct
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Run all checks on a Signal.
        Returns (valid: bool, reason: str).
        """
        checks = [
            self._check_prices,
            self._check_freshness,
            self._check_spread,
            self._check_size,
            self._check_price_deviation,
            self._check_cex_dex_ratio
        ]
        for check in checks:
            ok, reason = check(signal)
            if not ok:
                log.warning(
                    "VALIDATION_FAIL | signal=%s check=%s reason=%s",
                    signal.signal_id, check.__name__, reason
                )
                return False, reason

        # Record price after all checks pass
        mid = float(signal.cex_price + signal.dex_price) / 2
        self._price_history.add(signal.pair, mid)

        return True, "OK"

    def validate_post_fill(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        max_slippage_bps: float = 100.0
    ) -> tuple[bool, str]:
        """
        Check post-execution slippage against expectation.
        Call after leg 1 fills to decide whether to proceed with leg 2.
        """
        if expected_price <= 0:
            return False, "Expected price is zero"
        slippage_bps = abs(actual_price - expected_price) / expected_price * Decimal("10000")
        if float(slippage_bps) > max_slippage_bps:
            return False, (
                f"Post-fill slippage {float(slippage_bps):.1f}bps "
                f"exceeds limit {max_slippage_bps:.1f}bps"
            )
        return True, "OK"

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_prices(self, s: Signal) -> tuple[bool, str]:
        if s.cex_price <= 0:
            return False, f"Invalid CEX price: {s.cex_price}"
        if s.dex_price <= 0:
            return False, f"Invalid DEX price: {s.dex_price}"
        return True, "OK"

    def _check_freshness(self, s: Signal) -> tuple[bool, str]:
        age = float(s.age_seconds())
        if age > self._max_age:
            return False, f"Signal too old: {age:.1f}s > {self._max_age:.1f}s limit"
        return True, "OK"

    def _check_spread(self, s: Signal) -> tuple[bool, str]:
        bps = float(s.raw_spread_bps)
        if bps > self._max_spread_bps:
            return False, (
                f"Spread {bps:.1f}bps exceeds {self._max_spread_bps:.0f}bps - "
                "likely bad price data"
            )
        if bps < self._min_spread_bps:
            return False, (
                f"Spread {bps:.2f}bps below {self._min_spread_bps:.2f}bps - "
                "no real opportunity"
            )
        return True, "OK"

    def _check_size(self, s: Signal) -> tuple[bool, str]:
        if s.kelly_size <= 0:
            return False, f"Invalid trade size: {s.kelly_size}"
        return True, "OK"

    def _check_price_deviation(self, s: Signal) -> tuple[bool, str]:
        mid = (float(s.cex_price) + float(s.dex_price)) / 2
        return self._price_history.check_deviation(s.pair, mid)

    def _check_cex_dex_ratio(self, s: Signal) -> tuple[bool, str]:
        """Reject if the CEX/DEX price ratio is beyond 10% - almost certainly noise."""
        if s.dex_price <= 0:
            return True, "OK"
        ratio = float(s.cex_price / s.dex_price)
        if not (0.90 <= ratio <= 1.10):
            return False, (
                f"CEX/DEX price ratio {ratio:.3f} outside [0.90, 1.10] - "
                "prices are likely from different assets or stale"
            )
        return True, "OK"
