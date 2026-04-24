"""
Signal data structures.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum
from typing import Optional

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

_ZERO = Decimal("0")
_ONE = Decimal("1")


class Direction(Enum):
    BUY_CEX_SELL_DEX = "buy_cex_sell_dex"
    BUY_DEX_SELL_CEX = "buy_dex_sell_cex"


@dataclass
class KalmanState:
    """Posterior state of the Kalman spread filter."""

    mean: Decimal = Decimal("0")
    variance: Decimal = Decimal("1")
    process_noise: Decimal = Decimal("1E-5")
    observation_noise: Decimal = Decimal("1E-4")
    innovation: Decimal = Decimal("0")
    kalman_gain: Decimal = Decimal("0")
    tick: int = 0

    def __post_init__(self) -> None:
        self.mean = Decimal(str(self.mean))
        self.variance = Decimal(str(self.variance))
        self.process_noise = Decimal(str(self.process_noise))
        self.observation_noise = Decimal(str(self.observation_noise))
        self.innovation = Decimal(str(self.innovation))
        self.kalman_gain = Decimal(str(self.kalman_gain))


@dataclass
class Signal:
    """
    A validated arbitrage opportunity.
    """

    signal_id: str
    pair: str
    direction: Direction

    cex_price: Decimal
    dex_price: Decimal
    raw_spread_bps: Decimal
    filtered_spread: Decimal
    posterior_variance: Decimal
    signal_confidence: Decimal
    kelly_size: Decimal
    expected_net_pnl: Decimal

    score: Decimal
    timestamp: Decimal
    expiry: Decimal

    inventory_ok: bool
    within_limits: bool
    innovation_zscore: Decimal = Decimal("0")
    kalman_state: Optional[KalmanState] = None

    @classmethod
    def create(
        cls,
        pair: str,
        direction: Direction,
        *,
        cex_price,
        dex_price,
        raw_spread_bps,
        filtered_spread,
        posterior_variance,
        signal_confidence,
        kelly_size,
        expected_net_pnl,
        ttl_seconds=Decimal("5"),
        inventory_ok: bool = True,
        within_limits: bool = True,
        innovation_zscore=Decimal("0"),
        kalman_state: Optional[KalmanState] = None
    ) -> "Signal":
        now = Decimal(str(time.time()))
        ttl = Decimal(str(ttl_seconds))
        return cls(
            signal_id=f"{pair.replace('/', '')}_{uuid.uuid4().hex[:8]}",
            pair=pair,
            direction=direction,
            cex_price=Decimal(str(cex_price)),
            dex_price=Decimal(str(dex_price)),
            raw_spread_bps=Decimal(str(raw_spread_bps)),
            filtered_spread=Decimal(str(filtered_spread)),
            posterior_variance=Decimal(str(posterior_variance)),
            signal_confidence=Decimal(str(signal_confidence)),
            kelly_size=Decimal(str(kelly_size)),
            expected_net_pnl=Decimal(str(expected_net_pnl)),
            score=Decimal("0"),
            timestamp=now,
            expiry=now + ttl,
            inventory_ok=inventory_ok,
            within_limits=within_limits,
            innovation_zscore=Decimal(str(innovation_zscore)),
            kalman_state=kalman_state
        )

    def is_expired(self) -> bool:
        return Decimal(str(time.time())) >= self.expiry

    def age_seconds(self) -> Decimal:
        return Decimal(str(time.time())) - self.timestamp

    def ttl(self) -> Decimal:
        return self.expiry - self.timestamp

    def remaining_ttl(self) -> Decimal:
        r = self.expiry - Decimal(str(time.time()))
        return r if r > _ZERO else _ZERO

    def is_valid(self) -> bool:
        return (
            not self.is_expired()
            and self.inventory_ok
            and self.within_limits
            and self.expected_net_pnl > _ZERO
            and self.signal_confidence > _ZERO
        )

    def is_anomalous(self, zscore_threshold=Decimal("3")) -> bool:
        return abs(self.innovation_zscore) > Decimal(str(zscore_threshold))
