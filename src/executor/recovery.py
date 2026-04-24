"""
SPRT-Based Circuit Breaker.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Optional
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam
)

from src.strategy.signal import Signal

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

log = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")


def _d(v) -> Decimal:
    return Decimal(str(v))


# ---------------------------------------------------------------------------
# SPRT config
# ---------------------------------------------------------------------------

@dataclass
class SPRTConfig:
    p0: Decimal = Decimal("0.10")
    p1: Decimal = Decimal("0.40")
    alpha: Decimal = Decimal("0.05")
    beta: Decimal = Decimal("0.10")
    gamma: Decimal = Decimal("0.95")
    base_cooldown: Decimal = Decimal("60")
    max_cooldown: Decimal = Decimal("3600")
    max_backoff_steps: int = 6

    def __post_init__(self) -> None:
        for f in ("p0", "p1", "alpha", "beta", "gamma", "base_cooldown", "max_cooldown"):
            setattr(self, f, _d(getattr(self, f)))


# ---------------------------------------------------------------------------
# SPRT Circuit Breaker
# ---------------------------------------------------------------------------

class SPRTCircuitBreaker:
    """
    Discounted SPRT circuit breaker.
    Lambda_t = gamma * Lambda_{t-1} + X_t * ln(p1/p0) + (1-X_t)*ln((1-p1)/(1-p0))
    Trip  when Lambda_t >= B = ln((1-beta)/alpha)
    Reset when Lambda_t <= A = ln(beta/(1-alpha))
    """

    def __init__(self, config: Optional[SPRTConfig] = None) -> None:
        self._cfg = config or SPRTConfig()
        self._lambda = _ZERO
        self._tripped_at: Optional[Decimal] = None
        self._trip_count = 0
        self._total_obs = 0

        cfg = self._cfg
        # LLR increments
        self._llr_failure = _d(math.log(float(cfg.p1 / cfg.p0)))
        self._llr_success = _d(math.log(float((_ONE - cfg.p1) / (_ONE - cfg.p0))))

        # Decision boundaries
        self._B = _d(math.log(float((_ONE - cfg.beta) / cfg.alpha)))
        self._A = _d(math.log(float(cfg.beta / (_ONE - cfg.alpha))))

        log.debug("SPRT A=%.4f B=%.4f", float(self._A), float(self._B))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def record_success(self) -> None:
        self._update(success=True)

    def record_failure(self) -> None:
        self._update(success=False)
        if self._lambda >= self._B and self._tripped_at is None:
            self._trip()

    def is_open(self) -> bool:
        if self._tripped_at is None:
            return False
        if _d(time.time()) >= self._tripped_at + self._current_cooldown():
            log.info("Circuit breaker auto-reset.")
            self._reset()
            return False
        return True

    def trip(self) -> None:
        self._trip()

    def reset(self) -> None:
        self._reset()

    def time_until_reset(self) -> Decimal:
        if self._tripped_at is None:
            return _ZERO
        remaining = self._tripped_at + self._current_cooldown() - _d(time.time())
        return max(remaining, _ZERO)

    @property
    def lambda_statistic(self) -> Decimal:
        return self._lambda

    @property
    def trip_count(self) -> int:
        return self._trip_count

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update(self, success: bool) -> None:
        increment = self._llr_success if success else self._llr_failure
        self._lambda = self._cfg.gamma * self._lambda + increment
        self._lambda = max(self._lambda, self._A)
        self._total_obs += 1
        log.debug("SPRT Lambda=%.4f (A=%.4f B=%.4f)",
                  float(self._lambda), float(self._A), float(self._B))
        if self._lambda <= self._A and self._tripped_at is not None:
            log.info("SPRT reset boundary reached.")
            self._reset()

    def _trip(self) -> None:
        self._tripped_at = _d(time.time())
        self._trip_count += 1
        log.critical("CIRCUIT BREAKER TRIPPED #%d Lambda=%.4f >= B=%.4f",
                     self._trip_count, float(self._lambda), float(self._B))

    def _reset(self) -> None:
        self._tripped_at = None
        self._lambda = _ZERO

    def _current_cooldown(self) -> Decimal:
        k = min(self._trip_count, self._cfg.max_backoff_steps)
        cd = self._cfg.base_cooldown * _d(2 ** max(k - 1, 0))
        return min(cd, self._cfg.max_cooldown)


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
# LLM Anomaly Advisor
# ---------------------------------------------------------------------------

@dataclass
class AnomalyExplanation:
    anomaly_type: str
    suppress_signal: bool
    confidence: Decimal
    reasoning: str
    raw_response: str = ""


class LLMAnomalyAdvisor:
    SYSTEM_PROMPT = (
        "You are an expert quantitative analyst reviewing anomalous price spreads "
        "in a CEX-DEX arbitrage system. Respond ONLY with valid JSON: "
        '{"anomaly_type":"flash_crash|pump|liquidity_gap|noise",'
        '"suppress_signal":true|false,"confidence":0.0-1.0,"reasoning":"..."}'
    )

    def __init__(
            self,
            api_key: str = "",
            model: str = "gpt-5.4-nano",
            zscore_threshold: Decimal = Decimal("3")
    ) -> None:
        self._model = model
        self._threshold = _d(zscore_threshold)
        self._enabled = bool(api_key)

        self._client: Optional[AsyncOpenAI] = None
        if self._enabled:
            self._client = AsyncOpenAI(api_key=api_key)

    def should_query(
            self,
            innovation_zscore: Decimal,
            sprt_lambda: Decimal,
            sprt_A: Decimal,
            sprt_B: Decimal
    ) -> bool:
        if not self._enabled:
            return False

        anomalous = abs(innovation_zscore) > self._threshold
        indeterminate = sprt_A < sprt_lambda < sprt_B
        return anomalous and indeterminate

    async def advise(
            self,
            pair: str,
            innovation_zscore: Decimal,
            filtered_spread: Decimal,
            raw_spread_bps: Decimal,
            recent_pnl_summary: dict
    ) -> Optional["AnomalyExplanation"]:
        if not self._enabled or self._client is None:
            return None

        import json

        try:
            user_content = (
                f"Pair: {pair}\n"
                f"Kalman innovation z-score: {float(innovation_zscore):.3f}\n"
                f"Filtered log-spread: {float(filtered_spread):.6f}\n"
                f"Raw spread (bps): {float(raw_spread_bps):.2f}\n"
                f"Recent PnL summary: {json.dumps(recent_pnl_summary)}\n"
                "Classify and advise whether to suppress the signal."
            )

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    ChatCompletionSystemMessageParam(
                        role="system",
                        content=self.SYSTEM_PROMPT
                    ),
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=user_content
                    )
                ],
                max_tokens=256,
                temperature=0.0
            )

            raw = response.choices[0].message.content.strip()
            parsed = json.loads(raw)

            return AnomalyExplanation(
                anomaly_type=parsed.get("anomaly_type", "noise"),
                suppress_signal=bool(parsed.get("suppress_signal", False)),
                confidence=_d(parsed.get("confidence", 0.5)),
                reasoning=parsed.get("reasoning", ""),
                raw_response=raw
            )

        except Exception as exc:
            log.warning("LLM advisor failed: %s", exc)
            return None
