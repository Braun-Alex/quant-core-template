"""
Entropy-Weighted MCDM Opportunity Scorer.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Optional

import numpy as np

from src.strategy.signal import Signal

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

log = logging.getLogger(__name__)

_EPS = 1e-12
_ZERO = Decimal("0")
_ONE = Decimal("1")


def _d(v) -> Decimal:
    """Cast any numeric to Decimal."""
    return Decimal(str(v))


@dataclass
class ScorerConfig:
    min_history: int = 3
    history_window: int = 100
    decay_halflife: Decimal = Decimal("3")
    excellent_pnl_usd: Decimal = Decimal("50")

    def __post_init__(self) -> None:
        self.decay_halflife = _d(self.decay_halflife)
        self.excellent_pnl_usd = _d(self.excellent_pnl_usd)


class EntropyCRITIC:
    """
    CRITIC entropy weights from a float64 numpy decision matrix.
    """

    @staticmethod
    def weights(matrix: np.ndarray) -> np.ndarray:
        n, m = matrix.shape
        if n < 2:
            return np.ones(m) / m

        col_min = matrix.min(axis=0)
        col_max = matrix.max(axis=0)
        denom = np.where(col_max - col_min > _EPS, col_max - col_min, _EPS)
        R = (matrix - col_min) / denom   # Normalised [0, 1]

        sigma = R.std(axis=0)

        # Pearson correlation with NaN guard
        if n >= 2 and m >= 2:
            try:
                with np.errstate(invalid="ignore"):
                    rho = np.corrcoef(R.T)
                rho = np.nan_to_num(rho, nan=0.0)
            except Exception:
                rho = np.eye(m)
        else:
            rho = np.eye(m)

        # Contrast intensity
        C = np.array([
            sigma[j] * sum(1.0 - rho[j, k] for k in range(m) if k != j)
            for j in range(m)
        ])

        total = C.sum()
        if total < _EPS:
            return np.ones(m) / m
        return C / total


class TOPSIS:
    """TOPSIS scoring."""

    @staticmethod
    def score(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        n, m = matrix.shape
        if n == 1:
            return np.array([1.0])

        col_min = matrix.min(axis=0)
        col_max = matrix.max(axis=0)
        denom = np.where(col_max - col_min > _EPS, col_max - col_min, _EPS)
        R = (matrix - col_min) / denom
        V = R * weights[np.newaxis, :]

        A_plus = V.max(axis=0)
        A_minus = V.min(axis=0)

        D_plus = np.sqrt(((V - A_plus) ** 2).sum(axis=1))
        D_minus = np.sqrt(((V - A_minus) ** 2).sum(axis=1))

        denom_d = D_plus + D_minus
        denom_d = np.where(denom_d > _EPS, denom_d, _EPS)
        return D_minus / denom_d


class SignalScorer:
    """
    Entropy-CRITIC + TOPSIS scorer with temporal decay.
    """

    def __init__(self, config: Optional[ScorerConfig] = None) -> None:
        self._cfg = config or ScorerConfig()
        self._history: dict[str, deque[bool]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, signal: Signal, inventory_skews: list[dict]) -> Decimal:
        """Score a single signal — returns Decimal in [0, 1]."""
        vec = self._criteria_vec(signal, inventory_skews)
        raw = _d(float(np.mean(vec)))
        return self._apply_decay(raw, signal.age_seconds(), signal.ttl())

    def score_batch(
        self,
        signals: list[Signal],
        inventory_skews: list[dict]
    ) -> dict[str, Decimal]:
        """Batch TOPSIS."""
        if not signals:
            return {}

        matrix = np.array([self._criteria_vec(s, inventory_skews) for s in signals])
        if len(signals) >= self._cfg.min_history:
            weights = EntropyCRITIC.weights(matrix)
        else:
            weights = np.ones(matrix.shape[1]) / matrix.shape[1]

        raw_scores = TOPSIS.score(matrix, weights)

        return {
            sig.signal_id: self._apply_decay(
                _d(float(raw)), sig.age_seconds(), sig.ttl()
            )
            for sig, raw in zip(signals, raw_scores)
        }

    def record_result(self, pair: str, success: bool) -> None:
        if pair not in self._history:
            self._history[pair] = deque(maxlen=self._cfg.history_window)
        self._history[pair].append(success)

    def success_rate(self, pair: str) -> Decimal:
        h = self._history.get(pair)
        if not h:
            return Decimal("0.5")
        return _d(sum(h) / len(h))

    # ------------------------------------------------------------------
    # Criteria
    # ------------------------------------------------------------------

    def _criteria_vec(self, signal: Signal, inventory_skews: list[dict]) -> np.ndarray:
        x1 = float(signal.signal_confidence)
        x2 = float(self._pnl_score(signal.expected_net_pnl))
        x3 = float(self._market_impact_score(signal))
        x4 = float(self._inventory_health(signal, inventory_skews))
        x5 = float(self._latency_score(signal))
        x6 = float(self.success_rate(signal.pair))
        return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

    def _pnl_score(self, net_pnl: Decimal) -> Decimal:
        pnl_f = float(max(net_pnl, _ZERO))
        exc = float(self._cfg.excellent_pnl_usd)
        return _d(1.0 - math.exp(-pnl_f / exc)) if exc > 0 else _ZERO

    def _market_impact_score(self, signal: Signal) -> Decimal:
        var_f = float(signal.posterior_variance) * 1e4
        cert = math.exp(-min(var_f, 10.0))
        z_f = float(signal.innovation_zscore)
        anomaly = math.exp(-0.5 * z_f ** 2)
        return _d(0.5 * cert + 0.5 * anomaly)

    def _inventory_health(self, signal: Signal, skews: list[dict]) -> Decimal:
        base = signal.pair.split("/")[0]
        relevant = [s for s in skews if s.get("asset") == base]
        if not relevant:
            return _d("0.6")
        max_dev = max(float(s.get("max_deviation_pct", 0)) for s in relevant)
        return _d(math.exp(-max_dev / 50.0))

    def _latency_score(self, signal: Signal) -> Decimal:
        ttl_f = max(float(signal.ttl()), 1e-3)
        age_f = float(signal.age_seconds())
        return _d(math.exp(-age_f / ttl_f))

    # ------------------------------------------------------------------
    # Temporal decay
    # ------------------------------------------------------------------

    def _apply_decay(self, score: Decimal, age: Decimal, ttl: Decimal) -> Decimal:
        """S(t) = S * exp(-ln2 / halflife * age), capped at ttl."""
        hl = max(self._cfg.decay_halflife, _d("1E-3"))
        age = min(age, ttl)
        ln2 = _d(math.log(2))
        decay = _d(math.exp(-float(ln2 / hl) * float(age)))
        result = score * decay
        return min(max(result, _ZERO), _ONE)
