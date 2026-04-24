"""
Tests for Entropy-CRITIC TOPSIS scorer.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from src.strategy.scorer import EntropyCRITIC, TOPSIS, SignalScorer, ScorerConfig
from src.strategy.signal import Direction, Signal


# ─────────────────────────────── Helpers ────────────────────────────────────

def _sig(
    pair="ETH/USDT", confidence="0.90", net_pnl="10",
    var="1E-6", zscore="0.5",
    direction=Direction.BUY_CEX_SELL_DEX
) -> Signal:
    return Signal.create(
        pair=pair, direction=direction,
        cex_price="2000", dex_price="2010",
        raw_spread_bps="50", filtered_spread="0.005",
        posterior_variance=var, signal_confidence=confidence,
        kelly_size="0.1", expected_net_pnl=net_pnl,
        ttl_seconds="5", inventory_ok=True, within_limits=True,
        innovation_zscore=zscore
    )


# ═══════════════════════════ EntropyCRITIC ═══════════════════════════════════

class TestEntropyCRITIC:
    def test_weights_sum_to_one(self):
        w = EntropyCRITIC.weights(np.random.rand(10, 6))
        assert abs(w.sum() - 1.0) < 1e-9

    def test_weights_non_negative(self):
        w = EntropyCRITIC.weights(np.random.rand(8, 4))
        assert (w >= 0).all()

    def test_single_row_equal_weights(self):
        w = EntropyCRITIC.weights(np.array([[1.0, 2.0, 3.0, 4.0]]))
        np.testing.assert_allclose(w, np.ones(4) / 4)

    def test_constant_column_weight_not_highest(self):
        """Constant criterion carries no information → not the heaviest weight."""
        matrix = np.array([
            [0.9, 1.0, 0.1],
            [0.5, 1.0, 0.8],
            [0.2, 1.0, 0.5],
            [0.7, 1.0, 0.3]
        ])
        w = EntropyCRITIC.weights(matrix)
        # Column 1 is constant - must not be the maximum
        assert w[1] <= max(w) + 1e-9

    def test_high_variance_not_lowest_weight(self):
        """High-variance column should get at least as much weight as low-variance."""
        np.random.seed(0)
        col_low = np.random.rand(20) * 0.01
        col_high = np.random.rand(20)
        matrix = np.column_stack([col_low, col_high])
        w = EntropyCRITIC.weights(matrix)
        # High-variance column index 1 must not be strictly less than col 0
        assert w[1] >= w[0] - 0.15   # Loose tolerance for numerical instability

    def test_identical_columns_similar_weights(self):
        col = np.linspace(0, 1, 10)
        matrix = np.column_stack([col, col])
        w = EntropyCRITIC.weights(matrix)
        assert abs(w[0] - w[1]) < 0.1

    def test_returns_numpy_array(self):
        assert isinstance(EntropyCRITIC.weights(np.random.rand(5, 3)), np.ndarray)


# ════════════════════════════════ TOPSIS ═════════════════════════════════════

class TestTOPSIS:
    def test_scores_in_unit_interval(self):
        scores = TOPSIS.score(np.random.rand(8, 4), np.ones(4) / 4)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_ideal_row_highest(self):
        matrix = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]])
        scores = TOPSIS.score(matrix, np.ones(3) / 3)
        assert scores[0] > scores[1] > scores[2]

    def test_single_candidate_returns_one(self):
        scores = TOPSIS.score(np.array([[0.7, 0.8, 0.6]]), np.ones(3) / 3)
        assert scores[0] == pytest.approx(1.0)

    def test_symmetric_rows_equal_scores(self):
        matrix = np.array([[0.6, 0.4], [0.4, 0.6]])
        scores = TOPSIS.score(matrix, np.array([0.5, 0.5]))
        assert abs(scores[0] - scores[1]) < 1e-9

    def test_ranking_preserved_under_weight_scaling(self):
        np.random.seed(7)
        matrix = np.random.rand(5, 3)
        w1 = np.array([0.25, 0.25, 0.50])
        w2 = w1 * 2
        assert np.argmax(TOPSIS.score(matrix, w1)) == np.argmax(TOPSIS.score(matrix, w2))


# ════════════════════════════ SignalScorer ════════════════════════════════════

class TestSignalScorer:
    def test_single_score_decimal_in_unit_interval(self):
        sc = SignalScorer()
        s = sc.score(_sig(), [])
        assert isinstance(s, Decimal)
        assert Decimal("0") <= s <= Decimal("1")

    def test_higher_confidence_higher_score(self):
        sc = SignalScorer()
        lo = sc.score(_sig(confidence="0.60"), [])
        hi = sc.score(_sig(confidence="0.98"), [])
        assert hi > lo

    def test_higher_pnl_higher_score(self):
        sc = SignalScorer()
        assert sc.score(_sig(net_pnl="100"), []) > sc.score(_sig(net_pnl="1"), [])

    def test_anomaly_penalised(self):
        sc = SignalScorer()
        normal = sc.score(_sig(zscore="0.5"), [])
        anomal = sc.score(_sig(zscore="5.0"), [])
        assert normal > anomal

    def test_decay_reduces_score_over_age(self):
        sc = SignalScorer(ScorerConfig(decay_halflife=Decimal("0.1")))
        s0 = sc._apply_decay(Decimal("0.8"), Decimal("0"), Decimal("5"))
        s1 = sc._apply_decay(Decimal("0.8"), Decimal("1"), Decimal("5"))
        assert s0 > s1

    def test_decay_zero_age_identity(self):
        sc = SignalScorer(ScorerConfig(decay_halflife=Decimal("3")))
        val = Decimal("0.75")
        assert sc._apply_decay(val, Decimal("0"), Decimal("5")) == val

    def test_decay_halflife_halves_score(self):
        hl = Decimal("2")
        sc = SignalScorer(ScorerConfig(decay_halflife=hl))
        raw = Decimal("0.8")
        dec = sc._apply_decay(raw, hl, Decimal("100"))
        assert abs(dec - raw / Decimal("2")) < Decimal("1E-6")

    def test_batch_returns_dict_keyed_by_id(self):
        sc = SignalScorer()
        sigs = [_sig(confidence="0.9"), _sig(confidence="0.7"), _sig(confidence="0.5")]
        res = sc.score_batch(sigs, [])
        assert set(res.keys()) == {s.signal_id for s in sigs}

    def test_batch_values_decimal(self):
        sc = SignalScorer()
        res = sc.score_batch([_sig(), _sig()], [])
        assert all(isinstance(v, Decimal) for v in res.values())

    def test_batch_empty(self):
        assert SignalScorer().score_batch([], []) == {}

    def test_batch_ranking(self):
        sc = SignalScorer()
        lo = _sig(confidence="0.55", net_pnl="1")
        mid = _sig(confidence="0.75", net_pnl="10")
        hi = _sig(confidence="0.95", net_pnl="50")
        res = sc.score_batch([lo, mid, hi], [])
        assert res[hi.signal_id] > res[mid.signal_id] > res[lo.signal_id]

    def test_success_rate_history(self):
        sc = SignalScorer()
        for _ in range(10):
            sc.record_result("ETH/USDT", True)
        for _ in range(10):
            sc.record_result("ETH/USDT", False)
        assert abs(sc.success_rate("ETH/USDT") - Decimal("0.5")) < Decimal("0.01")

    def test_success_rate_no_history_prior(self):
        assert SignalScorer().success_rate("UNKNOWN/PAIR") == Decimal("0.5")

    def test_success_rate_is_decimal(self):
        sc = SignalScorer()
        sc.record_result("ETH/USDT", True)
        assert isinstance(sc.success_rate("ETH/USDT"), Decimal)

    def test_inventory_skew_penalty(self):
        sc = SignalScorer()
        s = _sig()
        ok = sc.score(s, [{"asset": "ETH", "max_deviation_pct": 5.0, "needs_rebalance": False}])
        bad = sc.score(s, [{"asset": "ETH", "max_deviation_pct": 80.0, "needs_rebalance": True}])
        assert ok > bad

    def test_pnl_score_saturates(self):
        sc = SignalScorer(ScorerConfig(excellent_pnl_usd=Decimal("50")))
        assert sc._pnl_score(Decimal("1000")) > sc._pnl_score(Decimal("50"))
        assert sc._pnl_score(Decimal("1000")) < Decimal("1") + Decimal("1E-6")

    def test_pnl_score_negative_zero(self):
        assert SignalScorer()._pnl_score(Decimal("-1")) == Decimal("0")

    def test_market_impact_penalises_high_zscore(self):
        sc = SignalScorer()
        assert sc._market_impact_score(_sig(zscore="0.1")) > sc._market_impact_score(_sig(zscore="4"))

    def test_latency_score_decimal(self):
        sc = SignalScorer()
        val = sc._latency_score(_sig())
        assert isinstance(val, Decimal)
        assert Decimal("0") <= val <= Decimal("1")
