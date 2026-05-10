"""
Bayesian Kalman Spread Estimator.
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from typing import Optional

from src.inventory.tracker import Venue
from src.strategy.signal import Direction, KalmanState, Signal

try:
    from src.exchange.price_feed import PriceState
except ImportError:
    PriceState = None

getcontext().prec = 28
getcontext().rounding = ROUND_HALF_EVEN

log = logging.getLogger(__name__)

_ZERO = Decimal("0")
_ONE = Decimal("1")
_TEN_THOU = Decimal("10000")
_EPS = Decimal("1E-12")


# ============================================================================
# Fee structure
# ============================================================================

@dataclass
class FeeStructure:
    cex_taker_bps: Decimal = Decimal("10")
    dex_swap_bps: Decimal = Decimal("30")

    # Lower default for simulation/demo
    gas_cost_usd: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        self.cex_taker_bps = Decimal(str(self.cex_taker_bps))
        self.dex_swap_bps = Decimal(str(self.dex_swap_bps))
        self.gas_cost_usd = Decimal(str(self.gas_cost_usd))

    def total_fee_bps(self, trade_value_usd: Decimal) -> Decimal:
        if trade_value_usd <= _ZERO:
            return Decimal("Inf")

        gas_bps = (self.gas_cost_usd / trade_value_usd) * _TEN_THOU

        log.info(f"Gas cost: {self.gas_cost_usd}$ | trade value: {trade_value_usd}$")

        total = (
            self.cex_taker_bps
            + self.dex_swap_bps
            + gas_bps
        )

        log.info(
            "Fee model | trade_value=%.6f gas_bps=%.4f "
            "cex_bps=%.4f dex_bps=%.4f total_bps=%.4f",
            float(trade_value_usd),
            float(gas_bps),
            float(self.cex_taker_bps),
            float(self.dex_swap_bps),
            float(total)
        )

        return total

    def breakeven_log_spread(self, trade_value_usd: Decimal) -> Decimal:
        threshold = self.total_fee_bps(trade_value_usd) / _TEN_THOU

        log.info(
            "Break-even spread | trade_value=%.6f threshold=%.8f",
            float(trade_value_usd),
            float(threshold)
        )

        return threshold


# ============================================================================
# Kalman Filter
# ============================================================================

class KalmanSpreadFilter:
    def __init__(
        self,
        init_Q: Decimal = Decimal("1E-5"),
        init_R: Decimal = Decimal("1E-4"),
        em_window: int = 50
    ) -> None:

        self._state = KalmanState(
            mean=Decimal("0"),
            variance=Decimal("1"),
            process_noise=Decimal(str(init_Q)),
            observation_noise=Decimal(str(init_R))
        )

        self._em_window = em_window

        self._buf: deque[
            tuple[Decimal, Decimal, Decimal, Decimal, Decimal]
        ] = deque(maxlen=em_window)

        self._last_zscore: Decimal = Decimal("0")

    def update(self, z_float: float) -> KalmanState:
        z = Decimal(str(z_float))

        Q = self._state.process_noise
        R = self._state.observation_noise

        prior_mean = self._state.mean
        prior_var = self._state.variance + Q

        innov = z - prior_mean
        innov_var = prior_var + R

        K = prior_var / max(innov_var, _EPS)

        post_mean = prior_mean + K * innov
        post_var = (_ONE - K) * prior_var

        innov_std = Decimal(
            str(math.sqrt(float(max(innov_var, _EPS))))
        )

        zscore = innov / max(innov_std, _EPS)

        self._buf.append(
            (z, prior_mean, prior_var, post_mean, post_var)
        )

        self._last_zscore = zscore

        tick = self._state.tick + 1

        self._state = KalmanState(
            mean=post_mean,
            variance=max(post_var, _EPS),
            process_noise=Q,
            observation_noise=R,
            innovation=innov,
            kalman_gain=K,
            tick=tick
        )

        log.info(
            "Kalman update | tick=%d z=%.8f prior_mean=%.8f "
            "post_mean=%.8f variance=%.8f innovation=%.8f "
            "zscore=%.4f K=%.6f",
            tick,
            float(z),
            float(prior_mean),
            float(post_mean),
            float(post_var),
            float(innov),
            float(zscore),
            float(K)
        )

        if tick % self._em_window == 0:
            self._em_update()

        return self._state

    @property
    def state(self) -> KalmanState:
        return self._state

    @property
    def last_innovation_zscore(self) -> Decimal:
        return self._last_zscore

    def _em_update(self) -> None:
        buf = list(self._buf)
        n = Decimal(str(len(buf)))

        r_acc = sum(
            ((z - pm) ** 2 - pv for z, pm, pv, _, _ in buf),
            start=Decimal("0")
        )

        new_R = max(r_acc / n, Decimal("1E-8"))

        q_acc = Decimal("0")

        for i in range(1, len(buf)):
            _, _, pv_prior, pm_cur, pv_cur = buf[i]
            _, _, _, pm_prev, _ = buf[i - 1]

            q_acc += (
                pv_cur
                + (pm_cur - pm_prev) ** 2
                - pv_prior
            )

        new_Q = max(
            q_acc / max(n - _ONE, Decimal("1")),
            Decimal("1E-9")
        )

        log.info(
            "EM update | Q %.2e -> %.2e | R %.2e -> %.2e",
            float(self._state.process_noise),
            float(new_Q),
            float(self._state.observation_noise),
            float(new_R)
        )

        self._state = KalmanState(
            mean=self._state.mean,
            variance=self._state.variance,
            process_noise=new_Q,
            observation_noise=new_R,
            innovation=self._state.innovation,
            kalman_gain=self._state.kalman_gain,
            tick=self._state.tick
        )


# ============================================================================
# Config
# ============================================================================

@dataclass
class SignalGeneratorConfig:

    alpha: Decimal = Decimal("0.30")
    kelly_fraction: Decimal = Decimal("0.25")
    max_position_usd: Decimal = Decimal("10000")
    signal_ttl_seconds: Decimal = Decimal("5")
    cooldown_seconds: Decimal = Decimal("0")
    inventory_buffer: Decimal = Decimal("0.01")
    em_window: int = 50
    init_Q: Decimal = Decimal("1E-5")
    init_R: Decimal = Decimal("1E-4")
    anomaly_zscore_threshold: Decimal = Decimal("3")
    dex_premium_fraction: Decimal = Decimal("0.003")
    dex_discount_fraction: Decimal = Decimal("0.006")

    def __post_init__(self) -> None:
        for f in (
            "alpha",
            "kelly_fraction",
            "max_position_usd",
            "signal_ttl_seconds",
            "cooldown_seconds",
            "inventory_buffer",
            "init_Q",
            "init_R",
            "anomaly_zscore_threshold",
            "dex_premium_fraction",
            "dex_discount_fraction"
        ):
            setattr(self, f, Decimal(str(getattr(self, f))))


# ============================================================================
# Signal Generator
# ============================================================================

class SignalGenerator:

    def __init__(
        self,
        exchange_client,
        pricing_engine,
        inventory_tracker,
        fee_structure: FeeStructure,
        config: Optional[SignalGeneratorConfig] = None,
        dex_price_source=None
    ) -> None:

        self._exchange = exchange_client
        self._pricing = pricing_engine
        self._inventory = inventory_tracker
        self._fees = fee_structure
        self._cfg = config or SignalGeneratorConfig()
        self._dex_price = dex_price_source

        self._filters: dict[str, KalmanSpreadFilter] = {}
        self._last_signal_time: dict[str, Decimal] = {}

    # ---------------------------------------------------------------------

    def generate_from_feed(
        self,
        pair: str,
        size,
        price_state
    ) -> Optional[Signal]:

        size = Decimal(str(size))

        if self._in_cooldown(pair):
            log.info("%s skipped | cooldown active", pair)
            return None

        if not price_state.is_valid:
            log.info("%s skipped | invalid price state", pair)
            return None

        cex_bid = price_state.cex_bid
        cex_ask = price_state.cex_ask
        dex_price = price_state.dex_price

        fee_factor = (
            _ONE
            - Decimal(str(price_state.dex_fee_bps)) / _TEN_THOU
        )

        dex_sell = dex_price * fee_factor
        dex_buy = dex_price / fee_factor

        log.info(
            "[FEED] %s "
            "cex_bid=%.6f "
            "cex_ask=%.6f "
            "dex=%.6f "
            "dex_buy=%.6f "
            "dex_sell=%.6f",
            pair,
            float(cex_bid),
            float(cex_ask),
            float(dex_price),
            float(dex_buy),
            float(dex_sell)
        )

        prices = {
            "cex_bid": cex_bid,
            "cex_ask": cex_ask,
            "dex_buy": dex_buy,
            "dex_sell": dex_sell
        }

        return self._generate_from_prices(pair, size, prices)

    # ---------------------------------------------------------------------

    def _generate_from_prices(
        self,
        pair: str,
        size: Decimal,
        prices: dict
    ) -> Optional[Signal]:

        cex_bid = prices["cex_bid"]
        cex_ask = prices["cex_ask"]

        dex_buy = prices["dex_buy"]
        dex_sell = prices["dex_sell"]

        z_A = self._log_spread(dex_sell, cex_ask)
        z_B = self._log_spread(cex_bid, dex_buy)

        log.info(
            "%s spreads | "
            "z_A=%.8f BUY_CEX_SELL_DEX | "
            "z_B=%.8f BUY_DEX_SELL_CEX",
            pair,
            float(z_A or 0),
            float(z_B or 0)
        )

        if z_A is None and z_B is None:
            log.info("%s rejected | invalid spreads", pair)
            return None

        if (z_A or Decimal("-Inf")) >= (z_B or Decimal("-Inf")):

            best_z = z_A
            direction = Direction.BUY_CEX_SELL_DEX

            ref_cex = cex_ask
            ref_dex = dex_sell

        else:
            best_z = z_B
            direction = Direction.BUY_DEX_SELL_CEX

            ref_cex = cex_bid
            ref_dex = dex_buy

        if best_z is None:
            log.info("%s rejected | no valid spread", pair)
            return None

        log.info(
            "%s selected | direction=%s best_z=%.8f",
            pair,
            direction.name,
            float(best_z)
        )

        kf = self._get_or_create_filter(pair)

        state = kf.update(float(best_z))

        mu = state.mean

        sigma = Decimal(
            str(math.sqrt(float(max(state.variance, _EPS))))
        )

        zscore = kf.last_innovation_zscore

        trade_value = size * ref_cex

        c = self._fees.breakeven_log_spread(trade_value)

        z_stat = (mu - c) / sigma

        confidence = self._normal_cdf(z_stat)

        log.info(
            "%s Bayesian gate | "
            "mu=%.8f sigma=%.8f "
            "threshold=%.8f z_stat=%.8f "
            "confidence=%.6f required=%.6f",
            pair,
            float(mu),
            float(sigma),
            float(c),
            float(z_stat),
            float(confidence),
            float(_ONE - self._cfg.alpha)
        )

        if confidence < (_ONE - self._cfg.alpha):

            log.info(
                "%s rejected | low confidence %.6f < %.6f",
                pair,
                float(confidence),
                float(_ONE - self._cfg.alpha)
            )

            return None

        kelly_size = self._kelly_size(
            mu,
            c,
            sigma,
            ref_cex,
            pair,
            direction,
            size
        )

        log.info(
            "%s Kelly sizing | requested=%.6f actual=%.6f",
            pair,
            float(size),
            float(kelly_size)
        )

        if kelly_size <= _ZERO:
            log.debug("%s rejected | zero Kelly size", pair)
            return None

        actual_trade_value = kelly_size * ref_cex

        exp_mu = Decimal(str(math.exp(float(mu))))

        gross_pnl = (
            (exp_mu - _ONE)
            * actual_trade_value
        )

        fee_bps = self._fees.total_fee_bps(actual_trade_value)

        fee_cost = (
            fee_bps / _TEN_THOU
        ) * actual_trade_value

        net_pnl = gross_pnl - fee_cost

        log.info(
            "%s PnL | gross=%.8f fee_cost=%.8f "
            "net=%.8f trade_value=%.8f",
            pair,
            float(gross_pnl),
            float(fee_cost),
            float(net_pnl),
            float(actual_trade_value)
        )

        if net_pnl <= _ZERO:

            log.info(
                "%s rejected | negative pnl %.8f",
                pair,
                float(net_pnl)
            )

            return None

        inventory_ok = self._check_inventory(
            pair,
            direction,
            kelly_size,
            ref_cex
        )

        if not inventory_ok:
            log.info("%s rejected | insufficient inventory", pair)
            return None

        raw_spread_bps = (
            Decimal(str(math.exp(float(best_z))))
            - _ONE
        ) * _TEN_THOU

        sig = Signal.create(
            pair=pair,
            direction=direction,
            cex_price=ref_cex,
            dex_price=ref_dex,
            raw_spread_bps=raw_spread_bps,
            filtered_spread=mu,
            posterior_variance=state.variance,
            signal_confidence=confidence,
            kelly_size=kelly_size,
            expected_net_pnl=net_pnl,
            ttl_seconds=self._cfg.signal_ttl_seconds,
            inventory_ok=True,
            within_limits=True,
            innovation_zscore=zscore,
            kalman_state=state
        )

        log.info(
            "SIGNAL GENERATED | %s | dir=%s "
            "spread=%.4f bps | confidence=%.4f "
            "size=%.6f | expected_pnl=%.6f",
            pair,
            direction.name,
            float(raw_spread_bps),
            float(confidence),
            float(kelly_size),
            float(net_pnl)
        )

        self._last_signal_time[pair] = Decimal(str(time.time()))

        return sig

    # ---------------------------------------------------------------------

    def _get_or_create_filter(
        self,
        pair: str
    ) -> KalmanSpreadFilter:

        if pair not in self._filters:

            log.info("Creating Kalman filter for %s", pair)

            self._filters[pair] = KalmanSpreadFilter(
                init_Q=self._cfg.init_Q,
                init_R=self._cfg.init_R,
                em_window=self._cfg.em_window
            )

        return self._filters[pair]

    # ---------------------------------------------------------------------

    def _in_cooldown(self, pair: str) -> bool:

        last = self._last_signal_time.get(pair, Decimal("0"))

        delta = Decimal(str(time.time())) - last

        active = delta < self._cfg.cooldown_seconds

        if active:
            log.info(
                "%s cooldown | remaining=%.4f",
                pair,
                float(self._cfg.cooldown_seconds - delta)
            )

        return active

    # ---------------------------------------------------------------------

    @staticmethod
    def _log_spread(
        numerator: Decimal,
        denominator: Decimal
    ) -> Optional[Decimal]:

        if denominator <= _ZERO or numerator <= _ZERO:
            return None

        try:
            return Decimal(
                str(math.log(float(numerator / denominator)))
            )

        except Exception as exc:
            log.warning(f"Failed to determine log spread: {exc}")
            return None

    def _fetch_prices(self, pair: str, size: Decimal) -> Optional[dict[str, Decimal]]:
        """
        Fetch prices from CEX + DEX.

        Priority:
          1. Real DEX quote from dex_price_source
          2. PricingEngine routing quote
          3. CEX mid ± configurable offset (fallback)
        """
        try:
            ob = self._exchange.fetch_order_book(pair)
            cex_bid = Decimal(str(ob["bids"][0][0]))
            cex_ask = Decimal(str(ob["asks"][0][0]))
        except Exception as exc:
            log.info("CEX order book fetch failed for %s: %s", pair, exc)
            return None

        base, quote = pair.split("/")
        mid = (cex_bid + cex_ask) / Decimal("2")

        # --- Try real DEX source first ---
        if self._dex_price is not None:
            try:
                dex_data_buy = self._dex_price.get_dex_quote(base, quote, size)
                dex_data_sell = self._dex_price.get_dex_quote(quote, base, size)

                dex_buy_price = Decimal(str(dex_data_buy.get("price", "0")))
                dex_sell_price = Decimal(str(dex_data_sell.get("price", "0")))

                # dex_buy = how many quote tokens per 1 base (buying base on DEX)
                # dex_sell = how many quote tokens per 1 base (selling base on DEX)
                if dex_buy_price > _ZERO and dex_sell_price > _ZERO:
                    # buy quote: dex_data_buy["price"] = quote per base
                    # sell base: dex_data_sell["price"] = base per quote → invert
                    dex_buy = dex_buy_price   # Price to buy base with quote
                    dex_sell = _ONE / dex_sell_price if dex_sell_price > _ZERO else mid

                    log.info(
                        "%s DEX prices: buy=%.4f sell=%.4f (CEX mid=%.4f)",
                        pair, float(dex_buy), float(dex_sell), float(mid)
                    )
                    return {
                        "cex_bid": cex_bid, "cex_ask": cex_ask,
                        "dex_buy": dex_buy, "dex_sell": dex_sell
                    }
            except Exception as exc:
                log.info("DEX price fetch failed for %s: %s", pair, exc)

        # --- Try PricingEngine ---
        if self._pricing is not None and self._pricing._finder is not None:
            try:
                token_in, token_out = self._pair_to_tokens(pair)
                gas_price = 1
                buy_q = self._pricing.get_quote(
                    token_in, token_out,
                    int(float(size) * 10 ** token_in.decimals),
                    gas_price
                )
                sell_q = self._pricing.get_quote(
                    token_out, token_in,
                    int(float(size) * 10 ** token_out.decimals),
                    gas_price
                )
                dex_buy = Decimal(str(
                    float(buy_q.expected_net) / (float(size) * 10 ** token_out.decimals)
                ))
                dex_sell = Decimal(str(
                    float(sell_q.expected_net) / (float(size) * 10 ** token_in.decimals)
                ))
                return {
                    "cex_bid": cex_bid, "cex_ask": cex_ask,
                    "dex_buy": dex_buy, "dex_sell": dex_sell
                }
            except Exception as exc:
                log.info("PricingEngine quote failed for %s: %s", pair, exc)

        # --- Fallback: mid ± offset ---
        dex_buy = mid * (_ONE + self._cfg.dex_premium_fraction)
        dex_sell = mid * (_ONE + self._cfg.dex_discount_fraction)
        log.info(
            "%s using fallback DEX prices: buy=%.4f sell=%.4f",
            pair, float(dex_buy), float(dex_sell)
        )
        return {
            "cex_bid": cex_bid, "cex_ask": cex_ask,
            "dex_buy": dex_buy, "dex_sell": dex_sell
        }

    def _pair_to_tokens(self, pair: str):
        """Look up Token objects from the pricing engine's pool registry."""
        if self._pricing is None:
            raise NotImplementedError("No pricing engine available")
        base, quote = pair.split("/")
        for pool in self._pricing._pools.values():
            syms = {pool.left.symbol.upper(), pool.right.symbol.upper()}
            if {base.upper(), quote.upper()} <= syms:
                t_base = (
                    pool.left if pool.left.symbol.upper() == base.upper() else pool.right
                )
                t_quote = pool.right if t_base == pool.left else pool.left
                return t_base, t_quote
        raise ValueError(f"No pool found for pair {pair}")

    def _kelly_size(
        self,
        mu: Decimal,
        c: Decimal,
        sigma: Decimal,
        price: Decimal,
        pair: str,
        direction: Direction,
        requested_size: Decimal
    ) -> Decimal:
        if sigma <= _ZERO or price <= _ZERO:
            return _ZERO

        base, quote = pair.split("/")

        try:
            if direction == Direction.BUY_CEX_SELL_DEX:
                # Need quote asset on CEX
                w_usd = Decimal(str(
                    self._inventory.available(Venue.BINANCE, quote) or 0
                ))
            else:
                # Need base asset on DEX wallet
                w_usd = Decimal(str(
                    self._inventory.available(Venue.WALLET, base) or 0
                )) * price
        except Exception:
            w_usd = _ZERO

        w_usd = min(w_usd, self._cfg.max_position_usd)
        if w_usd <= _ZERO:
            return _ZERO

        raw_fraction = self._cfg.kelly_fraction * (mu - c) / max(sigma ** 2, _EPS)
        raw_fraction = min(max(raw_fraction, _ZERO), _ONE)

        kappa = self._inventory_kappa(pair)
        size_usd = raw_fraction * w_usd * kappa
        size_base = size_usd / price
        return min(size_base, requested_size)

    def _inventory_kappa(self, pair: str) -> Decimal:
        lambda_inv = Decimal("2")
        try:
            base, _ = pair.split("/")
            skews = self._inventory.get_skews()
            relevant = [s for s in skews if s.get("asset") == base]
            if not relevant:
                return _ONE
            max_dev = max(
                Decimal(str(s.get("max_deviation_pct", 0))) for s in relevant
            )
            delta = max_dev / Decimal("100")
            kappa_f = math.exp(-float(lambda_inv) * abs(float(delta)))
            return Decimal(str(kappa_f))
        except Exception:
            return _ONE

    def _check_inventory(
        self, pair: str, direction: Direction, size: Decimal, price: Decimal
    ) -> bool:
        buf = _ONE + self._cfg.inventory_buffer
        base, quote = pair.split("/")
        try:
            if direction == Direction.BUY_CEX_SELL_DEX:
                cex_quote_ok = (
                        Decimal(str(
                            self._inventory.available(Venue.BINANCE, quote) or 0
                        ))
                        >= size * price * buf
                )

                dex_base_ok = (
                        Decimal(str(
                            self._inventory.available(Venue.WALLET, base) or 0
                        ))
                        >= size
                )

                return cex_quote_ok and dex_base_ok

            else:
                dex_quote_ok = (
                        Decimal(str(
                            self._inventory.available(Venue.WALLET, quote) or 0
                        ))
                        >= size * price * buf
                )

                cex_base_ok = (
                        Decimal(str(
                            self._inventory.available(Venue.BINANCE, base) or 0
                        ))
                        >= size
                )

                return dex_quote_ok and cex_base_ok
        except Exception as exc:
            raise RuntimeError(f"Failed to check inventory: {exc}")

    # ---------------------------------------------------------------------

    @staticmethod
    def _normal_cdf(z: Decimal) -> Decimal:
        val = 0.5 * math.erfc(-float(z) / math.sqrt(2.0))
        return Decimal(str(val))
