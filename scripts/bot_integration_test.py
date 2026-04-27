"""
Bot integration test: Binance Testnet (CEX) and Ethereum Mainnet (DEX).

Test mode  → builds transactions but does NOT broadcast them.
Production → full execution (requires to be funded wallet and both API keys).

Usage:
    OPERATION_MODE=test python3 -m scripts.real_integration_test
    OPERATION_MODE=production python3 -m scripts.real_integration_test
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("integration")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hr(title: str) -> None:
    log.info("=" * 60)
    log.info("  %s", title)
    log.info("=" * 60)


def _ok(msg: str) -> None:
    log.info("✅  %s", msg)


def _warn(msg: str) -> None:
    log.warning("⚠️   %s", msg)


def _fail(msg: str) -> None:
    log.error("❌  %s", msg)


# ---------------------------------------------------------------------------
# Main integration workflow
# ---------------------------------------------------------------------------

async def run_integration_test() -> bool:
    from config.mode import SystemConfig
    from src.chain.client import ChainClient
    from src.core.wallet import WalletManager
    from src.exchange.dex import DEXPriceSource, DEXExecutor
    from src.pricing.engine import PricingEngine
    from src.pricing.fork_simulator import ForkedChain, TradeSimulator
    from src.executor.engine import Executor, ExecutorConfig
    from src.executor.recovery import SPRTCircuitBreaker, SPRTConfig
    from src.strategy.generator import FeeStructure, SignalGenerator, SignalGeneratorConfig

    cfg = SystemConfig.from_env()
    _hr(f"Integration test | mode={cfg.mode.value}")
    log.info("dry_run=%s", cfg.dex.dry_run)

    all_passed = True

    # ------------------------------------------------------------------
    # 1) Wallet
    # ------------------------------------------------------------------
    _hr("Step 1. Wallet")
    try:
        wallet = WalletManager.from_env()
        _ok(f"Wallet loaded: {wallet.address}")
    except Exception as exc:
        _fail(f"Wallet load failed: {exc}")
        return False

    # ------------------------------------------------------------------
    # 2) Chain connectivity (Ethereum Mainnet)
    # ------------------------------------------------------------------
    _hr("Step 2. Ethereum Mainnet connectivity")
    rpc_url = cfg.dex.rpc_url or os.getenv("ETH_RPC_URL", "")
    if not rpc_url:
        _warn("ETH_RPC_URL not set. Skipping on-chain steps")
        chain_client = None
    else:
        try:
            chain_client = ChainClient([rpc_url])
            chain_id = chain_client.get_chain_id()
            balance = chain_client.get_balance(
                __import__("src.core.types", fromlist=["Address"]).Address.from_string(
                    wallet.address
                )
            )
            _ok(f"Chain ID: {chain_id} | ETH balance: {balance.human:.6f}")
        except Exception as exc:
            _warn(f"Chain connectivity failed: {exc}")
            chain_client = None
            all_passed = False

    # ------------------------------------------------------------------
    # 3) DEX price source
    # ------------------------------------------------------------------
    _hr("Step 3. DEX price source (Uniswap V2 / Ethereum Mainnet)")
    dex_price_source = None
    if chain_client and cfg.dex.pool_addresses:
        try:
            fork_chain = ForkedChain(
                __import__("web3", fromlist=["Web3"]).Web3(
                    __import__("web3", fromlist=["Web3"]).Web3.HTTPProvider(rpc_url)
                )
            )
            simulator = TradeSimulator(fork_chain)
            pricing_engine = PricingEngine(chain_client, simulator, cfg.dex.ws_url)

            dex_price_source = DEXPriceSource(
                pricing_engine=pricing_engine,
                chain_client=chain_client,
                router_address=cfg.dex.router_address,
                pool_addresses=cfg.dex.pool_addresses,
                fee_bps=cfg.dex_swap_bps,
                slippage_bps=cfg.dex.slippage_tolerance_bps,
                gas_price_gwei=20
            )
            dex_price_source.initialize()

            # Test a quote
            quote_data = dex_price_source.get_dex_quote("ETH", "USDT", Decimal("1"))
            if quote_data.get("price", Decimal("0")) > Decimal("0"):
                _ok(f"DEX quote ETH → USDT: price={quote_data['price']:.2f}  "
                    f"impact={quote_data['impact_bps']:.2f}bps")
            else:
                _warn("DEX quote returned zero price (pool may not be configured)")

        except Exception as exc:
            _warn(f"DEX price source init failed: {exc}")
            pricing_engine = None
            dex_price_source = None
    else:
        _warn("No pool addresses configured. DEX price source skipped")
        pricing_engine = None

    # ------------------------------------------------------------------
    # 4) DEX executor (dry-run safe)
    # ------------------------------------------------------------------
    _hr("Step 4. DEX executor")
    dex_executor = None
    if chain_client:
        try:
            dex_executor = DEXExecutor(
                chain_client=chain_client,
                wallet=wallet,
                router_address=cfg.dex.router_address,
                gas_limit_swap=cfg.dex.gas_limit_swap,
                gas_limit_approval=cfg.dex.gas_limit_approval,
                slippage_bps=cfg.dex.slippage_tolerance_bps,
                deadline_seconds=cfg.dex.deadline_seconds,
                dry_run=cfg.dex.dry_run
            )
            _ok(f"DEX executor ready | dry_run={cfg.dex.dry_run}")
        except Exception as exc:
            _warn(f"DEX executor init failed: {exc}")

    # ------------------------------------------------------------------
    # 5) CEX (Binance Testnet or Mainnet)
    # ------------------------------------------------------------------
    _hr(f"Step 5. CEX ({'Testnet' if cfg.cex.sandbox else 'Mainnet'})")
    exchange = None
    try:
        from src.exchange.client import BinanceClient
        exchange = BinanceClient({
            "apiKey": cfg.cex.api_key,
            "secret": cfg.cex.secret,
            "sandbox": cfg.cex.sandbox,
            "options": {"defaultType": cfg.cex.default_type},
            "enableRateLimit": cfg.cex.enable_rate_limit
        })
        ob = exchange.fetch_order_book("ETH/USDT", limit=5)
        best_bid = ob["best_bid"][0]
        best_ask = ob["best_ask"][0]
        _ok(
            f"CEX ETH/USDT: bid={float(best_bid):.2f}  "
            f"ask={float(best_ask):.2f}  "
            f"spread={float(ob['spread_bps']):.2f}bps"
        )
    except Exception as exc:
        _warn(f"CEX connectivity failed: {exc}")
        all_passed = False

    # ------------------------------------------------------------------
    # 6) Signal generator
    # ------------------------------------------------------------------
    _hr("Step 6. Signal generator warm-up")
    from src.inventory.tracker import VenueTracker, Venue

    inventory = VenueTracker([Venue.BINANCE, Venue.WALLET])
    inventory.update_from_cex(
        Venue.BINANCE,
        {
            "ETH": {"free": "10", "locked": "0"},
            "USDT": {"free": "20000", "locked": "0"}
        },
    )
    inventory.update_from_wallet(Venue.WALLET, {"ETH": "5", "USDT": "10000"})

    gen_cfg = SignalGeneratorConfig(
        alpha=Decimal("0.10"),
        kelly_fraction=Decimal(str(cfg.kelly_fraction)),
        max_position_usd=Decimal(str(cfg.max_position_usd)),
        signal_ttl_seconds=Decimal(str(cfg.signal_ttl_seconds)),
        cooldown_seconds=Decimal("0"),  # No cooldown during test
        dex_premium_fraction=Decimal("0.003"),
        dex_discount_fraction=Decimal("0.006")
    )

    generator = SignalGenerator(
        exchange_client=exchange or _make_stub_exchange(),
        pricing_engine=pricing_engine if pricing_engine and hasattr(pricing_engine, "_pools") else None,
        inventory_tracker=inventory,
        fee_structure=FeeStructure(
            cex_taker_bps=Decimal(str(cfg.cex_taker_bps)),
            dex_swap_bps=Decimal(str(cfg.dex_swap_bps)),
            gas_cost_usd=Decimal(str(cfg.gas_cost_usd))
        ),
        config=gen_cfg,
        dex_price_source=dex_price_source
    )

    # Warm up the Kalman filter
    kf = generator._get_or_create_filter("ETH/USDT")
    for _ in range(200):
        kf.update(0.006)

    signal = generator.generate("ETH/USDT", Decimal("0.1"))
    if signal:
        _ok(
            f"Signal generated: direction={signal.direction.value}  "
            f"confidence={float(signal.signal_confidence):.4f}  "
            f"net_pnl=${float(signal.expected_net_pnl):.4f}  "
            f"kelly={float(signal.kelly_size):.6f}"
        )
    else:
        _warn("No signal generated (filter may need more warm-up or spread is too tight)")

    # ------------------------------------------------------------------
    # 7) Executor dry-run
    # ------------------------------------------------------------------
    _hr("Step 7. Executor (dry-run / test mode)")
    if signal:
        exec_cfg = ExecutorConfig(
            leg1_timeout=cfg.executor.leg1_timeout,
            leg2_timeout=cfg.executor.leg2_timeout,
            min_fill_ratio=cfg.executor.min_fill_ratio,
            var_confidence=cfg.executor.var_confidence,
            vol_per_sqrt_second=cfg.executor.vol_per_sqrt_second,
            use_dex_first=cfg.executor.use_dex_first,
            simulation_mode=(exchange is None),
            max_recovery_attempts=cfg.executor.max_recovery_attempts,
            unwind_timeout=cfg.executor.unwind_timeout,
            unwind_slippage_extra_bps=cfg.executor.unwind_slippage_bps
        )

        executor = Executor(
            exchange_client=exchange,
            pricing_engine=pricing_engine if pricing_engine and hasattr(pricing_engine, "_pools") else None,
            inventory_tracker=inventory,
            circuit_breaker=SPRTCircuitBreaker(
                SPRTConfig(p0=Decimal("0.10"), p1=Decimal("0.40"), gamma=Decimal("0.95"))
            ),
            config=exec_cfg,
            dex_price_source=dex_price_source,
            dex_executor=dex_executor
        )

        try:
            ctx = await executor.execute(signal)
            log.info(
                "Execution result: state=%s  pnl=$%s  duration=%.2fs",
                ctx.state.name,
                f"{float(ctx.actual_net_pnl):.4f}" if ctx.actual_net_pnl else "N/A",
                float(ctx.duration())
            )

            if ctx.leg1_raw_tx:
                log.info("Leg1 DEX tx built: to=%s  gas=%s",
                         ctx.leg1_raw_tx.get("to", "?"),
                         ctx.leg1_raw_tx.get("gas", "?"))
            if ctx.leg2_raw_tx:
                log.info("Leg2 DEX tx built: to=%s  gas=%s",
                         ctx.leg2_raw_tx.get("to", "?"),
                         ctx.leg2_raw_tx.get("gas", "?"))

            if ctx.state.name == "DONE":
                _ok("Execution completed successfully")
            else:
                _warn(f"Execution ended with state={ctx.state.name}  error={ctx.error}")

        except Exception as exc:
            _warn(f"Executor error: {exc}")
            all_passed = False
    else:
        _warn("Skipping executor test - no signal available")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _hr("Integration test summary")
    if all_passed:
        _ok("All steps passed")
    else:
        _warn("Some steps failed - check warnings above")

    return all_passed


def _make_stub_exchange():
    """Minimal exchange stub for when CEX credentials are missing."""
    import time
    from unittest.mock import MagicMock
    ex = MagicMock()
    mid = 2000.0
    ex.fetch_order_book.return_value = {
        "symbol": "ETH/USDT",
        "timestamp": int(time.time() * 1000),
        "bids": [(mid - 0.5, 10.0)],
        "asks": [(mid + 0.5, 10.0)],
        "best_bid": (mid - 0.5, 10.0),
        "best_ask": (mid + 0.5, 10.0),
        "mid_price": mid,
        "spread_bps": 5.0
    }
    ex.fetch_balance.return_value = {
        "ETH": {"free": "10", "locked": "0", "total": "10"},
        "USDT": {"free": "20000", "locked": "0", "total": "20000"}
    }
    ex.create_limit_ioc_order.return_value = {
        "id": "stub_001", "status": "filled",
        "amount_filled": 0.1, "avg_fill_price": 2000.0
    }
    return ex


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    success = asyncio.run(run_integration_test())
    sys.exit(0 if success else 1)
