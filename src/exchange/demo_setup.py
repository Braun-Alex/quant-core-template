"""
Demo mode: Binance Demo Trading and Anvil fork of Arbitrum Mainnet.

This module wires together:
  - BinanceDemoClient         → real-time mainnet CEX orderbook, virtual fills
  - AnvilInventoryProvisioner → test tokens on Arbitrum fork via cheatcodes
  - LiveOrderBook             → Binance Demo WS depth stream
  - DEXPriceFeed              → Arbitrum fork reserve polling
  - ARB/USDC pair             → Arbitrum-native token, exploitable CEX-DEX spreads

Three modes compared
--------------------
  Testnet    → synthetic CEX book and Sepolia DEX: unrealistic, no arbitrage
  Demo       → real CEX book (Demo Trading) and Anvil fork: most realistic
  Production → real CEX book (Mainnet) and Arbitrum Mainnet: real risk

Why ARB/USDC for arbitrage?
----------------------------
  • ARB is Arbitrum's native governance token
  • Price discovery happens primarily on Arbitrum DEXes (Uniswap, Camelot)
  • CEX (Binance) lags DEX by 1-5 seconds on high-volatility events
  • Uniswap V2 pool on Arbitrum has $2-10M TVL → slippage is real but manageable
  • Typical exploitable spread: 10-80 bps (vs ~5 bps for ETH/USDC)
  • Gas cost on Arbitrum: ~$0.01-$0.05

Demo setup steps
----------------
1. Register at https://demo-trading.binance.com (free, instant)
   → Get BINANCE_DEMO_API_KEY and BINANCE_DEMO_SECRET
2. Start Anvil fork:
   anvil --fork-url $ARBITRUM_RPC_URL --port 8545 --chain-id 42161
3. Run provisioner to fund wallet:
   python3 -m src.exchange.demo_setup fund --wallet $WALLET_ADDRESS
4. Run bot in demo mode:
   OPERATION_MODE=demo python3 -m scripts.arb_bot

Environment variables
---------------------
  OPERATION_MODE=demo
  BINANCE_DEMO_API_KEY
  BINANCE_DEMO_SECRET
  BINANCE_DEMO_REST_URL   (default https://demo-trading.binance.com)
  BINANCE_DEMO_WS_URL   (default wss://demo-trading.binance.com/ws)
  ANVIL_RPC_URL   (default http://localhost:8545)
  ANVIL_FORK_URL   ($ARBITRUM_RPC_URL for forking)
  DEMO_WALLET_ADDRESS   (hot wallet address)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from decimal import Decimal
from dotenv import load_dotenv

from config.mode import BinanceTradingRules
from src.exchange.feed import LiveOrderBook

load_dotenv()
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Demo-mode constants
# ---------------------------------------------------------------------------

DEMO_TRADING_PAIR = "ARB/USDC"
DEMO_TRADE_SIZE_BASE = 0.001   # Start very small

# ARB/USDC Uniswap V2 pool on Arbitrum One
DEMO_POOL_ADDRESS = "0xd65ef54b1ff5d9a452b32ac0c304d1674f761061"

# Binance Demo Trading endpoints
DEMO_WS_BASE_URL = os.getenv(
    "BINANCE_DEMO_WS_URL", "wss://demo-trading.binance.com/ws"
)
DEMO_REST_BASE_URL = os.getenv(
    "BINANCE_DEMO_REST_URL", "https://demo-trading.binance.com"
)

# Anvil fork
ANVIL_RPC_URL = os.getenv("ANVIL_RPC_URL", "http://localhost:8545")
ANVIL_FORK_URL = os.getenv("ANVIL_FORK_URL", os.getenv("ARBITRUM_RPC_URL", ""))


# ---------------------------------------------------------------------------
# Binance Demo Trading rules for ARB/USDC
# ---------------------------------------------------------------------------

ARB_USDC_TRADING_RULES = BinanceTradingRules(
    pair="ARB/USDC",
    min_notional_usd=5.0,   # Binance minimum order value
    lot_size_step=0.1,   # ARB step size (0.1 ARB)
    price_tick=0.0001,   # $0.0001 price tick for ARB/USDC
    min_quantity=0.1   # 0.1 ARB minimum
)


# ---------------------------------------------------------------------------
# Demo mode SystemConfig factory
# ---------------------------------------------------------------------------

def build_demo_config():
    """
    Build a SystemConfig for demo mode:
      - CEX: Binance Demo Trading
      - DEX: Anvil fork of Arbitrum (localhost:8545)
      - Pair: ARB/USDC
    """
    from config.mode import (
        SystemConfig, OperationMode, DEXConfig, CEXConfig,
        ExecutorSettings, RiskConfig, ARBITRUM_ONE
    )

    dex = DEXConfig(
        rpc_url=ANVIL_RPC_URL,
        ws_url="",
        chain_id=42161,   # Same chain ID as Arbitrum (fork preserves it)
        router_address=ARBITRUM_ONE.router_address,
        factory_address=ARBITRUM_ONE.factory_address,
        pool_addresses=[DEMO_POOL_ADDRESS],
        gas_limit_swap=500_000,
        gas_limit_approval=100_000,
        slippage_tolerance_bps=Decimal("100"),   # More lenient for fork
        deadline_seconds=300,
        dry_run=False   # Execute real txs against the fork
    )

    cex = CEXConfig(
        api_key=os.getenv("BINANCE_DEMO_API_KEY", ""),
        secret=os.getenv("BINANCE_DEMO_SECRET", ""),
        sandbox=False,   # Demo Trading uses mainnet-style API, not sandbox flag
        enable_rate_limit=True,
        default_type="spot",
        trading_rules=ARB_USDC_TRADING_RULES
    )

    executor = ExecutorSettings(
        leg1_timeout=Decimal("5"),
        leg2_timeout=Decimal("30"),   # Fork is local → faster
        min_fill_ratio=Decimal("0.80"),
        var_confidence=Decimal("0.95"),
        vol_per_sqrt_second=Decimal("0.001"),   # ARB is more volatile than ETH
        use_dex_first=True,
        max_recovery_attempts=2,
        unwind_timeout=Decimal("15"),
        unwind_slippage_bps=Decimal("200")
    )

    risk = RiskConfig(
        max_trade_usd=10.0,   # Very conservative for demo
        max_trade_pct=0.05,
        max_position_per_token=20.0,
        max_open_positions=1,
        max_loss_per_trade=2.0,
        max_daily_loss=10.0,
        max_drawdown_pct=0.15,
        max_trades_per_hour=30,
        consecutive_loss_limit=4,
        max_spread_bps=500.0,
        max_signal_age_seconds=5.0,
        initial_capital=1000.0   # Notional demo capital
    )

    return SystemConfig(
        mode=OperationMode.TEST,   # Treated as test internally
        dex=dex,
        cex=cex,
        executor=executor,
        risk=risk,
        signal_ttl_seconds=Decimal("5"),
        cooldown_seconds=Decimal("1"),   # Faster cooldown for demo
        kelly_fraction=Decimal("0.15"),   # More conservative for volatile ARB
        max_position_usd=Decimal("500"),
        cex_taker_bps=Decimal("10"),
        dex_swap_bps=Decimal("30"),
        gas_cost_usd=Decimal("0.05"),   # Anvil fork: minimal gas
        trading_pair=DEMO_TRADING_PAIR,
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        log_dir="logs",
        dry_run=False
    )


# ---------------------------------------------------------------------------
# LiveOrderBook override for Demo Trading WS
# ---------------------------------------------------------------------------

def make_demo_order_book(symbol: str = DEMO_TRADING_PAIR) -> "LiveOrderBook":
    """
    Create a LiveOrderBook connected to Binance Demo Trading WS.

    The Demo Trading WS endpoint mirrors the real Binance order book in
    real time - you get the same depth stream as production, just fills
    are virtual.
    """
    import os
    from src.exchange.feed import LiveOrderBook

    # Patch env vars so LiveOrderBook picks up the demo WS URLs
    os.environ["BINANCE_MAINNET_WS"] = DEMO_WS_BASE_URL
    os.environ["BINANCE_MAINNET_REST"] = DEMO_REST_BASE_URL

    # Use testnet=False so it uses MAINNET (which we've overridden to demo)
    return LiveOrderBook(
        symbol=symbol,
        testnet=False,   # Use the overridden mainnet = demo URLs
        max_depth=20,
        reconnect=True
    )


# ---------------------------------------------------------------------------
# Anvil fork management
# ---------------------------------------------------------------------------

async def start_anvil_fork(
    fork_url: str = ANVIL_FORK_URL,
    port: int = 8545,
    block_time: int = 1,
) -> asyncio.subprocess.Process:
    """
    Start an Anvil fork of Arbitrum Mainnet programmatically.
    Blocks until the RPC is responsive.

    Requires `anvil` binary in PATH (part of Foundry).
    """
    if not fork_url:
        raise ValueError(
            "ARBITRUM_RPC_URL must be set to fork Arbitrum Mainnet with Anvil. "
            "Get a free endpoint at https://alchemy.com or https://infura.io"
        )

    cmd = [
        "anvil",
        "--fork-url", fork_url,
        "--port", str(port),
        "--chain-id", "42161",
        "--block-time", str(block_time),
        "--host", "0.0.0.0",
        "--silent"
    ]

    log.info("Starting Anvil fork of Arbitrum | port=%d", port)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    # Wait until RPC is responsive (max 30s)
    from src.chain.client import ChainClient
    for attempt in range(30):
        await asyncio.sleep(1.0)
        try:
            client = ChainClient([f"http://localhost:{port}"], max_retries=1)
            cid = client.get_chain_id()
            log.info("Anvil fork ready | chain_id=%d port=%d", cid, port)
            return proc
        except Exception:
            pass

    proc.terminate()
    raise RuntimeError(f"Anvil fork did not start within 30s (port {port})")


def provision_demo_wallet(
    wallet_address: str,
    rpc_url: str = ANVIL_RPC_URL,
    tokens: list[str] | None = None
) -> dict[str, bool]:
    """
    Fund wallet with test tokens on the Anvil fork.
    Synchronous wrapper for use in scripts.
    """
    from src.exchange.demo_inventory import AnvilInventoryProvisioner
    p = AnvilInventoryProvisioner(rpc_url=rpc_url, wallet_address=wallet_address)
    results = p.fund_demo_wallet(tokens=tokens)

    # Print balance verification
    balances = p.verify_balances(tokens=tokens or ["ARB", "USDC", "WETH", "ETH"])
    log.info("Demo wallet balances after provisioning:")
    for sym, bal in balances.items():
        log.info("  %-6s %.4f", sym, float(bal))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Demo mode setup utilities",
        prog="python3 -m src.exchange.demo_setup"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Fund subcommand
    fund_p = sub.add_parser("fund", help="Fund demo wallet on Anvil fork")
    fund_p.add_argument("--wallet", required=True, help="Wallet address to fund")
    fund_p.add_argument("--rpc", default=ANVIL_RPC_URL, help="Anvil RPC URL")
    fund_p.add_argument(
        "--tokens", default="ARB,USDC,WETH",
        help="Comma-separated token symbols to fund",
    )

    # Check subcommand
    check_p = sub.add_parser("check", help="Check demo wallet balances")
    check_p.add_argument("--wallet", required=True)
    check_p.add_argument("--rpc", default=ANVIL_RPC_URL)

    # Pool subcommand
    sub.add_parser("pool", help="Print ARB/USDC pool info from fork")

    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if args.cmd == "fund":
        tokens = [t.strip().upper() for t in args.tokens.split(",")]
        results = provision_demo_wallet(
            wallet_address=args.wallet,
            rpc_url=args.rpc,
            tokens=tokens
        )
        ok = sum(1 for v in results.values() if v)
        print(f"\nFunded {ok}/{len(results)} tokens successfully")
        for sym, success in results.items():
            print(f"  {'✅' if success else '❌'}  {sym}")
        return 0 if ok > 0 else 1

    if args.cmd == "check":
        from src.exchange.demo_inventory import AnvilInventoryProvisioner
        p = AnvilInventoryProvisioner(rpc_url=args.rpc, wallet_address=args.wallet)
        balances = p.verify_balances()
        print(f"\nBalances for {args.wallet}:")
        for sym, bal in balances.items():
            print(f"  {sym:6s}  {float(bal):.4f}")
        return 0

    if args.cmd == "pool":
        from src.chain.client import ChainClient
        from src.pricing.amm import PoolState
        from src.core.types import Address
        client = ChainClient([ANVIL_RPC_URL])
        try:
            pool = PoolState.load(Address(DEMO_POOL_ADDRESS), client)
            print(f"\nARB/USDC Pool on fork ({DEMO_POOL_ADDRESS[:10]}...):")
            print(f"  {pool.left.symbol:<6}  {pool.qty_left / 10**pool.left.decimals:,.2f}")
            print(f"  {pool.right.symbol:<6}  {pool.qty_right / 10**pool.right.decimals:,.2f}")
        except Exception as exc:
            print(f"Failed to load pool: {exc}")
            return 1
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(_run_cli())
