"""
Production-mode arbitrage scenario: ARB/USDC on real Arbitrum One Mainnet.

⚠️ CAREFULLY: REAL FUNDS
---------------------------------
This script broadcasts real transactions to Arbitrum One Mainnet using the
bot's hot wallet (PRIVATE_KEY). Swaps cost real gas and move real tokens.

What this script does
---------------------
1. Reads live pool state on Arbitrum One.
2. Runs pre-flight checks (ETH gas, token balances).
3. Sends approve + swap transactions from the hot wallet.
4. Waits for the ArbBot to detect the CEX/DEX price gap and execute.
5. Optionally performs the reverse swap to trigger a second arb.
6. Posts Discord notifications throughout.

Prerequisites
-------------
  make prod-mode && make run   # Bot running in production mode

Usage
-----
  python3 -m scripts.arb_scenario_production   # Both directions
  python3 -m scripts.arb_scenario_production --direction up   # Buy ARB (price UP)
  python3 -m scripts.arb_scenario_production --direction down   # Sell ARB (price DOWN)
  python3 -m scripts.arb_scenario_production --dry-run   # Build tx, do not send
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from decimal import Decimal
from typing import Optional

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("arb_scenario_prod")

# ── Configuration ─────────────────────────────────────────────────────────────

ARB_RPC = os.getenv("ARBITRUM_RPC_URL", os.getenv("ETH_RPC_URL", "https://arb1.arbitrum.io/rpc"))
POOL_ADDRESS = os.getenv("POOL_ADDRESSES", "0xd65ef54b1ff5d9a452b32ac0c304d1674f761061").split(",")[0].strip()
ROUTER_ADDRESS = os.getenv("UNISWAP_V2_ROUTER", "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24")
ARB_TOKEN = "0x912CE59144191C1204E64559FE8253a0e49E6548"
USDC_TOKEN = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

# $5 of the $100 wallet moves the pool
DEFAULT_AMOUNT = 5   # USDC

ERC20_ABI = [
    {"name": "approve", "type": "function",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"type": "bool"}], "stateMutability": "nonpayable"},
    {"name": "balanceOf", "type": "function",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "allowance", "type": "function",
     "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"},
    {"name": "decimals", "type": "function", "inputs": [],
     "outputs": [{"type": "uint8"}], "stateMutability": "view"}
]
PAIR_ABI = [
    {"name": "getReserves", "type": "function", "inputs": [],
     "outputs": [{"name": "r0", "type": "uint112"}, {"name": "r1", "type": "uint112"},
                 {"name": "ts", "type": "uint32"}], "stateMutability": "view"},
    {"name": "token0", "type": "function", "inputs": [],
     "outputs": [{"type": "address"}], "stateMutability": "view"}
]
ROUTER_ABI = [
    {"name": "swapExactTokensForTokens", "type": "function",
     "inputs": [{"name": "amountIn", "type": "uint256"},
                {"name": "amountOutMin", "type": "uint256"},
                {"name": "path", "type": "address[]"},
                {"name": "to", "type": "address"},
                {"name": "deadline", "type": "uint256"}],
     "outputs": [{"type": "uint256[]"}], "stateMutability": "nonpayable"},
    {"name": "getAmountsOut", "type": "function",
     "inputs": [{"name": "amountIn", "type": "uint256"},
                {"name": "path", "type": "address[]"}],
     "outputs": [{"type": "uint256[]"}], "stateMutability": "view"}
]


# ── Arbitrum session ──────────────────────────────────────────────────────────

class ArbitrumSession:
    """Hot-wallet session on live Arbitrum One."""

    def __init__(self, rpc_url: str, private_key: str) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to {rpc_url}")
        from eth_account import Account
        self._account = Account.from_key(private_key)
        self.address = self._account.address
        chain_id = self.w3.eth.chain_id
        if chain_id != 42161:
            raise RuntimeError(
                f"Expected Arbitrum One (42161), got chain_id={chain_id}. "
                "Check ARBITRUM_RPC_URL."
            )
        log.info("Arbitrum One | chain_id=%d | wallet=%s", chain_id, self.address)

    # ── Balances ──────────────────────────────────────────────────────────────

    def eth_balance(self) -> Decimal:
        return Decimal(self.w3.eth.get_balance(self.address)) / Decimal(10 ** 18)

    def token_balance(self, token_addr: str, decimals: int = 18) -> Decimal:
        c = self.w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_ABI)
        return Decimal(c.functions.balanceOf(self.address).call()) / Decimal(10 ** decimals)

    def token_allowance(self, token_addr: str, spender: str) -> int:
        c = self.w3.eth.contract(address=Web3.to_checksum_address(token_addr), abi=ERC20_ABI)
        return c.functions.allowance(self.address, Web3.to_checksum_address(spender)).call()

    # ── Transaction helpers ───────────────────────────────────────────────────

    def _gas_price(self) -> int:
        blk = self.w3.eth.get_block("latest")
        base = blk.get("baseFeePerGas", 100_000_000)
        return int(base * 1.25) + 1_000_000   # 25 % buffer + 1 gwei tip

    def send(self, built_tx: dict) -> str:
        built_tx.setdefault("nonce", self.w3.eth.get_transaction_count(self.address, "pending"))
        built_tx.setdefault("chainId", 42161)
        built_tx.setdefault("gasPrice", self._gas_price())
        built_tx.pop("from", None)
        signed = self._account.sign_transaction(built_tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash.hex()

    def wait(self, tx_hash: str, timeout: int = 120) -> dict:
        return dict(self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=timeout))

    # ── Contracts ────────────────────────────────────────────────────────────

    def erc20(self, addr: str):
        return self.w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)

    def pair(self):
        return self.w3.eth.contract(address=Web3.to_checksum_address(POOL_ADDRESS), abi=PAIR_ABI)

    def router(self):
        return self.w3.eth.contract(address=Web3.to_checksum_address(ROUTER_ADDRESS), abi=ROUTER_ABI)


# ── Pool helpers ──────────────────────────────────────────────────────────────

def pool_state(session: ArbitrumSession) -> dict:
    r0, r1, _ = session.pair().functions.getReserves().call()
    t0 = session.pair().functions.token0().call().lower()
    if t0 == ARB_TOKEN.lower():
        arb_r, usdc_r = r0, r1
    else:
        usdc_r, arb_r = r0, r1
    arb_h = arb_r / 1e18
    usdc_h = usdc_r / 1e6
    price = usdc_h / arb_h if arb_h else 0.0
    return {"arb": arb_h, "usdc": usdc_h, "price": price, "tvl": usdc_h * 2}


def log_pool(label: str, s: dict) -> None:
    log.info("%s | ARB=%.2f  USDC=%.2f  price=$%.4f  TVL≈$%.0f",
             label, s["arb"], s["usdc"], s["price"], s["tvl"])


# ── Pre-flight ────────────────────────────────────────────────────────────────

def preflight(session: ArbitrumSession, direction: str, amount: int) -> None:
    eth = session.eth_balance()
    arb = session.token_balance(ARB_TOKEN, 18)
    usdc = session.token_balance(USDC_TOKEN, 6)
    log.info("Wallet | ETH=%.5f  ARB=%.4f  USDC=%.4f", eth, arb, usdc)

    if eth < Decimal("0.001"):
        raise RuntimeError(
            f"Insufficient ETH for gas: {eth:.5f}. Need ≥0.001 ETH on Arbitrum One."
        )
    if direction in ("up", "both") and usdc < Decimal(str(amount * 0.99)):
        raise RuntimeError(
            f"Insufficient USDC: have {usdc:.2f}, need ~{amount}. "
            "Fund the wallet on Arbitrum One."
        )
    if direction in ("down", "both"):
        state = pool_state(session)
        arb_needed = amount / max(state["price"], 0.001)
        if float(arb) < arb_needed * 0.80:
            raise RuntimeError(
                f"Insufficient ARB: have {arb:.2f}, need ~{arb_needed:.2f}. "
                "Fund the wallet on Arbitrum One."
            )
    log.info("Pre-flight OK ✅")


# ── Swap ──────────────────────────────────────────────────────────────────────

def do_swap(
    session: ArbitrumSession,
    token_in: str,
    token_out: str,
    amount_in: int,
    label: str,
    dry_run: bool,
    slippage_bps: int = 3000
) -> Optional[str]:
    """Approve + swap. Returns tx hash or None (dry-run)."""
    router_cs = Web3.to_checksum_address(ROUTER_ADDRESS)

    # Approve if needed
    allowance = session.token_allowance(token_in, ROUTER_ADDRESS)
    if allowance < amount_in:
        approve_tx = session.erc20(token_in).functions.approve(
            router_cs, 2 ** 256 - 1
        ).build_transaction({"from": session.address, "gas": 100_000})
        if dry_run:
            log.info("[DRY-RUN] Would approve %s", token_in[:10])
        else:
            h = session.send(approve_tx)
            rec = session.wait(h)
            log.info("Approve %s | tx=%s  status=%s", token_in[:10], h,
                     "OK" if rec["status"] else "FAIL")

    # Build swap
    path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
    try:
        out = session.router().functions.getAmountsOut(amount_in, path).call()
        min_out = int(out[-1] * (10_000 - slippage_bps) // 10_000)
    except Exception:
        min_out = 1

    swap_tx = session.router().functions.swapExactTokensForTokens(
        amount_in, min_out, path, session.address, int(time.time()) + 300
    ).build_transaction({"from": session.address, "gas": 500_000})

    if dry_run:
        log.info("[DRY-RUN] %s | token_in=%s  amount=%d  min_out=%d",
                 label, token_in[:10], amount_in, min_out)
        return None

    h = session.send(swap_tx)
    log.info("%s | tx=https://arbiscan.io/tx/%s", label, h)
    rec = session.wait(h, timeout=120)
    log.info("%s confirmed | status=%s  gas=%d", label,
             "OK" if rec["status"] else "FAIL", rec["gasUsed"])
    return h


# ── Discord helper ────────────────────────────────────────────────────────────

async def notify(msg: str, critical: bool = False) -> None:
    try:
        from src.safety.discord_notifier import DiscordAlerter
        alerter = DiscordAlerter.from_env()
        if critical:
            await alerter.critical(msg)
        else:
            await alerter.info(msg)
    except Exception as exc:
        log.debug("Discord notify failed: %s", exc)


# ── Main scenario ─────────────────────────────────────────────────────────────

async def run_scenario(direction: str, amount: int, dry_run: bool) -> None:
    log.info("=" * 65)
    log.info("ARB/USDC PRODUCTION Arb Scenario | direction=%s  amount=%d  dry_run=%s",
             direction, amount, dry_run)
    log.info("=" * 65)

    private_key = os.getenv("PRIVATE_KEY", "")
    if not private_key:
        log.error("PRIVATE_KEY not set. Aborting.")
        sys.exit(1)

    session = ArbitrumSession(ARB_RPC, private_key)
    preflight(session, direction, amount)

    before = pool_state(session)
    log_pool("POOL BEFORE", before)

    dr_tag = "[DRY-RUN] " if dry_run else "⚡LIVE "
    await notify(
        f"📊 **Production Arb Scenario** {dr_tag}\n"
        f"Pool: ARB={before['arb']:.2f}  USDC={before['usdc']:.2f}  "
        f"price=${before['price']:.4f}  TVL≈${before['tvl']:.0f}\n"
        f"Wallet: {session.address}"
    )

    usdc_raw = amount * 10 ** 6

    # ── Phase 1: buy ARB → price UP on DEX → bot fires BUY_CEX_SELL_DEX ─────
    if direction in ("up", "both"):
        log.info("--- Phase 1: Buying ARB with %d USDC (DEX price UP) ---", amount)
        do_swap(session, USDC_TOKEN, ARB_TOKEN, usdc_raw, "BUY_ARB", dry_run)
        after = pool_state(session)
        log_pool("POOL AFTER BUY_ARB", after)
        pct = (after["price"] - before["price"]) / max(before["price"], 1e-9) * 100
        log.info("Price impact: %+.1f %%", pct)
        await notify(
            f"🚀 **Price UP** on Arbitrum One {dr_tag}(+{pct:.1f}%)\n"
            f"${before['price']:.4f} → ${after['price']:.4f}\n"
            f"Bot should fire **BUY_CEX_SELL_DEX** now..."
        )
        log.info("Waiting 8s for bot to react...")
        await asyncio.sleep(8)

    # ── Phase 2: sell ARB → price DOWN on DEX → bot fires BUY_DEX_SELL_CEX ──
    if direction in ("down", "both"):
        current = pool_state(session)
        arb_price = current["price"] or 0.5
        arb_raw = int((amount / arb_price) * 1e18)
        log.info("--- Phase 2: Selling %.4f ARB → USDC (DEX price DOWN) ---",
                 arb_raw / 1e18)
        do_swap(session, ARB_TOKEN, USDC_TOKEN, arb_raw, "SELL_ARB", dry_run)
        after = pool_state(session)
        log_pool("POOL AFTER SELL_ARB", after)
        pct = (after["price"] - current["price"]) / max(current["price"], 1e-9) * 100
        log.info("Price impact: %+.1f %%", pct)
        await notify(
            f"📉 **Price DOWN** on Arbitrum One {dr_tag}({pct:.1f}%)\n"
            f"${current['price']:.4f} → ${after['price']:.4f}\n"
            f"Bot should fire **BUY_DEX_SELL_CEX** now..."
        )
        log.info("Waiting 8s for bot to react...")
        await asyncio.sleep(8)

    final = pool_state(session)
    log_pool("POOL FINAL", final)
    await notify(
        f"✅ **Production Scenario COMPLETE** {dr_tag}\n"
        f"Final: ARB={final['arb']:.2f}  USDC={final['usdc']:.2f}  "
        f"price=${final['price']:.4f}\n"
        f"Check ArbBot logs for execution results."
    )
    log.info("Scenario complete - check bot logs.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production ARB/USDC arbitrage scenario on Arbitrum One",
        prog="python3 -m scripts.arb_scenario_production"
    )
    parser.add_argument(
        "--direction", choices=["up", "down", "both"], default="both",
        help="Price direction (default: both)"
    )
    parser.add_argument(
        "--amount", type=int, default=DEFAULT_AMOUNT,
        help=f"USDC equivalent to swap (default: {DEFAULT_AMOUNT})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Build transactions but do NOT broadcast"
    )
    args = parser.parse_args()

    if not args.dry_run:
        print(
            "\n⚠️ WARNING: REAL FUNDS on Arbitrum One Mainnet.\n"
            f"   Direction : {args.direction}\n"
            f"   Amount    : {args.amount} USDC\n"
            "   Press Ctrl-C within 15 seconds to abort.\n"
        )
        try:
            time.sleep(15)
        except KeyboardInterrupt:
            print("Aborted.")
            sys.exit(0)

    asyncio.run(run_scenario(
        direction=args.direction,
        amount=args.amount,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()
