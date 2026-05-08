"""
Demo-mode arbitrage scenario: ARB/USDC on Anvil fork of Arbitrum One.

What this script does
---------------------
1. Connects to the local Anvil fork (ANVIL_RPC_URL).
2. Reads the current ARB/USDC Uniswap V2 pool state.
3. Impersonates a whale and swaps a large amount of USDC → ARB (price UP)
   or ARB → USDC (price DOWN), moving the pool price by 30-60 %.
4. Waits for the bot's DEX feed to detect the gap.
5. The running ArbBot fires a CEX-DEX arbitrage automatically.
6. Optionally performs the reverse swap to trigger a second arb.
7. Sends Discord notifications at each step.

Prerequisites
-------------
  make run-demo   # Bot + Anvil fork running
  make fund-demo   # Wallet funded with test tokens

Usage
-----
  python3 -m scripts.arb_scenario_demo
  python3 -m scripts.arb_scenario_demo --direction up --amount 200
  python3 -m scripts.arb_scenario_demo --direction down --amount 200
  python3 -m scripts.arb_scenario_demo --direction both
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time

from dotenv import load_dotenv
from web3 import Web3

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("arb_scenario_demo")

# ── Configuration ─────────────────────────────────────────────────────────────

ANVIL_RPC = os.getenv("ANVIL_RPC_URL", "http://anvil:8545")
POOL_ADDRESS = os.getenv("POOL_ADDRESSES", "0xd65ef54b1ff5d9a452b32ac0c304d1674f761061").split(",")[0].strip()
ROUTER_ADDRESS = os.getenv("UNISWAP_V2_ROUTER", "0x4752ba5dbc23f44d87826276bf6fd6b1c372ad24")
ARB_TOKEN = "0x912CE59144191C1204E64559FE8253a0e49E6548"
USDC_TOKEN = "0xaf88d065e77c8cC2239327C5EDb3A432268e5831"

# Whales that hold large balances on Arbitrum One mainnet
USDC_WHALE = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"
ARB_WHALE = "0x912CE59144191C1204E64559FE8253a0e49E6548"

# Pool has ~$500 TVL → $200 USDC swap gives 30-50 % price impact
DEFAULT_AMOUNT = 200   # USDC

ERC20_ABI = [
    {"name": "approve", "type": "function",
     "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}],
     "outputs": [{"type": "bool"}], "stateMutability": "nonpayable"},
    {"name": "balanceOf", "type": "function",
     "inputs": [{"name": "account", "type": "address"}],
     "outputs": [{"type": "uint256"}], "stateMutability": "view"}
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


# ── Anvil session ─────────────────────────────────────────────────────────────

class AnvilSession:
    def __init__(self, rpc_url: str) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to Anvil at {rpc_url}")
        log.info("Anvil connected | chain_id=%d", self.w3.eth.chain_id)

    def impersonate(self, addr: str) -> None:
        self.w3.provider.make_request("anvil_impersonateAccount", [addr])
        self.w3.provider.make_request("anvil_setBalance", [addr, hex(10 ** 20)])

    def stop_impersonate(self, addr: str) -> None:
        try:
            self.w3.provider.make_request("anvil_stopImpersonatingAccount", [addr])
        except Exception:
            pass

    def mine(self) -> None:
        self.w3.provider.make_request("evm_mine", [])

    def erc20(self, addr: str):
        return self.w3.eth.contract(address=Web3.to_checksum_address(addr), abi=ERC20_ABI)

    def pair(self):
        return self.w3.eth.contract(address=Web3.to_checksum_address(POOL_ADDRESS), abi=PAIR_ABI)

    def router(self):
        return self.w3.eth.contract(address=Web3.to_checksum_address(ROUTER_ADDRESS), abi=ROUTER_ABI)


# ── Pool helpers ──────────────────────────────────────────────────────────────

def pool_state(session: AnvilSession) -> dict:
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


# ── Swap via impersonation ────────────────────────────────────────────────────

def whale_swap(
    session: AnvilSession,
    token_in: str,
    token_out: str,
    amount_in: int,
    whale: str,
    label: str,
    slippage_bps: int = 5000
) -> str:
    """Execute a swap from *token_in* to *token_out* as *whale* via impersonation."""
    whale_cs = Web3.to_checksum_address(whale)
    router = session.router()
    tok_in = session.erc20(token_in)

    session.impersonate(whale)

    # Approve router
    tok_in.functions.approve(
        Web3.to_checksum_address(ROUTER_ADDRESS), amount_in
    ).transact({"from": whale_cs, "gas": 120_000})

    # Estimate output
    path = [Web3.to_checksum_address(token_in), Web3.to_checksum_address(token_out)]
    try:
        out = router.functions.getAmountsOut(amount_in, path).call()
        min_out = int(out[-1] * (10_000 - slippage_bps) // 10_000)
    except Exception:
        min_out = 1

    deadline = int(time.time()) + 300
    tx_hash = router.functions.swapExactTokensForTokens(
        amount_in, min_out, path, whale_cs, deadline
    ).transact({"from": whale_cs, "gas": 600_000})

    session.stop_impersonate(whale)
    session.mine()

    receipt = session.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
    status = "OK" if receipt.status else "FAIL"
    log.info("%s | tx=%s  status=%s  gas=%d", label, tx_hash.hex(), status, receipt.gasUsed)
    return tx_hash.hex()


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

async def run_scenario(direction: str, amount: int) -> None:
    log.info("=" * 62)
    log.info("ARB/USDC Demo Arb Scenario | direction=%s  amount=%d USDC", direction, amount)
    log.info("=" * 62)

    session = AnvilSession(ANVIL_RPC)
    before = pool_state(session)
    log_pool("POOL BEFORE", before)

    await notify(
        f"📊 **Demo Arb Scenario START**\n"
        f"Pool: ARB={before['arb']:.2f}  USDC={before['usdc']:.2f}  "
        f"price=${before['price']:.4f}  TVL≈${before['tvl']:.0f}"
    )

    usdc_raw = amount * 10 ** 6

    # ── Phase 1: buy ARB with USDC → price UP on DEX ─────────────────────────
    # Gap: DEX price > CEX → bot fires BUY_CEX_SELL_DEX
    if direction in ("up", "both"):
        log.info("--- Phase 1: Buying ARB with %d USDC (DEX price UP) ---", amount)
        whale_swap(session, USDC_TOKEN, ARB_TOKEN, usdc_raw, USDC_WHALE, "BUY_ARB")
        after = pool_state(session)
        log_pool("POOL AFTER BUY_ARB", after)
        pct = (after["price"] - before["price"]) / max(before["price"], 1e-9) * 100
        log.info("Price impact: %+.1f %%", pct)
        await notify(
            f"🚀 **Price PUSHED UP** on Anvil fork (+{pct:.1f}%)\n"
            f"Swapped {amount} USDC → ARB  |  "
            f"${before['price']:.4f} → ${after['price']:.4f}\n"
            f"Bot should fire **BUY_CEX_SELL_DEX** now…"
        )
        log.info("Waiting 6s for bot to react...")
        await asyncio.sleep(6)

    # ── Phase 2: sell ARB for USDC → price DOWN on DEX ───────────────────────
    # Gap: DEX price < CEX → bot fires BUY_DEX_SELL_CEX
    if direction in ("down", "both"):
        current = pool_state(session)
        arb_price = current["price"] or 0.5
        arb_to_sell = int((amount / arb_price) * 1e18)
        log.info(
            "--- Phase 2: Selling %.2f ARB → USDC (DEX price DOWN) ---",
            arb_to_sell / 1e18,
        )
        whale_swap(session, ARB_TOKEN, USDC_TOKEN, arb_to_sell, ARB_WHALE, "SELL_ARB")
        after = pool_state(session)
        log_pool("POOL AFTER SELL_ARB", after)
        pct = (after["price"] - current["price"]) / max(current["price"], 1e-9) * 100
        log.info("Price impact: %+.1f %%", pct)
        await notify(
            f"📉 **Price PUSHED DOWN** on Anvil fork ({pct:.1f}%)\n"
            f"Swapped {arb_to_sell / 1e18:.2f} ARB → USDC  |  "
            f"${current['price']:.4f} → ${after['price']:.4f}\n"
            f"Bot should fire **BUY_DEX_SELL_CEX** now..."
        )
        log.info("Waiting 6s for bot to react…")
        await asyncio.sleep(6)

    final = pool_state(session)
    log_pool("POOL FINAL", final)
    await notify(
        f"✅ **Demo Scenario COMPLETE**\n"
        f"Final: ARB={final['arb']:.2f}  USDC={final['usdc']:.2f}  "
        f"price=${final['price']:.4f}\n"
        f"Check ArbBot logs for execution results."
    )
    log.info("Scenario complete - check bot logs for arb results.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo ARB/USDC arbitrage scenario on Anvil fork",
        prog="python3 -m scripts.arb_scenario_demo"
    )
    parser.add_argument(
        "--direction", choices=["up", "down", "both"], default="both",
        help="Price direction to push (default: both)"
    )
    parser.add_argument(
        "--amount", type=int, default=DEFAULT_AMOUNT,
        help=f"USDC equivalent to swap (default: {DEFAULT_AMOUNT})"
    )
    args = parser.parse_args()
    asyncio.run(run_scenario(direction=args.direction, amount=args.amount))


if __name__ == "__main__":
    main()
