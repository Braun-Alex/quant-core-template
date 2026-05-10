"""
Demo-mode arbitrage scenario: ARB/USDC.e on Anvil fork of Arbitrum One.

Bypass the Uniswap V2 router.

Direct pair swap protocol (Uniswap V2 low-level):
  1. Transfer tokenIn whale → pair contract (pair must hold tokenIn first)
  2. Call pair.swap(amount0Out, amount1Out, recipient, b"")
  3. Mine a block - pair emits Sync event, reserves update

This guarantees the swap happens on exactly POOL_ADDRESS.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os

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
POOL_ADDRESS = os.getenv(
    "POOL_ADDRESSES", "0xd65ef54b1ff5d9a452b32ac0c304d1674f761061"
).split(",")[0].strip()

ARB_TOKEN = "0x912CE59144191C1204E64559FE8253a0e49E6548"

# USDC.e (bridged) — the token actually in the target pool
USDC_TOKEN = "0xFF970A61A04b1cA14834A43f5dE4533eBDDB5CC8"

# Arbitrum Foundation reserve - large ARB holder
ARB_WHALE = "0xF3FC178157fb3c87548bAA86F9d24BA38E649B58"

DEFAULT_AMOUNT = 1   # USDC.e

# ── ABIs ──────────────────────────────────────────────────────────────────────

ERC20_ABI = [
    {
        "name": "transfer",
        "type": "function",
        "inputs": [
            {"name": "to", "type": "address"},
            {"name": "amount", "type": "uint256"}
        ],
        "outputs": [{"type": "bool"}],
        "stateMutability": "nonpayable"
    },
    {
        "name": "balanceOf",
        "type": "function",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"type": "uint256"}],
        "stateMutability": "view"
    }
]

PAIR_ABI = [
    {
        "name": "getReserves",
        "type": "function",
        "inputs": [],
        "outputs": [
            {"name": "reserve0", "type": "uint112"},
            {"name": "reserve1", "type": "uint112"},
            {"name": "blockTimestampLast", "type": "uint32"}
        ],
        "stateMutability": "view"
    },
    {
        "name": "token0",
        "type": "function",
        "inputs": [],
        "outputs": [{"type": "address"}],
        "stateMutability": "view"
    },
    {
        "name": "token1",
        "type": "function",
        "inputs": [],
        "outputs": [{"type": "address"}],
        "stateMutability": "view"
    },
    {
        "name": "swap",
        "type": "function",
        "inputs": [
            {"name": "amount0Out", "type": "uint256"},
            {"name": "amount1Out", "type": "uint256"},
            {"name": "to", "type": "address"},
            {"name": "data", "type": "bytes"}
        ],
        "outputs": [],
        "stateMutability": "nonpayable"
    }
]


# ── Anvil session ─────────────────────────────────────────────────────────────

class AnvilSession:
    def __init__(self, rpc_url: str) -> None:
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise RuntimeError(f"Cannot connect to Anvil at {rpc_url}")
        log.info("Anvil connected | chain_id=%d", self.w3.eth.chain_id)

        pair_cs = Web3.to_checksum_address(POOL_ADDRESS)
        p = self.w3.eth.contract(address=pair_cs, abi=PAIR_ABI)
        self._token0 = p.functions.token0().call().lower()
        self._token1 = p.functions.token1().call().lower()
        log.info(
            "Pair %s | token0=%s  token1=%s",
            POOL_ADDRESS[:10], self._token0[:10], self._token1[:10]
        )

    @property
    def token0(self) -> str:
        return self._token0

    @property
    def token1(self) -> str:
        return self._token1

    def impersonate(self, addr: str) -> None:
        self.w3.provider.make_request("anvil_impersonateAccount", [addr])
        self.w3.provider.make_request("anvil_setBalance", [addr, hex(10 ** 20)])

    def stop_impersonate(self, addr: str) -> None:
        try:
            self.w3.provider.make_request("anvil_stopImpersonatingAccount", [addr])
        except Exception as exc:
            raise RuntimeError(f"Failed to stop impersonation: {exc}")

    def mine(self) -> None:
        self.w3.provider.make_request("evm_mine", [])

    def erc20(self, addr: str):
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(addr), abi=ERC20_ABI
        )

    def pair(self):
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(POOL_ADDRESS), abi=PAIR_ABI
        )

    def token_balance(self, token_addr: str, holder: str) -> int:
        holder_cs = safe_checksum(holder)
        if holder_cs is None:
            raise ValueError(f"Invalid address: {holder}")

        return self.erc20(token_addr).functions.balanceOf(holder_cs).call()

    def fund_via_storage(
        self, token_addr: str, recipient: str, amount: int, slot: int
    ) -> bool:
        """Write amount into ERC-20 balances[recipient] storage slot."""
        from eth_abi import encode as abi_encode
        key = Web3.keccak(
            abi_encode(["address", "uint256"],
                       [Web3.to_checksum_address(recipient), slot])
        )
        val = "0x" + format(amount, "064x")
        for method in ("anvil_setStorageAt", "hardhat_setStorageAt"):
            try:
                self.w3.provider.make_request(
                    method,
                    [Web3.to_checksum_address(token_addr), key.hex(), val]
                )
                return True
            except Exception:
                continue
        return False


# ── AMM math ──────────────────────────────────────────────────────────────────

def get_amount_out(amount_in: int, reserve_in: int, reserve_out: int) -> int:
    """
    Uniswap V2 getAmountOut (0.3% fee).
    """
    assert amount_in > 0 and reserve_in > 0 and reserve_out > 0, \
        "Invalid inputs for getAmountOut"
    amount_in_with_fee = amount_in * 997
    numerator = amount_in_with_fee * reserve_out
    denominator = reserve_in * 1000 + amount_in_with_fee
    return numerator // denominator


# ── Pool state ────────────────────────────────────────────────────────────────

def pool_state(session: AnvilSession) -> dict:
    """Mine a block then read reserves from the target pair directly."""
    session.mine()
    r0, r1, _ = session.pair().functions.getReserves().call()

    if session.token0 == ARB_TOKEN.lower():
        arb_r, usdc_r = r0, r1
    else:
        usdc_r, arb_r = r0, r1

    arb_h = arb_r / 1e18
    usdc_h = usdc_r / 1e6
    price = usdc_h / arb_h if arb_h else 0.0
    return {
        "arb": arb_h, "usdc": usdc_h, "price": price,
        "tvl": usdc_h * 2, "arb_raw": arb_r, "usdc_raw": usdc_r
    }


def log_pool(label: str, s: dict) -> None:
    log.info(
        "%s | ARB=%.2f  USDC=%.2f  price=$%.6f  TVL≈$%.0f",
        label, s["arb"], s["usdc"], s["price"], s["tvl"]
    )


def safe_checksum(addr: str) -> str | None:
    try:
        if not isinstance(addr, str):
            return None
        addr = addr.strip()
        if not addr.startswith("0x") or len(addr) != 42:
            return None
        return Web3.to_checksum_address(addr)
    except Exception:
        return None


# ── Whale funding ─────────────────────────────────────────────────────────────

def ensure_funded(
    session: AnvilSession,
    token_addr: str,
    whale: str,
    amount_needed: int,
    storage_slot: int,
    decimals: int,
    label: str = ""
) -> None:
    """Verify whale balance; fund via storage slot if insufficient."""
    whale_cs = Web3.to_checksum_address(whale)
    bal = session.token_balance(token_addr, whale_cs)
    log.info(
        "%s whale %s | balance=%.4f  need=%.4f",
        label, whale[:10], bal / 10**decimals, amount_needed / 10**decimals
    )
    if bal < amount_needed:
        log.warning("Funding via storage slot %d …", storage_slot)
        ok = session.fund_via_storage(
            token_addr, whale_cs, amount_needed * 20, storage_slot
        )
        if not ok:
            raise RuntimeError(
                f"Storage-slot funding failed for {token_addr[:10]}"
            )
        new_bal = session.token_balance(token_addr, whale_cs)
        log.info("After funding: %.4f", new_bal / 10**decimals)
        if new_bal < amount_needed:
            raise RuntimeError(
                f"Still insufficient after storage-slot write: "
                f"{new_bal} < {amount_needed}"
            )


# ── Direct pair swap ──────────────────────────────────────────────────────────

def direct_pair_swap(
    session: AnvilSession,
    token_in_addr: str,
    amount_in: int,
    whale: str,
    label: str,
    token_in_decimals: int,
    token_in_slot: int
) -> str:
    """
    Execute a swap directly on POOL_ADDRESS (bypasses router / factory).

    Steps:
      1. Fund whale if needed
      2. Read current reserves → compute amountOut via AMM formula
      3. Impersonate whale → transfer(tokenIn, pair, amountIn)
      4. pair.swap(amount0Out, amount1Out, whale, b"")
      5. Stop impersonation → mine block
    """
    whale_cs = Web3.to_checksum_address(whale)
    pair_cs = Web3.to_checksum_address(POOL_ADDRESS)

    # 1. Ensure whale is funded
    ensure_funded(
        session, token_in_addr, whale_cs, amount_in,
        token_in_slot, token_in_decimals, label=label
    )

    # 2. Read reserves
    r0, r1, _ = session.pair().functions.getReserves().call()
    if session.token0 == token_in_addr.lower():
        reserve_in, reserve_out = r0, r1
        is_zero_for_one = True
    else:
        reserve_in, reserve_out = r1, r0
        is_zero_for_one = False

    amount_out = get_amount_out(amount_in, reserve_in, reserve_out)
    amount0Out = 0
    amount1Out = 0
    if is_zero_for_one:
        amount1Out = amount_out
    else:
        amount0Out = amount_out

    log.info(
        "%s | amount_in=%d  amount_out=%d  r_in=%d  r_out=%d  "
        "zero_for_one=%s  amount0Out=%d  amount1Out=%d",
        label, amount_in, amount_out, reserve_in, reserve_out,
        is_zero_for_one, amount0Out, amount1Out
    )

    if amount_out == 0:
        raise RuntimeError(f"{label}: AMM returned amount_out=0")

    # 3–4. Impersonate whale, transfer then swap
    session.impersonate(whale_cs)
    try:
        # Transfer tokenIn → pair (pair must hold it before swap() is called)
        transfer_tx = session.erc20(token_in_addr).functions.transfer(
            pair_cs, amount_in
        ).transact({"from": whale_cs, "gas": 120_000})
        session.mine()

        t_receipt = session.w3.eth.wait_for_transaction_receipt(
            transfer_tx, timeout=30
        )
        if not t_receipt.status:
            raise RuntimeError(f"{label}: transfer(tokenIn → pair) FAILED")
        log.info("%s | transfer ok tx=%s", label, transfer_tx.hex())

        # Direct swap on pair
        swap_tx = session.pair().functions.swap(
            amount0Out, amount1Out, whale_cs, b""
        ).transact({"from": whale_cs, "gas": 250_000})

    finally:
        session.stop_impersonate(whale_cs)

    # 5. Mine and confirm
    session.mine()
    if swap_tx is None:
        raise RuntimeError(f"{label}: swap transaction was never submitted")

    receipt = session.w3.eth.wait_for_transaction_receipt(swap_tx, timeout=60)
    status = "OK" if receipt.status else "FAIL"

    log.info(
        "%s | tx=%s  status=%s  gas=%d",
        label, swap_tx.hex(), status, receipt.gasUsed
    )

    if not receipt.status:
        log.error(
            "%s: pair.swap() REVERTED. Possible causes: "
            "insufficient reserve_out (%d raw), or pair detected imbalance.",
            label, reserve_out
        )

    return swap_tx.hex()


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
        log.info("Discord notify failed: %s", exc)


# ── Main scenario ─────────────────────────────────────────────────────────────

async def run_scenario(direction: str, amount: int) -> None:
    log.info("=" * 62)
    log.info(
        "ARB/USDC.e Demo Arb Scenario | direction=%s  amount=%d USDC.e",
        direction, amount
    )
    log.info("=" * 62)

    session = AnvilSession(ANVIL_RPC)
    before = pool_state(session)
    log_pool("POOL BEFORE", before)

    # ── Find a USDC.e whale that actually holds tokens ────────────────────────
    usdc_whale_candidates = [
        "0x1eED63EfCD1780C3b8A3B9D5A67f48f57E6d8B4",   # Circle treasury
        "0xe50fA9b3c56FfB159cB0FCA61F5c9D750e8128c5",   # Aave aUSDC.e
        "0x625E7708f30cA75bfd92586e17077590C60eb4cD",   # Aave v3 treasury
        "0x7F5c764cBc14f9669B88837ca1490cCa17c31607"   # Velodrome USDC pool
    ]
    usdc_whale = usdc_whale_candidates[0]
    usdc_raw = amount * 10 ** 6

    for candidate in usdc_whale_candidates:
        candidate_cs = safe_checksum(candidate)
        if candidate_cs is None:
            log.warning("Skip invalid address format: %s", candidate)
            continue

        try:
            bal = session.token_balance(USDC_TOKEN, candidate_cs)
        except Exception as e:
            log.warning("Balance check failed for %s: %s", candidate_cs, e)
            continue

        if bal >= usdc_raw:
            usdc_whale = candidate_cs
            log.info(
                "USDC.e whale selected: %s | balance=%.2f",
                candidate_cs[:10], bal / 1e6
            )
            break
    else:
        log.warning(
            "No valid USDC.e whale found - fallback to storage slot funding"
        )

    await notify(
        f"📊 **Demo Arb Scenario START**\n"
        f"Pool: {POOL_ADDRESS[:10]}...\n"
        f"ARB={before['arb']:.2f}  USDC={before['usdc']:.2f}  "
        f"price=${before['price']:.6f}  TVL≈${before['tvl']:.0f}\n"
        f"Method: direct pair.swap() (no router)"
    )

    # ── Phase 1: buy ARB with USDC.e → DEX price of ARB goes UP ──────────────
    if direction in ("up", "both"):
        log.info("--- Phase 1: Buying ARB with %d USDC.e (DEX price UP) ---", amount)
        direct_pair_swap(
            session,
            token_in_addr = USDC_TOKEN,
            amount_in = usdc_raw,
            whale = usdc_whale,
            label = "BUY_ARB",
            token_in_decimals = 6,
            token_in_slot = 9   # USDC.e balances slot on Arbitrum
        )
        after = pool_state(session)
        log_pool("POOL AFTER BUY_ARB", after)
        pct = (after["price"] - before["price"]) / max(before["price"], 1e-9) * 100
        log.info("Price impact: %+.4f %%", pct)

        await notify(
            f"🚀 **Price PUSHED UP** (+{pct:.4f}%)\n"
            f"${before['price']:.6f} → ${after['price']:.6f}\n"
            f"Bot should fire **BUY_CEX_SELL_DEX** now..."
        )
        log.info("Waiting 6s for bot to react...")
        await asyncio.sleep(6)

    # ── Phase 2: sell ARB for USDC.e → DEX price of ARB goes DOWN ────────────
    if direction in ("down", "both"):
        current = pool_state(session)
        arb_price = current["price"]   # USDC.e per ARB
        arb_to_sell = int((amount / max(arb_price, 1e-9)) * 1e18)

        log.info(
            "--- Phase 2: Selling %.4f ARB → USDC.e (DEX price DOWN) ---",
            arb_to_sell / 1e18
        )
        direct_pair_swap(
            session,
            token_in_addr = ARB_TOKEN,
            amount_in = arb_to_sell,
            whale = ARB_WHALE,
            label = "SELL_ARB",
            token_in_decimals = 18,
            token_in_slot = 51   # ARB token balances slot on Arbitrum
        )
        after = pool_state(session)
        log_pool("POOL AFTER SELL_ARB", after)
        pct = (after["price"] - current["price"]) / max(current["price"], 1e-9) * 100
        log.info("Price impact: %+.4f %%", pct)

        await notify(
            f"📉 **Price PUSHED DOWN** ({pct:.4f}%)\n"
            f"${current['price']:.6f} → ${after['price']:.6f}\n"
            f"Bot should fire **BUY_DEX_SELL_CEX** now..."
        )
        log.info("Waiting 6s for bot to react...")
        await asyncio.sleep(6)

    final = pool_state(session)
    log_pool("FINAL POOL", final)
    await notify(
        f"✅ **Demo Scenario COMPLETE**\n"
        f"Final: ARB={final['arb']:.2f}  USDC={final['usdc']:.2f}  "
        f"price=${final['price']:.6f}\n"
        f"Check ArbBot logs for execution results."
    )
    log.info("Scenario complete - check bot logs for arb results.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo ARB/USDC.e scenario - direct pair.swap()",
        prog="python3 -m scripts.arb_scenario_demo"
    )
    parser.add_argument(
        "--direction", choices=["up", "down", "both"], default="both",
        help="Price direction to push (default: both)"
    )
    parser.add_argument(
        "--amount", type=int, default=DEFAULT_AMOUNT,
        help=f"USDC.e to swap (default: {DEFAULT_AMOUNT})"
    )
    args = parser.parse_args()
    asyncio.run(run_scenario(direction=args.direction, amount=args.amount))


if __name__ == "__main__":
    main()
