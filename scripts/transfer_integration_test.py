import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from eth_account import Account

from src.core.wallet import WalletManager
from src.core.types import Address, TokenAmount
from src.chain.client import ChainClient
from src.chain.builder import TransactionBuilder


def log_step(message: str):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def format_gwei(wei: int) -> str:
    return f"{wei / 1_000_000_000:.9f}"


def main():
    print("=" * 50)
    print("QUANT CORE --- INTEGRATION TEST")
    print("=" * 50 + "\n")

    load_dotenv()
    SEPOLIA_RPC_URL = os.getenv("SEPOLIA_RPC_URL")
    if not SEPOLIA_RPC_URL:
        raise ValueError(f"{SEPOLIA_RPC_URL} not found in environment")

    # 1. Initialization and environment check
    try:
        wallet = WalletManager.from_env()
        log_step(f"Wallet loaded: {wallet.address}")
    except Exception as err:
        print(f"❌ Critical error ❌: could not load wallet from environment: {err}")
        sys.exit(1)

    # Use Sepolia testnet RPC
    client = ChainClient([SEPOLIA_RPC_URL])

    # Network connectivity check
    try:
        chain_id = client.get_chain_id()
        network_name = "Sepolia" if chain_id == 11155111 else f"Unknown ({chain_id})"
        log_step(f"Connected to network: {network_name}")
    except Exception as err:
        print(f"❌ Connection failed ❌: {err}")
        sys.exit(1)

    # 2. Balance check and funds audit
    balance = client.get_balance(Address.from_string(wallet.address))
    print(f"Balance: {balance.human:.6f} {balance.symbol}")

    if balance.human < 0.001:
        print(f"❌ Insufficient funds! Need at least 0.001 ETH, have {balance.human} ❌")
        sys.exit(1)

    # 3. Transaction preparation
    recipient = Address.from_string("0x742d35Cc6634C0532925a3b844Bc454e4438f44e")
    amount = TokenAmount.from_human("0.001", 18, "ETH")

    print("\n--------- Preparing transaction ---------")
    builder = (TransactionBuilder(client, wallet)
               .to(recipient)
               .value(amount)
               .with_gas_estimate(1.3)
               .with_gas_price("medium"))

    tx = builder.build()
    print(f"  Recipient:    {tx.to.checksum}")
    print(f"  Value:        {tx.value.human} {tx.value.symbol}")
    print(f"  Gas limit:    {tx.gas_limit}")
    print(f"  Max fee:      {format_gwei(tx.max_fee_per_gas)} gwei")
    print(f"  Max priority: {format_gwei(tx.max_priority_fee)} gwei")

    # 4. Local signature verification
    print("\n--------- Cryptographic security check ---------")
    signed_tx = builder.build_and_sign()

    # Recover address from signed transaction BEFORE broadcasting to the network
    recovered_addr = Account.recover_transaction(signed_tx.raw_transaction)
    if recovered_addr.lower() == wallet.address.lower():
        print("      Local signature verification: ✅ PASSED ✅")
        print(f"     Recovered Address: {recovered_addr}")
    else:
        print("      Local signature verification: ❌ FAILED ❌")
        print(f"     Expected: {wallet.address}")
        print(f"     Got:      {recovered_addr}")
        sys.exit(1)

    # 5. Broadcasting and monitoring
    print("\n--------- Network broadcast ---------")
    try:
        tx_hash = client.send_transaction(signed_tx.raw_transaction)
        log_step("✅ Transaction broadcasted successfully ✅")
        print(f"  🔗 Hash: {tx_hash}")
        print(f"  🌐 Explorer: https://sepolia.etherscan.io/tx/{tx_hash}")
    except Exception as err:
        print(f"❌ Broadcast failed ❌: {err}")
        sys.exit(1)

    # 6. Waiting for receipt and final analysis
    print("\n--------- Finalizing (waiting for block) ---------")
    start_time = time.time()
    receipt = client.wait_for_receipt(tx_hash, timeout=180)
    duration = time.time() - start_time

    print("\n--------- Transaction analysis ---------")
    status_icon = "✅" if receipt.status else "❌"
    print(f"  Status:     {status_icon} {'SUCCESS' if receipt.status else 'FAILED'}")
    print(f"  Block:      {receipt.block_number}")
    print(f"  Time taken: {duration:.3f} seconds")
    print(f"  Gas used:   {receipt.gas_used} ({int(receipt.gas_used / tx.gas_limit * 100)}% of limit)")
    print(f"  Fee:        {receipt.tx_fee.human:.10f} ETH")

    print("\n" + "=" * 50)
    print("✅ INTEGRATION TEST PASSED ✅")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as err:
        print(f"\n\n❌ Unexpected test failure ❌: {err}")
        sys.exit(1)
