# Quant-Core-Template

A high-integrity engineering baseline for a CEX-DEX arbitrage trading engine.

This template provides a robust foundation for building high-performance trading systems with real on-chain DEX integration, cross-venue inventory management, arbitrage signal generation, and a production-ready execution pipeline.

## Architecture Overview

```
tick(t)
  ├─ SignalGenerator
  │    ├─ CEX prices      ← Binance (Testnet or Mainnet)
  │    ├─ DEX prices      ← Uniswap V2 on Ethereum Mainnet
  │    └─ KalmanFilter    ← Bayesian spread estimator
  ├─ EntropyCRITIC + TOPSIS  ← Multi-criteria signal scoring
  ├─ PFA Executor
  │    ├─ Leg 1  ← CEX (Binance limit-IOC) or DEX (Uniswap V2 swap)
  │    ├─ Leg 2  ← DEX or CEX (opposite venue)
  │    └─ Unwind ← Automatic reversal on leg failure
  ├─ SPRT Circuit Breaker  ← Sequential probability ratio test
  └─ PnLTracker            ← Trade ledger and P&L reporting
```

## 🚀 Prerequisites

- Docker & Docker Compose
- Git
- Make

## 🚀 Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/Braun-Alex/quant-core-template.git
cd quant-core-template
```

**2. Configure environment:**
```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

| Variable | Description |
|---|---|
| `OPERATION_MODE` | `test` (safe default) or `production` |
| `PRIVATE_KEY` | Hot wallet private key (0x-prefixed) |
| `ETH_RPC_URL` | Ethereum Mainnet RPC (Alchemy / Infura) |
| `BINANCE_TESTNET_API_KEY` | Binance Testnet key (test mode) |
| `BINANCE_TESTNET_SECRET` | Binance Testnet secret (test mode) |
| `POOL_ADDRESSES` | Comma-separated Uniswap V2 pair addresses |

**3. Initialize and build:**
```bash
make init
make build
```

## 🚀 Operation Modes

The engine supports two modes controlled by the `OPERATION_MODE` environment variable.

### Test Mode (default)
- **CEX:** Binance Testnet
- **DEX:** Ethereum Mainnet - real on-chain AMM quotes, transactions are **built and signed but never broadcast**
- Safe for development and CI/CD
```bash
make test-mode               # Switch .env to test mode
make bot-integration-testnet # Run Testnet bot integration test
```

### Production Mode
- **CEX:** Binance Mainnet
- **DEX:** Ethereum Mainnet - transactions are **broadcast and confirmed**
- Requires wallet to be funded and mainnet API credentials
```bash
make prod-mode   # Switch .env to production mode (use with caution)
```

Switch modes at any time with:
```bash
make show-mode   # Print current mode
make test-mode   # Switch to test
make prod-mode   # Switch to production
```

## 🚀 Usage

### Core Commands

| Command | Description |
|---|---|
| `make up` | Start all services in detached mode |
| `make down` | Stop all services |
| `make logs` | Follow logs of all services |
| `make run` | Start services and follow app logs |

### Development & Testing

```bash
make test              # Run all unit tests
make test-bot          # Run bot tests
make lint              # Run flake8 linting
make check             # Full quality check (lint + test)
make clean             # Clean caches and containers
```

### Integration Testing

```bash
# Sepolia blockchain integration test
make transfer-integration

# Testnet bot integration test: Binance Testnet and Ethereum Mainnet
make bot-integration-testnet

# Mainnet bot integration test (requires funded wallet and mainnet credentials)
make bot-integration-mainnet
```

The Testnet integration test (`scripts/bot_integration_test.py`) runs a full pipeline check.

1. Loads and verifies the hot wallet.
2. Connects to Ethereum Mainnet and reads the chain ID and ETH balance.
3. Loads registered Uniswap V2 pool states from on-chain data.
4. Fetches a live Binance order book snapshot.
5. Generates an arbitrage signal using the Kalman filter.
6. Runs the executor in dry-run mode - builds swap transactions but does not broadcast them.

### Anvil Fork (Local Ethereum Mainnet)

```bash
make fork       # Start local Anvil fork of Ethereum mainnet
make stop-fork  # Stop only Anvil service
```

### Inventory & Rebalancing

```bash
make check-rebalance    # Show current cross-venue skew report
make plan-rebalance     # Generate rebalance plans for all unbalanced assets
```

### Arbitrage & Analytics

```bash
make check-arb    # Check arbitrage opportunity for ETH/USDT
make pnl          # Show position & PnL summary
make orderbook    # Live Binance order book snapshot
make impact       # Price impact analysis on Uniswap V2 pool
```

## 🚀 Configuration Reference

All parameters can be set in `.env`.

### Mode & Credentials

| Variable | Default | Description |
|---|---|---|
| `OPERATION_MODE` | `test` | `test` or `production` |
| `PRIVATE_KEY` | — | Hot wallet private key (0x-prefixed) |
| `ETH_RPC_URL` | — | Ethereum Mainnet HTTP RPC URL |
| `ETH_WS_URL` | — | Ethereum Mainnet WebSocket URL (mempool watcher) |
| `BINANCE_TESTNET_API_KEY` | — | Binance Testnet API key |
| `BINANCE_TESTNET_SECRET` | — | Binance Testnet secret |
| `BINANCE_API_KEY` | — | Binance Mainnet API key (production only) |
| `BINANCE_SECRET` | — | Binance Mainnet secret (production only) |
| `OPENAI_API_KEY` | — | OpenAI key for LLM anomaly advisor (optional) |

### DEX Parameters

| Variable | Default | Description |
|---|---|---|
| `POOL_ADDRESSES` | — | Comma-separated Uniswap V2 pair contract addresses |
| `UNISWAP_V2_ROUTER` | `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D` | Router address |
| `GAS_LIMIT_SWAP` | `250000` | Gas limit for swap transactions |
| `GAS_LIMIT_APPROVAL` | `60000` | Gas limit for ERC-20 approval transactions |
| `SLIPPAGE_TOLERANCE_BPS` | `50` | Slippage tolerance in basis points (0.5%) |
| `TX_DEADLINE_SECONDS` | `300` | Swap transaction deadline (seconds from now) |

### Executor Parameters

| Variable | Default | Description |
|---|---|---|
| `LEG1_TIMEOUT` | `5` | Leg 1 execution timeout (seconds) |
| `LEG2_TIMEOUT` | `60` | Leg 2 execution timeout (seconds) |
| `MIN_FILL_RATIO` | `0.80` | Minimum acceptable fill ratio (φ) |
| `VAR_CONFIDENCE` | `0.95` | VaR confidence level for risk filter |
| `VOL_PER_SQRT_SECOND` | `0.0002` | Volatility parameter for VaR calculation |
| `USE_DEX_FIRST` | `true` | Execute DEX leg first (`true`) or CEX first (`false`) |
| `MAX_RECOVERY_ATTEMPTS` | `2` | Maximum unwind retry attempts |
| `UNWIND_TIMEOUT` | `30` | Unwind transaction timeout (seconds) |
| `UNWIND_SLIPPAGE_BPS` | `100` | Extra slippage tolerance for unwind fills (1%) |

### Signal Generator Parameters

| Variable | Default | Description |
|---|---|---|
| `SIGNAL_TTL_SECONDS` | `5` | Signal expiry time-to-live |
| `COOLDOWN_SECONDS` | `2` | Minimum interval between signals per pair |
| `KELLY_FRACTION` | `0.25` | Kelly criterion fraction (position sizing) |
| `MAX_POSITION_USD` | `10000` | Maximum position size in USD |
| `CEX_TAKER_BPS` | `10` | CEX taker fee in basis points |
| `DEX_SWAP_BPS` | `30` | DEX swap fee in basis points |
| `GAS_COST_USD` | `5` | Estimated gas cost in USD per trade |

## 🚀 Integration Testing

The project includes two integration test suites.

### Blockchain Integration Test (Sepolia Testnet)

Simulates a real transaction lifecycle.

1. Loads wallet from environment.
2. Validates network connectivity and balance.
3. Builds and estimates a transaction.
4. Performs local cryptographic recovery to verify the signature.
5. Broadcasts to Sepolia and monitors for block confirmation.
```bash
make transfer-integration
```

### Testnet Trading Integration Test (Binance Testnet and Ethereum Mainnet)

Runs the full signal-to-execution pipeline in dry-run mode.

1. Loads and verifies the hot wallet.
2. Connects to Ethereum Mainnet, reads chain ID and ETH balance.
3. Loads Uniswap V2 pool states from on-chain data.
4. Fetches a live Binance Testnet order book snapshot.
5. Generates an arbitrage signal using the pre-warmed Kalman filter.
6. Runs the executor - builds signed swap transactions but does not broadcast them.
```bash
make bot-integration-testnet
```
