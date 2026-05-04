# Quant-Core-Template

A production-ready CEX-DEX arbitrage engine with real on-chain execution,
multi-layer risk management, automated kill switches, and structured monitoring.

## Architecture

```
Real-time price feeds (event-driven)
  ├─ LiveOrderBook (Binance WebSocket depth stream)
  │     └─ on each depth diff → PriceFeedManager._on_cex_update()
  └─ DEXPriceFeed (Uniswap V2 reserve polling, 1s interval)
        └─ on price change   → PriceFeedManager._on_dex_update()
                                         │
                              PriceState (shared in-memory)
                                         │
                              _on_price_update(pair, state)  ← fires on every update
                                         │
on_price_update callback (per update):
  ├─ 1. Heartbeat          ← dead man's switch
  ├─ 2. Manual kill switch ← file-based (/tmp/arb_bot_kill)
  ├─ 3. Auto kill switch   ← capital floor / error-rate
  ├─ 4. SPRT circuit breaker
  ├─ 5. SignalGenerator.generate_from_feed()
  │       └─ KalmanFilter  ← Bayesian spread estimator
  ├─ 6. PreTradeValidator  ← price sanity, freshness, spread bounds
  ├─ 7. BinanceTradingRules ← lot-size, tick-size, min-notional
  ├─ 8. RiskManager        ← risk limits (size, daily loss, drawdown)
  ├─ 9. safety_check()     ← ABSOLUTE hard ceilings (non-configurable)
  ├─ 10. PFA Executor
  │        ├─ Leg 1        ← Uniswap V2 swap (Arbitrum) or CEX order
  │        ├─ Leg 2        ← CEX order or Uniswap V2 swap
  │        └─ Unwind       ← automatic reversal on leg failure
  ├─ 11. BalanceVerifier   ← post-trade sanity check, stops on mismatch
  └─ 12. BotMonitor        ← health metrics, Telegram alerts, daily summary

Fallback (if WS connection fails): REST polling every 1 second
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

## 🚀 Commands

```bash
make up                   # Start all services
make down                 # Stop all services
make run                  # Start + follow app logs
make logs                 # Follow logs

make test                 # Unit tests
make test-bot             # Bot integration tests
make lint                 # flake8
make check                # lint + test

make integration          # Sepolia blockchain integration test
make integration-real     # Binance Testnet + Arbitrum/Mainnet dry-run

make fork                 # Start local Anvil fork
make stop-fork            # Stop Anvil

make check-rebalance      # Cross-venue skew report
make plan-rebalance       # Rebalance transfer plans
make check-arb            # Arbitrage opportunity check
make fork-arb             # Anvil fork of Arbitrum (for demo mode)
make fund-demo            # Fund wallet with test tokens on Anvil fork
make check-demo-balances  # Check demo wallet token balances
make pool-info            # ARB/USDC pool state on fork
make run-demo             # Start full demo stack (fork + fund + bot)
make pnl                  # P&L dashboard
make orderbook            # Live CEX order book snapshot
make impact               # DEX price impact analysis
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

### Network

| Variable | Default | Description |
|---|---|---|
| `USE_ARBITRUM` | `true` | Use Arbitrum One (false = Ethereum) |
| `ARBITRUM_RPC_URL` | Public Arb RPC | Arbitrum HTTP endpoint |
| `POOL_ADDRESSES` | — | Comma-separated Uniswap V2 pair addresses |
| `GAS_LIMIT_SWAP` | 500000 | Gas limit for swaps |
| `SLIPPAGE_TOLERANCE_BPS` | 50 | DEX slippage tolerance |

### Risk

| Variable | Default | Description |
|---|---|---|
| `INITIAL_CAPITAL` | 100.0 | Starting capital for drawdown tracking |
| `MAX_TRADE_USD` | 20.0 | Per-trade size ceiling (≤ $25 absolute) |
| `MAX_DAILY_LOSS` | 15.0 | Daily loss stop (≤ $20 absolute) |
| `MAX_DRAWDOWN_PCT` | 0.20 | Peak-to-trough drawdown stop |
| `CONSECUTIVE_LOSS_LIMIT` | 3 | Pause after N consecutive losses |

### Monitoring

| Variable | Description |
|---|---|
| `LOG_DIR` | Log file directory (default: `logs`) |
| `TELEGRAM_BOT_TOKEN` | Telegram bot token for alerts |
| `TELEGRAM_CHAT_ID` | Telegram chat/channel ID |
| `KILL_SWITCH_FILE` | Kill switch file path |
| `HEARTBEAT_FILE` | Heartbeat file path |

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
