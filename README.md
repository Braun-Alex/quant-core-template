# Quant-Core-Template

A high-integrity engineering baseline for a trading engine.

This template provides a robust foundation for building high-performance trading systems,
cross-venue inventory management, arbitrage bots and DeFi applications
with strong focus on type safety, reliability and observability.

## 🚀 Prerequisites 🚀

Before you begin, ensure you have the following installed:
- Docker & Docker Compose
- Git
- Make

## 🚀 Quick Start 🚀

1. Clone the repository:
```bash
git clone https://github.com/Braun-Alex/quant-core-template.git
```

2. Create your local environment file from the provided template:
```bash
cp .env.example .env
```
Open ```.env``` and add your ```PRIVATE_KEY``` to enable blockchain interactions.
```SEPOLIA_RPC_URL``` is required to run integration test.

3. Initialize the project:
```make
make init
```

4. Build the project:
```make
make build
```

## 🚀 Usage 🚀

### Core Commands

| Command                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `make up`                | Start all services in detached mode              |
| `make down`              | Stop all services                                |
| `make logs`              | Follow logs of all services                      |
| `make run`               | Start services and follow app logs               |

### Development & Testing

```bash
make test           # Run unit tests
make test-bot       # Run integration tests for bot
make lint           # Run flake8 linting
make check          # Full quality check (lint + test)
make clean          # Cleanup caches and containers
````


### Anvil Fork (Local Ethereum Mainnet)

```bash
make fork           # Start local Anvil fork
make stop-fork      # Stop only Anvil service
```

### Inventory & Rebalancing Tools

```bash
make check-rebalance    # Show current cross-venue skew report
make plan-rebalance     # Generate rebalance plans for all unbalanced assets
```

### Arbitrage & Trading Tools

```bash
make check-arb          # Check arbitrage opportunity for ETH/USDT
make pnl                # Show position & PnL summary
make orderbook          # Live Binance testnet order book snapshot
make impact             # Price impact analysis on Uniswap V2 pool`
```

## 🚀 Integration Testing 🚀

The project includes a comprehensive integration test that
simulates a real-world workflow on the Sepolia Testnet.

1. Loads wallet from environment.
2. Validates network connectivity and balance.
3. Builds and estimates a transaction.
4. Performs local cryptographic recovery to verify the signature.
5. Broadcasts to the network and monitors for block confirmation.

To run the integration test:
```make
make integration
```
