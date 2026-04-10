# Quant-Core-Template

A high-integrity engineering baseline for a trading engine.

This template provides a robust foundation for building decentralized finance
applications and trading bots with a focus on type safety and reliability.

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

1. Running the engine:
```make
make run
```

2. Execute unit tests:
```make
make test
```

3. To check the code for style and syntax issues using ```flake8```:
```make
make lint
```

4. Full quality check:
```make
make check
```

5. To start local Anvil fork of Ethereum mainnet:
```make
make fork
```

6. To stop only the Anvil service:
```make
make stop-fork
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
