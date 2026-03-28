# Quant-Core-Template

A high-integrity engineering baseline for a trading engine, featuring multi-chain address 
validation (BTC & ETH). Built with Python 3.11, Docker, and Make.

Supports Bitcoin (P2PKH, P2WPKH, P2TR) and Ethereum address formats.

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

Optional: open ```.env``` and add your ```OPENAI_API_KEY``` to enable AI features.

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

2. Running tests:
```make
make test
```

3. To check the code:
```make
make lint
```