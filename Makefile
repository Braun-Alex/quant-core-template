.PHONY: init build up down logs fork stop-fork fork-arb fund-demo check-demo-balances pool-info run-demo \
		arb-demo arb-demo-up arb-demo-down arb-prod-dryrun arb-prod scout scout-demo scout-fast \
        show-mode test-mode demo-mode prod-mode run test test-bot transfer-integration bot-integration-testnet \
        bot-integration-mainnet run-bot lint check impact orderbook check-rebalance \
        plan-rebalance pnl check-arb clean

include .env
export

DOCKER_COMPOSE = docker compose
APP_RUN = $(DOCKER_COMPOSE) run --rm app

# ====================== Setup commands ======================

init:
	test -d .venv || python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install pre-commit==4.5.1
	.venv/bin/pre-commit install
	@echo "Development environment initialized!"

build:
	$(DOCKER_COMPOSE) build

# ====================== Docker services ======================

up:
	$(DOCKER_COMPOSE) up -d

down:
	$(DOCKER_COMPOSE) down

logs:
	$(DOCKER_COMPOSE) logs -f

# ====================== Demo mode ======================

# Start Anvil fork of Arbitrum Mainnet (for demo mode DEX)
fork-arb:
	$(DOCKER_COMPOSE) up -d anvil
	@echo "Arbitrum fork running at http://localhost:8545"

# Stop only the Anvil service
stop-fork:
	$(DOCKER_COMPOSE) stop anvil

# Fund demo wallet with test tokens (ARB, USDC, WETH) via Anvil cheatcodes
fund-demo:
	$(APP_RUN) python3 -m src.exchange.demo_setup fund \
		--wallet "${DEMO_WALLET_ADDRESS}" \
		--rpc http://anvil:8545 \
		--tokens ARB,USDC,WETH

# Check demo wallet balances on the fork
check-demo-balances:
	$(APP_RUN) python3 -m src.exchange.demo_setup check \
		--wallet "${DEMO_WALLET_ADDRESS}" \
		--rpc http://anvil:8545

# Print ARB/USDC pool state from the Arbitrum fork
pool-info:
	$(APP_RUN) python3 -m src.exchange.demo_setup pool

# Run the bot in demo mode (Binance Demo Trading and Anvil fork of Arbitrum One)
# 1) Start Anvil fork in the background and wait until healthy
# 2) Fund the demo wallet with test tokens via cheatcodes
# 3) Start the bot with OPERATION_MODE=demo DRY_RUN=false
# 4) Tail the bot logs
run-demo:
	@echo "==> Starting Anvil fork of Arbitrum Mainnet..."
	OPERATION_MODE=demo DRY_RUN=false $(DOCKER_COMPOSE) up -d anvil
	@echo "==> Starting arbitrage bot in demo mode..."
	OPERATION_MODE=demo DRY_RUN=false $(APP_RUN) python3 -m scripts.arb_bot

# ====================== Arbitrage scenarios ======================

# Run demo arb scenario (both price directions, 3 USDC)
# Bot must already be running: make run-demo
arb-demo:
	$(APP_RUN) python3 -m scripts.arb_scenario_demo --direction both --amount 3

# Run demo scenario pushing price UP only
arb-demo-up:
	$(APP_RUN) python3 -m scripts.arb_scenario_demo --direction up --amount 3

# Run demo scenario pushing price DOWN only
arb-demo-down:
	$(APP_RUN) python3 -m scripts.arb_scenario_demo --direction down --amount 3

# Run production arb scenario in DRY-RUN (build txs, do not send)
arb-prod-dryrun:
	$(APP_RUN) python3 -m scripts.arb_scenario_production --direction both --dry-run

# Run production arb scenario with REAL funds - use with caution
arb-prod:
	$(APP_RUN) python3 -m scripts.arb_scenario_production --direction both --amount 3

# ====================== Mode helpers ======================

# Print current operation mode
show-mode:
	@grep -E '^OPERATION_MODE' .env 2>/dev/null || echo "OPERATION_MODE not set (defaults to test)"

# Switch to test mode
test-mode:
	@sed -i 's/^OPERATION_MODE=.*/OPERATION_MODE=test/' .env 2>/dev/null || \
		echo "OPERATION_MODE=test" >> .env
	@echo "Switched to TEST mode (Binance Testnet + Arbitrum dry-run)"

# Switch to demo mode
demo-mode:
	@sed -i 's/^OPERATION_MODE=.*/OPERATION_MODE=demo/' .env 2>/dev/null || \
		echo "OPERATION_MODE=demo" >> .env
	@echo "Switched to DEMO mode (Binance Demo Trading + Anvil fork of Arbitrum)"

# Switch to production mode (use with caution)
prod-mode:
	@sed -i 's/^OPERATION_MODE=.*/OPERATION_MODE=production/' .env 2>/dev/null || \
		echo "OPERATION_MODE=production" >> .env
	@echo "Switched to PRODUCTION mode (Binance Mainnet + Arbitrum One live)"

# ====================== Application commands ======================

# Run the main application
run: up
	$(DOCKER_COMPOSE) logs -f app

test:
	$(APP_RUN) pytest tests/

test-bot:
	$(APP_RUN) pytest scripts/test_bot.py

# Sepolia blockchain integration test
transfer-integration:
	$(APP_RUN) python3 -m scripts.transfer_integration_test

# Testnet bot integration test: Binance Testnet + Arbitrum (dry-run)
bot-integration-testnet:
	$(APP_RUN) env OPERATION_MODE=test python3 -m scripts.bot_integration_test

# Mainnet bot integration test: requires funded wallet and mainnet credentials
bot-integration-mainnet:
	$(APP_RUN) env OPERATION_MODE=production python3 -m scripts.bot_integration_test

run-bot:
	$(APP_RUN) python3 -m scripts.arb_bot

lint:
	$(APP_RUN) flake8 src/ tests/ scripts/

check: lint test

# Price impact analysis example
impact:
	$(APP_RUN) python3 -m src.pricing.impact_analyzer \
		0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc \
		--sell USDC \
		--amounts 1000,10000,100000,1000000,3000000,9000000 \
		--rpc http://anvil:8545

# Live Binance Testnet order book snapshot example
orderbook:
	$(APP_RUN) python3 -m src.exchange.orderbook "ETH/USDT" --depth 30

# Cross-venue rebalance checker
check-rebalance:
	$(APP_RUN) python3 -m src.inventory.rebalancer --check

# Cross-venue rebalance planner
plan-rebalance:
	$(APP_RUN) python3 -m src.inventory.rebalancer --plan-all

# Arbitrage trade dashboard
pnl:
	$(APP_RUN) python3 -m src.inventory.pnl --summary

# Arbitrage checker
check-arb:
	$(APP_RUN) python3 -m src.inventory.arb_checker "ETH/USDT" --size 3.0

# ====================== Cleanup ======================

clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	rm -rf .pytest_cache .pre-commit-cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup completed!"
