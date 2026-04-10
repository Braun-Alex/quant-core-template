.PHONY: init build up down fork stop-fork logs clean test lint check impact

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

# ====================== Anvil fork management ======================

# Start local Anvil fork of Ethereum mainnet
fork:
	$(DOCKER_COMPOSE) up -d anvil
	@echo "Anvil fork is running at http://localhost:8545"
	@echo "Use ETH_RPC_URL=http://localhost:8545 for local development"

# Stop only the Anvil service
stop-fork:
	$(DOCKER_COMPOSE) stop anvil

# ====================== Application commands ======================

# Run the main application
run: up
	$(DOCKER_COMPOSE) logs -f app

test:
	$(APP_RUN) pytest tests/

integration:
	$(APP_RUN) python -m scripts.integration_test

lint:
	$(APP_RUN) flake8 src/ tests/ scripts/

check: lint test

# Price impact analysis example
impact:
	$(APP_RUN) python -m src.pricing.impact_analyzer \
		0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc \
		--token-in USDC \
		--sizes 1000,10000,100000 \
		--rpc http://anvil:8545

# ====================== Cleanup ======================

clean:
	$(DOCKER_COMPOSE) down -v --remove-orphans
	rm -rf .pytest_cache .pre-commit-cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleanup completed!"
