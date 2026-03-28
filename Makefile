# .PHONY ensures that these targets do not conflict with files of the same name
.PHONY: init build run test lint format check clean

DOCKER_COMPOSE = docker compose
APP_RUN = $(DOCKER_COMPOSE) run --rm app

init:
	test -d .venv || python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install pre-commit==4.5.1
	.venv/bin/pre-commit install
	-sudo groupadd docker
	sudo usermod -aG docker $${USER}
	newgrp docker

build:
	$(DOCKER_COMPOSE) build

run:
	$(DOCKER_COMPOSE) up app

test:
	$(APP_RUN) pytest tests/

lint:
	$(APP_RUN) flake8 src/ tests/

check: lint test

clean:
	rm -rf .pytest_cache .pre-commit-cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete