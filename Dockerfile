# Multi-stage build: first stage gets Foundry tools (Anvil)
FROM ghcr.io/foundry-rs/foundry:latest AS foundry

# Main Python image
FROM python:3.11-slim

# Copy Anvil and other Foundry binaries from the foundry image
COPY --from=foundry /usr/local/bin/anvil /usr/local/bin/anvil
COPY --from=foundry /usr/local/bin/forge /usr/local/bin/forge
COPY --from=foundry /usr/local/bin/cast  /usr/local/bin/cast

# Install system dependencies
RUN apt-get update && apt-get install -y \
    make \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies with layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Expose the default Anvil port
EXPOSE 8545

# Default command (can be overridden in docker-compose)
CMD ["python3", "src/main.py"]
