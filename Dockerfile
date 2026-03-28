FROM python:3.11-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    make \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency list first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

CMD ["python3", "src/main.py"]