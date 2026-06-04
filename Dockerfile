FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install OpenTelemetry instrumentation (graceful no-op if not used)
RUN pip install --no-cache-dir opentelemetry-api opentelemetry-sdk \
    opentelemetry-instrumentation-fastapi || true

COPY src/ src/
COPY configs/ configs/

# Create data directories
RUN mkdir -p /app/data/reports

EXPOSE 8000

# Use the production-grade serving app
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "30"]
