"""Production-grade ICU Forecast Model Server.

This is the backward-compatible entry point that delegates to
``src.serving.app`` which contains the full production FastAPI app.

Key changes vs. the original serve_model.py:
  1. No hardcoded run_id — model loaded from MLflow model registry
     (Production stage with Staging fallback).
  2. Prediction bounds checking (ICU occupancy ≥ 0; flag if > 2× training max).
  3. Audit logging to append-only JSONL (data/reports/prediction_audit.jsonl).
  4. OpenTelemetry instrumentation (FastAPIInstrumentor).
  5. Request-timing middleware (X-Process-Time-ms header).
  6. In-memory rate limiting (100 req/min per IP).
  7. Proper health check that verifies model is loaded (503 if not).
  8. /metrics endpoint stub for Prometheus.
  9. Backward-compatible /predict endpoint shape (+ optional ``flags`` field).
"""

# Re-export the production app so that `python serve_model.py` and
# `uvicorn serve_model:app` continue to work exactly as before.
from src.serving.app import app, config  # noqa: F401

if __name__ == "__main__":
    import uvicorn

    from src.serving.app import HOST, PORT

    uvicorn.run(app, host=HOST, port=PORT)
