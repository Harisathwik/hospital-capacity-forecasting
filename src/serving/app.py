"""Production-grade FastAPI serving application.

Replaces the old serve_model.py with:
  • MLflow model registry (Production / Staging fallback) — no hardcoded run_id
  • Prediction bounds checking (ICU occupancy ≥ 0, flag if > 2× training max)
  • Audit logging to append-only JSONL (data/reports/prediction_audit.jsonl)
  • OpenTelemetry instrumentation (FastAPIInstrumentor)
  • Request-timing middleware
  • Simple in-memory rate limiting (100 req/min by default)
  • Proper health check that verifies the model is loaded
  • /metrics endpoint stub for Prometheus
  • Backward-compatible /predict endpoint shape
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.serving.predict import (
    DEFAULT_TRAINING_MAX_ICU,
    audit_log,
    load_model_from_registry,
    predict,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load the project YAML config."""
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config file %s not found — using defaults", config_path)
        return {}


config = load_config()

SERVING_CFG = config.get("serving", {})
HOST = SERVING_CFG.get("host", "0.0.0.0")
PORT = int(SERVING_CFG.get("port", 8000))
MODEL_STAGE = SERVING_CFG.get("model_stage", "Production")
MODEL_NAME = SERVING_CFG.get("model_name", "icu_forecast_xgboost")
TRAINING_MAX_ICU = float(SERVING_CFG.get("training_max_icu", DEFAULT_TRAINING_MAX_ICU))

GOV_CFG = config.get("governance", {})
REGISTRY_STAGES = GOV_CFG.get("registry_stages", {})
PREFERRED_STAGE = REGISTRY_STAGES.get("production", "Production")
FALLBACK_STAGE = REGISTRY_STAGES.get("staging", "Staging")

# ── Application ────────────────────────────────────────────────────────────

app = FastAPI(
    title="ICU Forecast XGBoost Model Server",
    version="0.2.0",
    description="Hospital ICU bed demand forecasting — production serving layer",
)

# ── Global model state ────────────────────────────────────────────────────

_model_state: Dict[str, Any] = {
    "model": None,            # mlflow pyfunc model
    "model_stage": None,      # e.g. "Production"
    "model_uri": None,        # e.g. "models:/icu_forecast_xgboost/Production"
    "loaded": False,
}


def _load_model() -> None:
    """Load model from MLflow registry into global state."""
    try:
        model, stage, uri = load_model_from_registry(
            model_name=MODEL_NAME,
            preferred_stage=PREFERRED_STAGE,
            fallback_stage=FALLBACK_STAGE,
        )
        _model_state.update({
            "model": model,
            "model_stage": stage,
            "model_uri": uri,
            "loaded": True,
        })
        logger.info("Model loaded successfully — stage=%s uri=%s", stage, uri)
    except Exception as exc:
        logger.error("Model loading failed: %s", exc)
        _model_state["loaded"] = False


# Load eagerly at import time so health check can validate
_load_model()

# ── OpenTelemetry instrumentation ─────────────────────────────────────────

try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry FastAPI instrumentation enabled")
except ImportError:
    logger.info("opentelemetry-instrumentation-fastapi not installed — skipping OTel")
except Exception as exc:
    logger.warning("OTel instrumentation failed: %s", exc)

# ── Request-timing middleware ──────────────────────────────────────────────

@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Add X-Process-Time-ms header to every response."""
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.3f}"
    return response

# ── Rate limiting (in-memory, 100 req/min) ────────────────────────────────

RATE_LIMIT_WINDOW = 60          # seconds
RATE_LIMIT_MAX_REQUESTS = 100   # per window per IP
_rate_store: Dict[str, List[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple per-IP rate limiter — returns 429 if over threshold."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    # Prune old entries
    timestamps = _rate_store[client_ip]
    _rate_store[client_ip] = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW]
    # Check limit
    if len(_rate_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded — try again later"},
        )
    _rate_store[client_ip].append(now)
    return await call_next(request)

# ── Pydantic schemas (backward compatible) ────────────────────────────────

class PredictRequest(BaseModel):
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    predictions: List[float]
    target_names: List[str]
    flags: Optional[List[str]] = None        # new field, optional for compat


# ── Health check ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Verify model is loaded; return degraded status if not."""
    if _model_state["loaded"]:
        return {
            "status": "healthy",
            "model_stage": _model_state["model_stage"],
            "model_uri": _model_state["model_uri"],
        }
    return JSONResponse(
        status_code=503,
        content={
            "status": "unhealthy",
            "detail": "Model not loaded — check MLflow model registry",
        },
    )

# ── Metrics endpoint stub (Prometheus) ────────────────────────────────────

@app.get("/metrics")
async def metrics():
    """Stub for Prometheus metrics — to be extended with prometheus_client."""
    return {
        "model_loaded": int(_model_state["loaded"]),
        "model_stage": _model_state.get("model_stage", "none"),
        "model_uri": _model_state.get("model_uri", "none"),
        # Extend with real Prometheus exposition here
    }

# ── Prediction endpoint ──────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(request: PredictRequest):
    """
    Single-row ICU occupancy prediction.

    Backward-compatible: returns ``predictions`` and ``target_names`` as before,
    plus an optional ``flags`` list for bounds warnings.
    """
    if not _model_state["loaded"]:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = _model_state["model"]
    start = time.perf_counter()

    try:
        predictions, flags = predict(
            model=model,
            features=request.features,
            feature_names=None,       # pyfunc model handles feature ordering
            training_max=TRAINING_MAX_ICU,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Audit log
    audit_log(
        request_features=request.features,
        predictions=predictions,
        target_names=[],       # not available from pyfunc model
        flags=flags,
        model_stage=_model_state["model_stage"],
        model_uri=_model_state["model_uri"],
        latency_ms=latency_ms,
    )

    return PredictResponse(
        predictions=predictions,
        target_names=[],       # kept for backward compat
        flags=flags if flags else None,
    )

# ── Startup / shutdown events ─────────────────────────────────────────────

@app.on_event("startup")
async def _on_startup():
    logger.info(
        "ICU Forecast server starting — port=%d model_stage=%s",
        PORT, MODEL_STAGE,
    )
    if not _model_state["loaded"]:
        logger.error("Model not loaded at startup — /health will return 503")


@app.on_event("shutdown")
async def _on_shutdown():
    logger.info("ICU Forecast server shutting down")

# ── Entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
