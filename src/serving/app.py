"""FastAPI serving layer for churn prediction."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="Churn Prediction API",
    description="Real-time churn prediction serving layer",
    version="1.0.0",
)

# ── Model artifacts ───────────────────────────────────────────
MODEL_PATH = Path("models/churn_model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

_model = None
_preprocessor = None
_model_metadata = {}


def load_artifacts():
    """Load model and preprocessor into memory."""
    global _model, _preprocessor, _model_metadata
    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        _model_metadata = {
            "name": "churn_xgboost",
            "version": "1.0.0",
            "trained_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(),
            "path": str(MODEL_PATH),
        }
    if PREPROCESSOR_PATH.exists():
        _preprocessor = joblib.load(PREPROCESSOR_PATH)


@app.on_event("startup")
async def startup():
    load_artifacts()


# ── Request / Response schemas ───────────────────────────────
class ChurnRequest(BaseModel):
    gender: str = Field(..., example="Male")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.0)
    TotalCharges: float = Field(..., example=840.0)


class ChurnResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_metadata: dict


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_metadata=_model_metadata,
    )


@app.get("/model-info")
async def model_info():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return _model_metadata


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _preprocessor is None:
        raise HTTPException(status_code=503, detail="Preprocessor not loaded")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([request.dict()])

        # Feature engineering (same as training)
        from src.features.engineer import engineer_features
        input_df = engineer_features(input_df)

        # Preprocess
        X = _preprocessor.transform(input_df)

        # Predict
        prob = float(_model.predict_proba(X)[0, 1])
        prediction = prob >= 0.5

        # Risk level
        if prob >= 0.7:
            risk = "high"
        elif prob >= 0.4:
            risk = "medium"
        else:
            risk = "low"

        return ChurnResponse(
            churn_probability=round(prob, 4),
            churn_prediction=bool(prediction),
            risk_level=risk,
            model_version=_model_metadata.get("version", "unknown"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
