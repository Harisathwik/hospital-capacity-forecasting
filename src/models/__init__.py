"""Model training and evaluation for churn prediction."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    fbeta_score, precision_score, recall_score,
    average_precision_score, roc_auc_score, confusion_matrix
)
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models"


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Dict[str, Any] = None,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier for churn prediction."""
    if params is None:
        params = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "eval_metric": "logloss",
            "use_label_encoder": False,
        }

    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    params["scale_pos_weight"] = scale_pos_weight

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Evaluate model and return key metrics."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "f2_score": float(fbeta_score(y_test, y_pred, beta=2)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, y_prob)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "fpr": float(fp / max(fp + tn, 1)),
        "threshold": threshold,
    }
    return metrics


def passes_guardrails(metrics: Dict[str, float], fpr_threshold: float = 0.30) -> bool:
    """Check if model passes business guardrails."""
    # FPR must be below threshold (avoid excessive false alarms)
    # F2-score should be reasonable
    return metrics["fpr"] < fpr_threshold and metrics["f2_score"] > 0.5


def save_model(model: xgb.XGBClassifier, name: str = "churn_model") -> Path:
    """Save model to disk."""
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    path = MODEL_PATH / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str = "churn_model") -> xgb.XGBClassifier:
    """Load model from disk."""
    path = MODEL_PATH / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")
    return joblib.load(path)
