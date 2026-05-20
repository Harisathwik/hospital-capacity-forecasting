"""ZenML pipeline definition for churn prediction training."""
from typing import Tuple
import pandas as pd
import numpy as np
import xgboost as xgb

from src.data.loader import load_data, validate_data, split_data
from src.features.engineer import engineer_features
from src.models import train_model, evaluate_model, passes_guardrails


# ── ZenML step: load ──────────────────────────────────────────
@step
def load_and_validate() -> pd.DataFrame:
    """Load and validate the raw churn dataset."""
    print("Loading data...")
    df = load_data()
    print(f"Raw shape: {df.shape}")
    df = validate_data(df)
    print(f"Validated shape: {df.shape}")
    print(f"Churn rate: {df['Churn'].mean():.2%}")
    return df


# ── ZenML step: engineer features ─────────────────────────────
@step
def engineer_step(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering."""
    print("Engineering features...")
    df = engineer_features(df)
    print(f"Feature columns: {df.drop(columns=['Churn']).shape[1]}")
    return df


# ── ZenML step: split ─────────────────────────────────────────
@step
def split_step(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train and test sets."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ── ZenML step: preprocess ────────────────────────────────────
@step
def preprocess_step(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit preprocessor on train, transform both."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline

    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges",
                        "AvgMonthlyCharge", "ChargeDiff", "NumServices", "IsNewCustomer"]
    categorical_features = [c for c in X_train.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Save preprocessor
    from pathlib import Path
    import joblib
    prep_path = Path("models/preprocessor.joblib")
    prep_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, prep_path)
    print(f"Processed train shape: {X_train_processed.shape}")

    return X_train_processed, X_test_processed


# ── ZenML step: train ─────────────────────────────────────────
@step
def train_step(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> xgb.XGBClassifier:
    """Train XGBoost classifier."""
    print("Training XGBoost model...")
    model = train_model(X_train, y_train)
    print(f"Model trained with {model.n_estimators} estimators")
    # Convert y_train to numpy if needed
    y_pred = model.predict(X_train)
    from sklearn.metrics import accuracy_score
    train_acc = accuracy_score(
        y_train.values if hasattr(y_train, 'values') else y_train,
        y_pred
    )
    print(f"Train accuracy: {train_acc:.4f}")
    return model


# ── ZenML step: evaluate ──────────────────────────────────────
@step
def evaluate_step(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
) -> dict:
    """Evaluate model and check guardrails."""
    print("Evaluating model...")
    y_test_np = y_test.values if hasattr(y_test, 'values') else y_test
    metrics = evaluate_model(model, X_test, y_test_np)
    passes = passes_guardrails(metrics)

    print(f"F2-Score: {metrics['f2_score']:.4f}")
    print(f"Recall:   {metrics['recall']:.4f}")
    print(f"PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"FPR:      {metrics['fpr']:.4f}")
    print(f"Passes guardrails: {passes}")

    if passes:
        from src.models import save_model
        save_model(model)
        print("Model saved to disk.")

    return {"metrics": metrics, "passes_guardrails": passes}


# ── ZenML pipeline ────────────────────────────────────────────
@pipeline(name="churn_training_pipeline")
def churn_training_pipeline(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end churn prediction training pipeline."""
    df = load_and_validate()
    df = engineer_step(df)
    X_train, X_test, y_train, y_test = split_step(df, test_size, random_state)
    X_train_proc, X_test_proc = preprocess_step(X_train, X_test)
    model = train_step(X_train_proc, y_train)
    evaluate_step(model, X_test_proc, y_test)


if __name__ == "__main__":
    churn_training_pipeline()
