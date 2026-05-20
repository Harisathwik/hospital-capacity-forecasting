"""Data loader and validator for Telco Churn dataset."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple


DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data() -> pd.DataFrame:
    """Load the Telco Customer Churn dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Download from: https://www.kaggle.com/datasets/blastchar/telco-customer-churn"
        )
    df = pd.read_csv(DATA_PATH)
    return df


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the dataset."""
    # Drop customerID — not useful for prediction
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert TotalCharges to numeric (has empty strings)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing TotalCharges with 0 (tenure=0 customers)
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    return df


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test sets."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test
