"""Feature engineering for churn prediction."""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List
import joblib


# Column groups
BINARY_COLS = [
    "gender", "Partner", "Dependents", "PhoneService",
    "PaperlessBilling"
]

MULTI_CATEGORY_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract",
    "PaymentMethod"
]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn preprocessing pipeline."""
    binary_pipeline = Pipeline([
        ("label_encode", LabelEncoderProxy())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", LabelEncoderProxy(), BINARY_COLS + MULTI_CATEGORY_COLS),
        ],
        remainder="drop"
    )
    return preprocessor


class LabelEncoderProxy:
    """Simple label encoder that works in ColumnTransformer."""

    def fit(self, X, y=None):
        self.encoders_ = {}
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        for col in X_df.columns:
            le = LabelEncoder()
            le.fit(X_df[col].astype(str))
            self.encoders_[col] = le
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        for col in X_df.columns:
            if col in self.encoders_:
                X_df[col] = self.encoders_[col].transform(X_df[col].astype(str))
        return X_df.values

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features."""
    df = df.copy()

    # Average monthly charge (handles tenure=0)
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"]
    )

    # Charge difference (are they paying more than average?)
    df["ChargeDiff"] = df["MonthlyCharges"] - df["AvgMonthlyCharge"]

    # Tenure group
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=["0-12", "12-24", "24-48", "48-60", "60+"],
        right=False,
    ).astype(str)

    # Is new customer
    df["IsNewCustomer"] = (df["tenure"] <= 12).astype(int)

    # Has multiple services
    service_cols = [
        "PhoneService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["NumServices"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == "Yes"), axis=1
    )

    return df


def get_feature_columns() -> List[str]:
    """Return all feature columns after engineering."""
    return NUMERIC_COLS + BINARY_COLS + MULTI_CATEGORY_COLS + [
        "AvgMonthlyCharge", "ChargeDiff", "TenureGroup",
        "IsNewCustomer", "NumServices"
    ]
