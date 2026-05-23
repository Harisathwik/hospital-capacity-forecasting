"""Model training: Ridge and XGBoost regressors for multi-output forecasting."""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

from src.core.config import load_config


def split_temporal(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_cols: list[str],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and validation sets using a temporal split.

    The split is done by date: earliest data for training, latest for validation.
    Rows with NaN in feature columns are dropped after splitting (lag/rolling boundaries).

    Returns:
        X_train, y_train, X_val, y_val
    """
    df = df.sort_values(["state", "date"]).reset_index(drop=True)

    unique_dates = sorted(df["date"].unique())
    total_days = len(unique_dates)
    val_size_days = config["data"]["validation_size_days"]
    test_size_days = config["data"]["test_size_days"]

    val_start_idx = total_days - val_size_days - test_size_days
    train_dates = unique_dates[:val_start_idx]
    val_dates = unique_dates[val_start_idx:val_start_idx + val_size_days]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()

    def _extract_clean(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = d[feature_cols].values.astype(np.float64)
        y = d[target_cols].values.astype(np.float64)
        valid = ~np.isnan(X).any(axis=1)
        return X[valid], y[valid]

    X_train, y_train = _extract_clean(train_df)
    X_val, y_val = _extract_clean(val_df)

    return X_train, y_train, X_val, y_val


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alpha: float = 1.0,
) -> tuple:
    """Train Ridge regression with multi-output wrapper.

    Returns: (model, y_val_pred)
    """
    model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=42))
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return model, y_val_pred


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
) -> tuple:
    """Train XGBoost with multi-output wrapper.

    Returns: (model, y_val_pred)
    """
    if params is None:
        params = {}

    model = MultiOutputRegressor(
        XGBRegressor(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            objective="reg:squarederror",
            n_jobs=4,
            random_state=42,
            tree_method="hist",
        )
    )
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    return model, y_val_pred
