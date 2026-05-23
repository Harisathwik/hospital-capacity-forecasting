import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    test_size_days: int = 28,  # holdout last 4 weeks per state
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test using temporal split per state.
    For each state, we take the most recent `test_size_days` days as test,
    and the rest as train.
    Returns train and test DataFrames.
    """
    # Ensure sorted by state and date
    df = df.sort_values([state_col, date_col]).reset_index(drop=True)

    train_list = []
    test_list = []

    for state, group in df.groupby(state_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        # If the group has fewer than test_size_days*2, we might adjust.
        # For simplicity, we take the last test_size_days as test if available.
        if len(group) > test_size_days:
            test = group.iloc[-test_size_days:]
            train = group.iloc[:-test_size_days]
        else:
            # If not enough data, we put everything in train and none in test.
            train = group
            test = pd.DataFrame(columns=group.columns)
        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list, ignore_index=True)
    test_df = pd.concat(test_list, ignore_index=True)

    return train_df, test_df


def asymmetric_rmse(y_true: np.ndarray, y_pred: np.ndarray, underprediction_weight: float = 3.0) -> float:
    """
    Compute asymmetric RMSE where underprediction errors are weighted more.
    Formula: sqrt( mean( weight * (y_true - y_pred)^2 for underprediction, (y_true - y_pred)^2 for overprediction ) )
    """
    errors = y_true - y_pred
    squared_errors = errors ** 2
    # Apply weight to underprediction errors (where y_true > y_pred, i.e., errors > 0)
    weighted_squared_errors = np.where(errors > 0, underprediction_weight * squared_errors, squared_errors)
    return np.sqrt(np.mean(weighted_squared_errors))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a set of metrics: asymmetric RMSE (primary), MAE, underprediction rate.
    """
    from sklearn.metrics import mean_absolute_error

    # Asymmetric RMSE
    asym_rmse = asymmetric_rmse(y_true, y_pred, underprediction_weight=3.0)
    # MAE
    mae = mean_absolute_error(y_true, y_pred)
    # Underprediction rate: proportion of predictions that are below actual
    underprediction_rate = np.mean(y_pred < y_true)

    return {
        "asymmetric_rmse": float(asym_rmse),
        "mae": float(mae),
        "underprediction_rate": float(underprediction_rate),
    }


if __name__ == "__main__":
    # Example usage
    from src.core.data_pull import pull_hhs_data
    from src.core.feature_engineering import build_features
    from src.core.target_construction import build_targets
    df, _ = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy,inpatient_beds_used,adult_icu_bed_utilization",
        where_clause="date>='2024-01-01T00:00:00.000'",
        limit=1000
    )
    df_with_targets, target_cols = build_targets(df)
    features, feature_cols = build_features(df_with_targets)
    # For simplicity, we'll use the first target horizon as the target for this example
    target_col = target_cols[0]  # e.g., 'staffed_adult_icu_bed_occupancy_t_plus_1'
    # Prepare X and y
    X = features
    y = df_with_targets[target_col].values
    # Split
    train_df, test_df = temporal_split(df_with_targets)
    # We need to align features and target with the split indices.
    # For simplicity, we'll just split the X and y arrays using the same indices as the data split.
    # But note: our features and target are already aligned with df_with_targets.
    # So we can do:
    train_indices = train_df.index
    test_indices = test_df.index
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    # Compute metrics on a dummy prediction (just to test)
    y_pred_dummy = np.full_like(y_test, np.mean(y_train))
    metrics = compute_metrics(y_test, y_pred_dummy)
    print(f"Metrics: {metrics}")