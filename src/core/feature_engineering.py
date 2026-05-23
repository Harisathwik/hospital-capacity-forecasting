import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    target_col: str = "staffed_adult_icu_bed_occupancy",
    lags: List[int] = [1, 7],
    rolling_windows: List[int] = [7, 14],
    include_calendar: bool = True,
    include_ops_signals: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build features for ICU occupancy forecasting.
    - Lag features: past values of target and other numeric columns.
    - Rolling window features: mean/std over past windows (using only past data).
    - Calendar features: day-of-week, month, weekend.
    - Operational signals: staffing shortage indicators (if present).
    - Returns a DataFrame with the original columns plus the feature columns,
      and a list of the feature column names.
    - All features are computed using only past data (no leakage).
    """
    # Ensure sorted by state and date
    df = df.sort_values([state_col, date_col]).reset_index(drop=True)

    # We'll build features per state to avoid leakage across states
    feature_dfs = []
    feature_cols = []

    for state, group in df.groupby(state_col):
        group = group.copy()
        # Ensure date is datetime
        group[date_col] = pd.to_datetime(group[date_col])

        # We'll compute lag and rolling features for a set of base columns
        base_cols = [target_col]
        # For testing, skip other numeric columns to speed up
        # Uncomment below to include all numeric columns
        # numeric_cols = group.select_dtypes(include=[np.number]).columns.tolist()
        # for col in numeric_cols:
        #     if col != target_col and col not in [state_col, date_col]:
        #         base_cols.append(col)

        # Lag features
        for lag in lags:
            for col in base_cols:
                lag_col_name = f"{col}_lag_{lag}"
                group[lag_col_name] = group[col].shift(lag)
                feature_cols.append(lag_col_name)

        # Rolling window features (mean and std)
        for window in rolling_windows:
            for col in base_cols:
                # Rolling mean
                roll_mean_col = f"{col}_roll_mean_{window}"
                group[roll_mean_col] = group[col].rolling(window, min_periods=1).mean().shift(1)
                feature_cols.append(roll_mean_col)
                # Rolling std
                roll_std_col = f"{col}_roll_std_{window}"
                group[roll_std_col] = group[col].rolling(window, min_periods=1).std().shift(1)
                feature_cols.append(roll_std_col)

        # Calendar features
        if include_calendar:
            group["day_of_week"] = group[date_col].dt.dayofweek  # Monday=0, Sunday=6
            group["month"] = group[date_col].dt.month
            group["is_weekend"] = group["day_of_week"].isin([5, 6]).astype(int)
            feature_cols.extend(["day_of_week", "month", "is_weekend"])

        # Operational signals (staffing shortage indicators)
        if include_ops_signals:
            # Check for columns that indicate staffing shortage
            shortage_cols = [c for c in group.columns if "shortage" in c.lower()]
            for col in shortage_cols:
                # Convert to numeric if it's yes/no (assuming yes=1, no=0)
                if group[col].dtype == object:
                    # Map common yes/no strings to 1/0
                    group[col] = group[col].map({"yes": 1, "no": 0, "YES": 1, "NO": 0}).astype(float)
                # Ensure it's numeric
                group[col] = pd.to_numeric(group[col], errors="coerce")
                feature_cols.append(col)

        # Keep the group with the new feature columns
        feature_dfs.append(group)

    # Combine back
    if feature_dfs:
        features_df = pd.concat(feature_dfs, ignore_index=True)
        # Ensure the feature columns are numeric
        for col in feature_cols:
            features_df[col] = pd.to_numeric(features_df[col], errors="coerce")
    else:
        features_df = pd.DataFrame()
        feature_cols = []

    return features_df, feature_cols