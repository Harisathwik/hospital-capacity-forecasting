import pandas as pd
from typing import Tuple, List, Any
import numpy as np


def build_targets(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    target_col: str = "staffed_adult_icu_bed_occupancy",
    horizons: List[int] = [1, 2, 3, 4, 5, 6, 7],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build target variables for multi-horizon forecasting.
    For each horizon h, create a column '{target_col}_t_plus_{h}' which is the target shifted by -h.
    This ensures that for a row at date t, the target_t_plus_h is the value at date t+h.
    We then drop rows where any target is missing (i.e., the last max(horizons) dates per state).
    Returns the dataframe with added target columns and a list of target column names.
    """
    # Ensure sorted by state and date
    df = df.sort_values([state_col, date_col]).reset_index(drop=True)

    target_cols = []
    for h in horizons:
        new_col_name = f"{target_col}_t_plus_{h}"
        df[new_col_name] = df.groupby(state_col)[target_col].shift(-h)
        target_cols.append(new_col_name)

    # Determine rows to keep: we need all horizons to be present
    # Drop rows where any target column is NA
    df_with_targets = df.dropna(subset=target_cols).copy()

    return df_with_targets, target_cols


if __name__ == "__main__":
    # Example usage
    from src.core.data_pull import pull_hhs_data
    df, _ = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy",
        where_clause="date>='2024-01-01T00:00:00.000'",
        limit=100
    )
    print(f"Original shape: {df.shape}")
    df_with_targets, target_cols = build_targets(df)
    print(f"Shape after target construction: {df_with_targets.shape}")
    print(f"Target columns: {target_cols}")
    print(df_with_targets[["state", "date", "staffed_adult_icu_bed_occupancy"] + target_cols].head())