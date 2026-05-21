"""Feature engineering: lags, rolling stats, calendar features, and target construction."""


from typing import Optional

import pandas as pd
import numpy as np


COL_OCCUPANCY = "staffed_adult_icu_bed_occupancy"
COL_UTILIZATION = "adult_icu_bed_utilization"
COL_INPATIENT_USED = "inpatient_beds_used"
COL_COVID_CONF = "previous_day_admission_adult_covid_confirmed"
COL_COVID_SUSP = "previous_day_admission_adult_covid_suspected"
COL_INFLUENZA = "previous_day_admission_influenza_confirmed"
COL_STAFF_SHORT_YES = "critical_staffing_shortage_today_yes"
COL_STAFF_SHORT_NO = "critical_staffing_shortage_today_no"
COL_STAFF_SHORT_ANTICIPATED = "anticipated_within_week_yes"

LAG_COLS = [
    COL_OCCUPANCY,
    COL_UTILIZATION,
    COL_INPATIENT_USED,
    COL_COVID_CONF,
    COL_COVID_SUSP,
    COL_INFLUENZA,
]

ROLLING_WINDOWS = [7, 14, 30]


def _add_lags(df: pd.DataFrame, col: str, lags: list[int] = [1, 7, 14]) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df.groupby("state")[col].shift(lag)
    return df


def _add_rolling(df: pd.DataFrame, col: str, windows: list[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    grp = df.groupby("state")[col]
    for w in windows:
        df[f"{col}_rolling_mean_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).mean())
        df[f"{col}_rolling_std_{w}"] = grp.transform(lambda x: x.rolling(w, min_periods=1).std())
    # trend proxy: short-term vs medium-term momentum
    if 7 in windows and 14 in windows:
        df[f"{col}_trend_7_14"] = (
            df[f"{col}_rolling_mean_7"] - df[f"{col}_rolling_mean_14"]
        )
    return df


def _add_missingness_indicators(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[f"{col}_is_missing"] = df[col].isna().astype(int)
    return df


def _add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_targets(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """Create direct multi-output targets: y_t+1 ... y_t+horizon."""
    for h in range(1, horizon + 1):
        df[f"y_target_{h}"] = df.groupby("state")[COL_OCCUPANCY].shift(-h)
    return df


def build_features(
    df: pd.DataFrame,
    horizon: int = 7,
    state_medians: Optional[dict] = None,
) -> tuple[pd.DataFrame, dict]:
    """Build feature frame from raw validated data.

    Returns:
        df: feature DataFrame (with targets)
        state_medians: per-state medians for imputation (fit on training data only)
    """
    df = df.sort_values(["state", "date"]).copy()

    # Missingness indicators (before imputation)
    _add_missingness_indicators(df, LAG_COLS)

    # Compute per-state medians for imputation
    if state_medians is None:
        state_medians = df.groupby("state")[LAG_COLS].median().to_dict("index")

    for col in LAG_COLS:
        if col in df.columns:
            df[col] = df.groupby("state")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Lags
    for col in LAG_COLS:
        _add_lags(df, col)

    # Rolling stats on occupancy + utilization
    _add_rolling(df, COL_OCCUPANCY)
    _add_rolling(df, COL_UTILIZATION)

    # Calendar features
    _add_calendar(df)

    # Staffing shortage features (already numeric counts)
    for col in [COL_STAFF_SHORT_YES, COL_STAFF_SHORT_NO, COL_STAFF_SHORT_ANTICIPATED]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Targets (direct multi-output)
    _add_targets(df, horizon=horizon)

    # Drop rows where any target is NaN (last `horizon` days per state)
    target_cols = [f"y_target_{h}" for h in range(1, horizon + 1)]
    df = df.dropna(subset=target_cols).reset_index(drop=True)

    return df, state_medians


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (exclude targets, state, date)."""
    exclude = {"state", "date"} | {f"y_target_{h}" for h in range(1, 8)}
    return [c for c in df.columns if c not in exclude]
