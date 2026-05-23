from zenml import step
from typing import Tuple, List, Any
import pandas as pd
from src.core.feature_engineering import build_features


@step
def feature_engineering_step(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    target_col: str = "staffed_adult_icu_bed_occupancy",
    lags: List[int] = [1, 7, 14],
    rolling_windows: List[int] = [7, 14, 30],
    include_calendar: bool = True,
    include_ops_signals: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    ZenML step to build features for ICU occupancy forecasting.
    Returns the feature DataFrame (X) and the list of feature column names.
    Note: The target variable is not included in the returned DataFrame.
    """
    features, feature_cols = build_features(
        df=df,
        state_col=state_col,
        date_col=date_col,
        target_col=target_col,
        lags=lags,
        rolling_windows=rolling_windows,
        include_calendar=include_calendar,
        include_ops_signals=include_ops_signals,
    )
    return features, feature_cols