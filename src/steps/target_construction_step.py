from zenml import step
from typing import Tuple, List, Any
import pandas as pd
from src.core.target_construction import build_targets


@step
def target_construction_step(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    target_col: str = "staffed_adult_icu_bed_occupancy",
    horizons: List[int] = [1, 2, 3, 4, 5, 6, 7],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    ZenML step to build target variables for multi-horizon forecasting.
    Returns the dataframe with added target columns and the list of target column names.
    """
    df_with_targets, target_cols = build_targets(
        df=df,
        state_col=state_col,
        date_col=date_col,
        target_col=target_col,
        horizons=horizons,
    )
    return df_with_targets, target_cols