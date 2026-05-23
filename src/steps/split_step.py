from zenml import step
from typing import Tuple, Any
import pandas as pd
from src.core.training import temporal_split


@step
def split_step(
    df: pd.DataFrame,
    state_col: str = "state",
    date_col: str = "date",
    test_size_days: int = 28,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML step to split data into train and test using temporal split per state.
    Returns train and test DataFrames.
    """
    train_df, test_df = temporal_split(
        df=df,
        state_col=state_col,
        date_col=date_col,
        test_size_days=test_size_days,
    )
    return train_df, test_df