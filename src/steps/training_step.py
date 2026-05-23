from zenml import step
from typing import Tuple, Any, List, Dict
import pandas as pd
from src.core.training import train_model


@step
def trainer_step(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    model_type: str = "ridge",
    model_params: Dict[str, Any] = None,
) -> Tuple[Any, Any, List[str], List[str], Dict[str, float]]:
    """
    ZenML step to train a model (Ridge or XGBoost) for multi-target regression.
    Returns the model, preprocessing pipeline, feature names, target names, and metrics.
    """
    if model_params is None:
        model_params = {}
    model, preprocessing, feature_names, target_names, metrics = train_model(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        model_type=model_type,
        **model_params,
    )
    return model, preprocessing, feature_names, target_names, metrics