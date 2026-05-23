from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from typing import List, Tuple
import numpy as np


def get_numeric_features(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """Return list of numeric column names, excluding specified columns."""
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [col for col in numeric_cols if col not in exclude_cols]


def build_preprocessing_pipeline(numeric_features: List[str]) -> Pipeline:
    """
    Build a preprocessing pipeline that:
    - Adds missing value indicators for numeric features
    - Imputes missing values with median (fit on training data only)
    Returns a sklearn Pipeline.
    """
    # For each numeric feature, we want to impute and add a missing indicator
    # We can use SimpleImputer with add_indicator=True
    preprocessor = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ])
    # Since we apply the same transformation to all numeric features, we can use ColumnTransformer
    # but if all numeric features get the same treatment, we can just apply the pipeline to all.
    # However, ColumnTransformer allows us to apply different transformations to different columns.
    # For simplicity, we apply the same imputation to all numeric features.
    # We'll use ColumnTransformer to apply the imputer to the numeric features and pass through others (if any).
    # But in our case, we will only pass numeric features through this pipeline.
    # So we can do:
    #   preprocessing = ColumnTransformer(
    #       transformers=[
    #           ('num', preprocessor, numeric_features)
    #       ],
    #       remainder='passthrough'
    #   )
    # However, we want to return a pipeline that only processes the numeric features we give it.
    # For the training step, we will separate numeric and non-numeric? Actually, our features are all numeric (after feature engineering).
    # So we can just return the imputer pipeline and apply it to the entire feature matrix.
    # Let's keep it simple: return a Pipeline that does imputation with add_indicator.
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True))
    ])


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from src.core.data_pull import pull_hhs_data
    from src.core.feature_engineering import build_features
    df, _ = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy,inpatient_beds_used,adult_icu_bed_utilization",
        where_clause="date>='2024-01-01T00:00:00.000'",
        limit=1000
    )
    features, _ = build_features(df)
    numeric_features = get_numeric_features(features)
    print(f"Numeric features: {numeric_features[:5]}")
    preprocess = build_preprocessing_pipeline(numeric_features)
    print(preprocess)