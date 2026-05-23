from zenml import step
from typing import Tuple, Dict, Any
import pandas as pd
from src.core.data_pull import pull_hhs_data
from src.core.validation import validate_raw_data


@step
def data_pull_step(
    dataset_id: str = "g62h-syeh",
    select_cols: str = None,
    where_clause: str = "date>='2020-01-01T00:00:00.000'",
    order_clause: str = "date ASC",
    limit: int = 500000,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    ZenML step to pull and validate HHS hospital capacity data.
    Returns the raw DataFrame and the pull manifest.
    Raises an exception if hard validation checks fail.
    """
    # If select_cols is None or empty string, we don't pass $select parameter
    if select_cols is None or select_cols == "":
        select_cols_param = None
    else:
        select_cols_param = select_cols

    # Pull data
    df, manifest = pull_hhs_data(
        dataset_id=dataset_id,
        select_cols=select_cols_param,
        where_clause=where_clause,
        order_clause=order_clause,
        limit=limit,
    )

    # Validate data
    validation_report = validate_raw_data(df)
    
    # Log validation report (in a real scenario, you might want to log this as an artifact or metric)
    print(f"Validation report: {validation_report}")

    # If any hard checks failed, raise an exception to fail the step
    if validation_report["overall_status"] == "FAIL":
        failed_checks = [name for name, passed in validation_report["hard_checks"].items() if not passed]
        raise ValueError(f"Hard validation checks failed: {failed_checks}")

    return df, manifest