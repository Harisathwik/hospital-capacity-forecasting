import pandas as pd
from typing import Dict, Any
import json
from pathlib import Path


def compute_eda_report(df: pd.DataFrame, target_col: str = "staffed_adult_icu_bed_occupancy") -> Dict[str, Any]:
    """
    Compute exploratory data analysis report for the DataFrame.
    Returns a dictionary with summary statistics, missingness, and target info.
    """
    report = {}

    # Basic info
    report["n_rows"] = int(len(df))
    report["n_cols"] = int(len(df.columns))
    report["columns"] = list(df.columns)

    # Missingness
    missingness = df.isna().mean().to_dict()
    report["missingness_fraction"] = {k: float(v) for k, v in missingness.items()}
    report["missingness_count"] = {k: int(v) for k, v in df.isna().sum().items()}

    # Data types
    report["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}

    # Summary statistics for numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        report["numeric_summary"] = df[numeric_cols].describe().to_dict()
    else:
        report["numeric_summary"] = {}

    # Target variable info (if present)
    if target_col in df.columns:
        target_series = df[target_col].dropna()
        report["target_mean"] = float(target_series.mean())
        report["target_std"] = float(target_series.std())
        report["target_min"] = float(target_series.min())
        report["target_max"] = float(target_series.max())
        report["target_median"] = float(target_series.median())
        # Skewness and kurtosis
        report["target_skew"] = float(target_series.skew())
        report["target_kurtosis"] = float(target_series.kurtosis())
    else:
        report["target_mean"] = None
        report["target_std"] = None
        report["target_min"] = None
        report["target_max"] = None
        report["target_median"] = None
        report["target_skew"] = None
        report["target_kurtosis"] = None

    # Save report to file (optional, but we can do it in the step)
    return report


def save_eda_report(report: Dict[str, Any], output_dir: str = "data/reports") -> str:
    """
    Save EDA report as JSON file.
    Returns the path to the saved file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    file_path = output_path / f"eda_report_{timestamp}.json"
    with open(file_path, "w") as f:
        json.dump(report, f, indent=2)
    return str(file_path)


if __name__ == "__main__":
    # Example usage (requires data)
    from src.core.data_pull import pull_hhs_data
    df, _ = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy,inpatient_beds_used,adult_icu_bed_utilization",
        where_clause="date>='2024-01-01T00:00:00.000'",
        limit=1000
    )
    report = compute_eda_report(df)
    print(json.dumps(report, indent=2))
    # Uncomment to save
    # saved_path = save_eda_report(report)
    # print(f"Report saved to {saved_path}")