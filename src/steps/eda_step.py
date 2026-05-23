from zenml import step
from typing import Dict, Any
import pandas as pd
from src.core.eda import compute_eda_report, save_eda_report


@step
def eda_step(df: pd.DataFrame, target_col: str = "staffed_adult_icu_bed_occupancy") -> Dict[str, Any]:
    """
    ZenML step to perform exploratory data analysis.
    Returns the EDA report as a dictionary and also saves it as an artifact (JSON).
    """
    # Compute report
    report = compute_eda_report(df, target_col=target_col)
    
    # Save report to file (this will be captured as an artifact by ZenML if we return the path?
    # For now, we just return the dict; we could also save and return the path.
    # Let's save it and return the path as well? But the step output should be the report dict.
    # We'll also save it to a known location for artifact tracking.
    output_dir = "data/reports"
    saved_path = save_eda_report(report, output_dir=output_dir)
    # Optionally, we could also log the path as a metadata or artifact.
    # For simplicity, we return the report and the user can access the saved file.
    # However, ZenML steps should return artifacts or data that can be passed downstream.
    # We'll return the report dict; the saved file is a side effect.
    # In a real scenario, we might want to return the path and let the next step read it.
    # But for now, we keep it simple.
    
    # Also, we can add the saved path to the report for reference.
    report["saved_report_path"] = saved_path
    
    return report