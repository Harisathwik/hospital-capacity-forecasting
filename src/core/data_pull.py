import os
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
import requests


def pull_hhs_data(
    dataset_id: str = "g62h-syeh",
    base_url: str = "https://healthdata.gov/resource",
    select_cols: str = None,
    where_clause: str = None,
    order_clause: str = "date ASC",
    limit: int = 500000,
    raw_data_dir: str = "data/raw",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Deterministic pull of HHS hospital capacity dataset from Socrata API.
    Returns DataFrame and pull manifest (metadata).
    """
    # Build query parameters
    params = {}
    if select_cols:
        params["$select"] = select_cols
    if where_clause:
        params["$where"] = where_clause
    if order_clause:
        params["$order"] = order_clause
    if limit:
        params["$limit"] = str(limit)

    url = f"{base_url}/{dataset_id}.csv"
    response = requests.get(url, params=params)
    response.raise_for_status()

    # Save raw CSV
    raw_dir = Path(raw_data_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_dir / f"hhs_hospital_capacity_{dataset_id}_{timestamp}.csv"
    with open(raw_path, "wb") as f:
        f.write(response.content)

    # Load into DataFrame
    df = pd.read_csv(raw_path)

    # Create pull manifest
    manifest = {
        "pull_timestamp_utc": timestamp,
        "dataset_id": dataset_id,
        "source_url": url,
        "query_params": params,
        "row_count": int(len(df)),
        "date_min": str(df["date"].min()) if "date" in df.columns else None,
        "date_max": str(df["date"].max()) if "date" in df.columns else None,
        "columns": list(df.columns),
        "schema_hash": _schema_hash(df),
        "raw_file_path": str(raw_path),
    }

    # Save manifest alongside raw file
    manifest_path = raw_path.with_suffix(".json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return df, manifest


def _schema_hash(df: pd.DataFrame) -> str:
    """Simple hash of column names and dtypes."""
    schema_str = "".join(f"{col}:{str(dtype)}" for col, dtype in df.dtypes.items())
    return hashlib.md5(schema_str.encode()).hexdigest()


if __name__ == "__main__":
    # Example usage
    df, manifest = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy,inpatient_beds_used,adult_icu_bed_utilization",
        where_clause="date>='2020-01-01T00:00:00.000'",
    )
    print(f"Pulled {manifest['row_count']} rows")
    print(f"Manifest saved to {manifest['raw_file_path']}.json")