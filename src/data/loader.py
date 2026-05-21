"""Data ingestion: deterministic Socrata pull with local caching."""


import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.core.config import load_config

Socrata_URL = "https://healthdata.gov/resource/g62h-syeh.json"
DEFAULT_COLUMNS = [
    "state",
    "date",
    "total_staffed_adult_icu_beds",
    "staffed_adult_icu_bed_occupancy",
    "adult_icu_bed_utilization",
    "inpatient_beds",
    "inpatient_beds_used",
    "inpatient_bed_utilization",
    "previous_day_admission_adult_covid_confirmed",
    "previous_day_admission_adult_covid_suspected",
    "previous_day_admission_influenza_confirmed",
    "total_staffed_pediatric_icu_beds",
    "staffed_pediatric_icu_bed_occupancy",
    "critical_staffing_shortage_today_yes",
    "critical_staffing_shortage_today_no",
    "anticipated_within_week_yes",
    "anticipated_within_week_no",
]


def _build_query(columns: list[str], min_date: str = "2020-01-01") -> dict:
    return {
        "$select": ",".join(columns),
        "$where": f"date >= '{min_date}T00:00:00.000'",
        "$order": "date ASC",
        "$limit": 5_000_000,
    }


def pull_from_socrata(
    config: Optional[dict] = None,
    save: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Pull HHS hospital capacity data from Socrata API.

    Returns:
        df: DataFrame with the pulled data.
        manifest: metadata dict for reproducibility.
    """
    if config is None:
        config = load_config()

    query = _build_query(DEFAULT_COLUMNS)
    resp = requests.get(Socrata_URL, params=query, timeout=120)
    resp.raise_for_status()
    records = resp.json()

    df = pd.DataFrame(records)

    # parse date + numeric
    df["date"] = pd.to_datetime(df["date"], format="ISO8601")
    numeric_cols = [c for c in DEFAULT_COLUMNS if c not in ("state", "date")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    manifest = {
        "pull_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": Socrata_URL,
        "dataset_id": "g62h-syeh",
        "query": query,
        "row_count": len(df),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "schema_hash": hashlib.md5(
            json.dumps(list(df.columns), sort_keys=True).encode()
        ).hexdigest(),
    }

    if save:
        raw_dir = Path(config["data"]["raw_dir"])
        raw_dir.mkdir(parents=True, exist_ok=True)
        out = raw_dir / "hhs_hospital_capacity_g62h-syeh.csv"
        df.to_csv(out, index=False)

        manifest_dir = Path(config["data"]["raw_dir"]) / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"manifest_{manifest['pull_timestamp_utc'][:19].replace(':','')}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    return df, manifest


def load_from_cache(config: Optional[dict] = None) -> pd.DataFrame:
    """Load the cached CSV if it exists, otherwise pull from API."""
    if config is None:
        config = load_config()

    cache_path = Path(config["data"]["raw_dir"]) / "hhs_hospital_capacity_g62h-syeh.csv"
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
        numeric_cols = [c for c in df.columns if c not in ("state", "date")]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    df, _ = pull_from_socrata(config)
    return df
