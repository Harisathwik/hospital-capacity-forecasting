import pandas as pd
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def validate_raw_data(df: pd.DataFrame, missingness_thresholds: Dict[str, float] = None) -> Dict[str, Any]:
    """
    Validate raw HHS hospital capacity DataFrame.
    Returns validation report with pass/fail status and details.
    Hard checks: fail the run if any fail.
    Soft checks: warn but may not fail if severe.
    """
    report = {
        "hard_checks": {},
        "soft_checks": {},
        "overall_status": "PASS",
        "issues": []
    }

    # Hard checks (must pass)
    hard_checks = [
        ("required_columns", _check_required_columns(df)),
        ("date_parseable", _check_date_parseable(df)),
        ("numeric_parseable", _check_numeric_parseable(df)),
        ("no_duplicate_state_date", _check_duplicate_state_date(df)),
    ]

    for name, passed in hard_checks:
        report["hard_checks"][name] = passed
        if not passed:
            report["overall_status"] = "FAIL"
            report["issues"].append(f"Hard check failed: {name}")

    # Soft checks (warn)
    soft_checks = [
        ("missingness_within_threshold", _check_missingness(df, missingness_thresholds)),
        ("utilization_range", _check_utilization_range(df)),
        ("freshness_check", _check_freshness(df)),
    ]

    for name, result in soft_checks:
        # result is tuple (passed, details)
        passed, details = result
        report["soft_checks"][name] = {"passed": passed, "details": details}
        if not passed:
            report["issues"].append(f"Soft check warning: {name} - {details}")

    return report


def _check_required_columns(df: pd.DataFrame) -> bool:
    required = {"state", "date", "staffed_adult_icu_bed_occupancy"}
    return required.issubset(set(df.columns))


def _check_date_parseable(df: pd.DataFrame) -> bool:
    try:
        pd.to_datetime(df["date"], errors="raise")
        return True
    except Exception:
        return False


def _check_numeric_parseable(df: pd.DataFrame) -> bool:
    # Check a few key numeric columns
    numeric_cols = ["staffed_adult_icu_bed_occupancy", "inpatient_beds_used", "adult_icu_bed_utilization"]
    for col in numeric_cols:
        if col in df.columns:
            # Try to convert to numeric, coercing errors
            converted = pd.to_numeric(df[col], errors="coerce")
            # If all are NaN after conversion, that's a problem (but we allow some missingness)
            # We'll check that not ALL are NaN (i.e., at least some values are numeric)
            if converted.isna().all():
                return False
    return True


def _check_duplicate_state_date(df: pd.DataFrame) -> bool:
    if "state" in df.columns and "date" in df.columns:
        duplicates = df.duplicated(subset=["state", "date"], keep=False)
        duplicate_count = duplicates.sum()
        # Fail if more than 1% duplicates
        return duplicate_count < (0.01 * len(df))
    return True


def _check_missingness(df: pd.DataFrame, thresholds: Dict[str, float] = None) -> Tuple[bool, str]:
    if thresholds is None:
        # Default thresholds from problem statement: ICU ~9%, influenza ~14%
        thresholds = {
            "staffed_adult_icu_bed_occupancy": 0.12,  # alert if >12%
            "inpatient_beds_used": 0.10,
            "adult_icu_bed_utilization": 0.10,
        }
    issues = []
    for col, max_missing in thresholds.items():
        if col in df.columns:
            missing_ratio = df[col].isna().mean()
            if missing_ratio > max_missing:
                issues.append(f"{col}: {missing_ratio:.1%} > {max_missing:.1%}")
    if issues:
        return False, "; ".join(issues)
    return True, "All missingness within thresholds"


def _check_utilization_range(df: pd.DataFrame) -> Tuple[bool, str]:
    if "adult_icu_bed_utilization" in df.columns:
        # Utilization should be between 0 and 1.5 (allowing for overcapacity)
        util = pd.to_numeric(df["adult_icu_bed_utilization"], errors="coerce")
        out_of_range = ((util < 0) | (util > 1.5)).sum()
        if out_of_range > 0:
            return False, f"{out_of_range} rows with utilization outside [0, 1.5]"
    return True, "Utilization within expected range"


def _check_freshness(df: pd.DataFrame, max_days_stale: int = 2) -> Tuple[bool, str]:
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(df["date"])
            latest_date = dates.max()
            days_stale = (pd.Timestamp.now(tz='UTC').tz_localize(None) - latest_date).days
            if days_stale > max_days_stale:
                return False, f"Latest data is {days_stale} days stale (>{max_days_stale} days)"
        except Exception:
            return False, "Could not parse dates for freshness check"
    return True, "Data is fresh enough"


if __name__ == "__main__":
    # Simple test
    from src.core.data_pull import pull_hhs_data
    df, _ = pull_hhs_data(
        select_cols="state,date,staffed_adult_icu_bed_occupancy,inpatient_beds_used,adult_icu_bed_utilization",
        where_clause="date>='2024-01-01T00:00:00.000'",
        limit=1000
    )
    report = validate_raw_data(df)
    print(f"Overall status: {report['overall_status']}")
    for check, passed in report["hard_checks"].items():
        print(f"  Hard {check}: {'PASS' if passed else 'FAIL'}")
    for check, result in report["soft_checks"].items():
        print(f"  Soft {check}: {'PASS' if result['passed'] else 'FAIL'} - {result['details']}")