"""Tests for data validation — src/core/validation.py and src/data/validator.py."""
import pandas as pd
import numpy as np
import pytest
from datetime import datetime, timedelta

from src.core.validation import (
    validate_raw_data,
    _check_required_columns,
    _check_date_parseable,
    _check_numeric_parseable,
    _check_duplicate_state_date,
    _check_missingness,
    _check_utilization_range,
    _check_freshness,
)


def _make_valid_df(n_rows=100, stale=False):
    """Create a valid test DataFrame mimicking HHS hospital data."""
    states = ["CA", "TX", "NY", "FL", "IL"]
    dates = pd.date_range("2024-01-01", periods=n_rows // len(states) + 1, freq="D")
    rows = []
    for state in states:
        for d in dates[: n_rows // len(states)]:
            date_val = str(d.date()) if not stale else str((d - timedelta(days=10)).date())
            rows.append({
                "state": state,
                "date": date_val,
                "staffed_adult_icu_bed_occupancy": np.random.uniform(100, 5000),
                "inpatient_beds_used": np.random.uniform(200, 10000),
                "adult_icu_bed_utilization": np.random.uniform(0.3, 0.95),
            })
    return pd.DataFrame(rows[:n_rows])


class TestHardChecks:
    def test_required_columns_present(self):
        df = _make_valid_df()
        assert _check_required_columns(df) == True

    def test_required_columns_missing(self):
        df = pd.DataFrame({"state": ["CA"], "date": ["2024-01-01"]})
        assert _check_required_columns(df) == False

    def test_date_parseable_valid(self):
        df = _make_valid_df()
        assert _check_date_parseable(df) == True

    def test_date_parseable_invalid(self):
        df = _make_valid_df()
        df["date"] = "not-a-date"
        assert _check_date_parseable(df) == False

    def test_numeric_parseable_valid(self):
        df = _make_valid_df()
        assert _check_numeric_parseable(df) == True

    def test_numeric_parseable_all_nan(self):
        df = _make_valid_df()
        df["staffed_adult_icu_bed_occupancy"] = "not_numeric"
        assert _check_numeric_parseable(df) == False

    def test_no_duplicate_state_date_clean(self):
        df = _make_valid_df()
        assert _check_duplicate_state_date(df) == True

    def test_duplicate_state_date_flagged(self):
        df = _make_valid_df()
        dupes = df.iloc[: len(df) // 5].copy()
        df = pd.concat([df, dupes], ignore_index=True)
        assert _check_duplicate_state_date(df) == False


class TestSoftChecks:
    def test_missingness_within_threshold(self):
        df = _make_valid_df()
        passed, details = _check_missingness(df)
        assert passed == True

    def test_missingness_exceeds_threshold(self):
        df = _make_valid_df()
        df.loc[df.sample(frac=0.5).index, "staffed_adult_icu_bed_occupancy"] = np.nan
        passed, details = _check_missingness(df)
        assert passed == False

    def test_utilization_in_range(self):
        df = _make_valid_df()
        passed, details = _check_utilization_range(df)
        assert passed == True

    def test_utilization_out_of_range(self):
        df = _make_valid_df()
        df.loc[0, "adult_icu_bed_utilization"] = 5.0
        passed, details = _check_utilization_range(df)
        assert passed == False

    def test_freshness_with_large_tolerance(self):
        """Freshness passes with large tolerance (2024 data is stale)."""
        df = _make_valid_df(stale=False)
        passed, details = _check_freshness(df, max_days_stale=999)
        assert passed == True

    def test_freshness_stale_data(self):
        df = _make_valid_df(stale=True)
        passed, details = _check_freshness(df, max_days_stale=2)
        assert passed == False


class TestValidateRawData:
    def test_valid_data_passes_with_large_freshness(self):
        df = _make_valid_df()
        # Data from 2024 is stale — override freshness in soft checks
        report = validate_raw_data(df)
        # With default 2-day freshness, this will warn. Check hard checks pass.
        for name, passed in report["hard_checks"].items():
            assert passed == True, f"Hard check {name} failed"

    def test_missing_columns_fails_hard(self):
        df = pd.DataFrame({"state": ["CA"], "date": ["2024-01-01"]})
        report = validate_raw_data(df)
        assert report["overall_status"] == "FAIL"

    def test_bad_dates_fails_hard(self):
        df = _make_valid_df()
        df["date"] = "bad-date"
        report = validate_raw_data(df)
        assert report["overall_status"] == "FAIL"

    def test_high_missingness_warns(self):
        df = _make_valid_df()
        df.loc[df.sample(frac=0.5).index, "staffed_adult_icu_bed_occupancy"] = np.nan
        report = validate_raw_data(df)
        assert any("missingness" in i.lower() for i in report["issues"])


class TestValidatorModule:
    def test_validation_report_import(self):
        from src.data.validator import ValidationReport
        report = ValidationReport()
        assert report.passed == True
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_validation_report_add_error(self):
        from src.data.validator import ValidationReport
        report = ValidationReport()
        report.add_error("something broke")
        assert report.passed == False
        assert len(report.errors) == 1

    def test_validation_report_add_warning(self):
        from src.data.validator import ValidationReport
        report = ValidationReport()
        report.add_warning("soft issue")
        assert report.passed == True
        assert len(report.warnings) == 1
