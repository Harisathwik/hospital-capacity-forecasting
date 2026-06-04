"""Tests for feature engineering — src/features/engineer.py and src/core/target_construction.py."""
import pandas as pd
import numpy as np
import pytest

from src.core.target_construction import build_targets


def _make_full_feature_df(n_days=60, states=None):
    """Create DataFrame with ALL columns that build_features expects."""
    if states is None:
        states = ["CA", "TX"]
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for state in states:
        for d in dates:
            rows.append({
                "state": state,
                "date": d,
                "staffed_adult_icu_bed_occupancy": np.random.uniform(100, 5000),
                "inpatient_beds": np.random.uniform(200, 10000),
                "inpatient_beds_used": np.random.uniform(150, 9000),
                "inpatient_beds_used_covid": np.random.uniform(0, 2000),
                "inpatient_beds_utilization": np.random.uniform(0.3, 0.95),
                "total_staffed_adult_icu_beds": np.random.uniform(100, 6000),
                "adult_icu_bed_utilization": np.random.uniform(0.3, 0.95),
                "previous_day_admission_adult_covid_confirmed": np.random.uniform(0, 100),
                "previous_day_admission_adult_covid_suspected": np.random.uniform(0, 50),
                "previous_day_admission_influenza_confirmed": np.random.uniform(0, 30),
                "total_patients_hospitalized_confirmed_influenza": np.random.uniform(0, 500),
                "critical_staffing_shortage_today_yes": 0,
                "critical_staffing_shortage_today_no": 1,
                "critical_staffing_shortage_anticipated_within_week_yes": 0,
                "critical_staffing_shortage_anticipated_within_week_no": 1,
            })
    return pd.DataFrame(rows)


def _make_simple_df(n_days=30, states=None):
    """Simple DF for target construction tests."""
    if states is None:
        states = ["CA", "TX"]
    rows = []
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    for state in states:
        for d in dates:
            rows.append({
                "state": state,
                "date": d,
                "staffed_adult_icu_bed_occupancy": np.random.uniform(100, 5000),
            })
    return pd.DataFrame(rows)


class TestTargetConstruction:
    def test_build_targets_creates_columns(self):
        df = _make_simple_df(n_days=30)
        result, target_cols = build_targets(df, target_col="staffed_adult_icu_bed_occupancy")
        assert result is not None
        assert len(target_cols) > 0

    def test_build_targets_drops_future_nan(self):
        df = _make_simple_df(n_days=30)
        result, target_cols = build_targets(df, target_col="staffed_adult_icu_bed_occupancy")
        n_states = df["state"].nunique()
        expected_max_rows = len(df) - 7 * n_states
        assert len(result) <= expected_max_rows

    def test_build_targets_no_leakage(self):
        df = _make_simple_df(n_days=30)
        result, target_cols = build_targets(df, target_col="staffed_adult_icu_bed_occupancy")
        for col in target_cols:
            if col in result.columns:
                assert result[col].notna().all(), f"{col} has NaN — possible leakage"


class TestFeatureEngineering:
    def test_build_features_imports(self):
        from src.features.engineer import build_features
        assert callable(build_features)

    def test_build_features_returns_tuple(self):
        from src.features.engineer import build_features
        df = _make_full_feature_df(n_days=60)
        result = build_features(df, horizon=7)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_build_features_produces_extra_columns(self):
        from src.features.engineer import build_features
        df = _make_full_feature_df(n_days=60)
        features_df, meta = build_features(df, horizon=7)
        assert features_df.shape[1] > df.shape[1], "No new features produced"

    def test_build_features_has_lag_or_rolling(self):
        from src.features.engineer import build_features
        df = _make_full_feature_df(n_days=60)
        features_df, meta = build_features(df, horizon=7)
        lag_or_rolling = [c for c in features_df.columns
                         if "lag" in c.lower() or "rolling" in c.lower()]
        assert len(lag_or_rolling) > 0, "No lag or rolling features produced"

    def test_build_features_with_state_medians(self):
        from src.features.engineer import build_features
        df = _make_full_feature_df(n_days=60)
        features_df, meta = build_features(df, horizon=7, state_medians=None)
        assert features_df is not None
