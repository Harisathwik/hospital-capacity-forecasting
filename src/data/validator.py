"""Data validation: schema, type, range, and missingness checks."""


from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "state",
    "date",
    "staffed_adult_icu_bed_occupancy",
    "total_staffed_adult_icu_beds",
    "adult_icu_bed_utilization",
    "inpatient_beds",
    "inpatient_beds_used",
]

# missingness thresholds (from our dataset audit)
MISSINGNESS_WARN = {
    "staffed_adult_icu_bed_occupancy": 0.12,
    "adult_icu_bed_utilization": 0.12,
    "previous_day_admission_influenza_confirmed": 0.20,
}

MISSINGNESS_FAIL = {
    "staffed_adult_icu_bed_occupancy": 0.25,
    "state": 0.01,
    "date": 0.01,
}


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def __str__(self) -> str:
        lines = [f"Validation {'PASSED' if self.passed else 'FAILED'}"]
        if self.errors:
            lines.append("ERRORS:")
            for e in self.errors:
                lines.append(f"  - {e}")
        if self.warnings:
            lines.append("WARNINGS:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def validate(df: pd.DataFrame) -> ValidationReport:
    report = ValidationReport()

    # 1) Required columns present
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        report.add_error(f"Missing required columns: {missing_cols}")

    # 2) Date parses
    if "date" in df.columns:
        null_dates = df["date"].isna().sum()
        if null_dates > 0:
            report.add_error(f"{null_dates} rows with unparseable dates")

    # 3) Primary key uniqueness (state, date)
    if "state" in df.columns and "date" in df.columns:
        dupes = df.duplicated(subset=["state", "date"]).sum()
        if dupes > 0:
            report.add_warning(f"{dupes} duplicate (state, date) rows")

    # 4) Missingness checks
    for col, threshold in MISSINGNESS_FAIL.items():
        if col in df.columns:
            pct = df[col].isna().mean()
            if pct > threshold:
                report.add_error(
                    f"Column '{col}' missingness {pct:.1%} exceeds fail threshold {threshold:.1%}"
                )

    for col, threshold in MISSINGNESS_WARN.items():
        if col in df.columns:
            pct = df[col].isna().mean()
            if pct > threshold:
                report.add_warning(
                    f"Column '{col}' missingness {pct:.1%} exceeds warn threshold {threshold:.1%}"
                )

    # 5) Range checks
    if "adult_icu_bed_utilization" in df.columns:
        bad = (df["adult_icu_bed_utilization"] > 1.5).sum()
        if bad > 0:
            report.add_warning(f"{bad} rows with utilization > 1.5")

    if "staffed_adult_icu_bed_occupancy" in df.columns and "total_staffed_adult_icu_beds" in df.columns:
        bad = (
            df["staffed_adult_icu_bed_occupancy"]
            > df["total_staffed_adult_icu_beds"] * 1.1
        ).sum()
        if bad > 0:
            report.add_warning(
                f"{bad} rows where occupancy > 110% of total beds"
            )

    return report
