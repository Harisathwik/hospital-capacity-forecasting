# Problem Statement: Hospital Capacity Forecasting (ICU Occupancy) for Staffing

## Business Context

Hospitals need to forecast near-term ICU demand to staff nurses/doctors appropriately.

Getting this wrong has asymmetric consequences:
- **Understaffing (under-predicting ICU occupancy):** patient safety risk, inability to handle emergency cases, staff burnout, lower quality of care.
- **Overstaffing (over-predicting):** higher labor cost, but acceptable for this project ("we're not short on money").

Currently, many orgs use simple rules of thumb ("yesterday's value" / manual judgment). These break down during regime changes (COVID waves, seasonal surges, reporting shifts).

## ML Formulation

- **Problem type:** Regression (time-series style; temporal splits only)
- **Target variable:** `staffed_adult_icu_bed_occupancy` — adult ICU beds occupied (proxy for near-term ICU staffing demand)
- **Prediction horizon:** 7 days ahead (multi-step forecast)
- **Primary metric:** **Asymmetric RMSE (underprediction ×3)** — because understaffing is far worse than overstaffing
- **Guardrail metrics:** MAE, underprediction rate (% of days forecast < actual), and calibration of prediction intervals (later)
- **Current baseline:** Naive forecast ("yesterday's value")

## Data Summary

### Public dataset used (for this portfolio build)

We will start with a fully public, API-accessible dataset so the entire project is reproducible:

- **Dataset**: HHS / healthdata.gov — COVID-19 Reported Patient Impact and Hospital Capacity
  - Page: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Cap/g62h-syeh
  - Socrata API (JSON): https://healthdata.gov/resource/g62h-syeh.json?$limit=1
  - Socrata API (CSV): https://healthdata.gov/resource/g62h-syeh.csv

### Local snapshot (downloaded by query)

- File: `data/raw/hhs_hospital_capacity_g62h-syeh.csv` (ignored by git; pulled by deterministic query)
- Rows: ~81,713
- States: 54
- Date range: 2020-01-01 → 2024-04-27
- Known missingness: ~9% for ICU fields; ~14% for influenza fields

**Known issues:**
- Missing data during system outages
- ILINet data is weekly (needs interpolation for daily features)
- Hospital data may have reporting delays of 1-2 days

## Feature Engineering

We will engineer features from the dataset first (then optionally add external sources later):

- **Lag features**: ICU occupancy at 1, 7, 14 days ago
- **Rolling statistics**: 7/14/30-day rolling mean and rolling std
- **Calendar features**: day-of-week, month, holiday flags (optional)
- **Operational signals**: inpatient_beds_used, adult_icu_bed_utilization, prior-day admissions, staffing-shortage indicators

## Constraints

- **Latency:** Batch prediction daily (e.g., 6 AM) is sufficient
- **Interpretability:** Hospital administrators need to understand why the model predicts a surge
- **Regulatory/Privacy:** Dataset is public aggregated data; if replaced with internal hospital data later, de-identification and access controls apply

## Success Criteria

The model is in production when:
1. Primary metric (asymmetric RMSE under×3) improves meaningfully vs naive baseline
2. Underprediction rate is below an agreed threshold (ops safety guardrail)
3. Drift detection is operational and alerts are routed
4. Rollback procedure is tested and takes <5 minutes
