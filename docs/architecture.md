# Architecture: Hospital Capacity Forecasting (ICU Occupancy) for Staffing

## MLOps Pipeline Overview

A production ML system is not a model  it is a system of pipelines.
We design for the full lifecycle:

1) **Data Ingestion**  2) **Data Validation**  3) **Feature Engineering**  4) **Model Training**  5) **Model Evaluation**  6) **Model Registry**  7) **Deployment**  8) **Monitoring**  9) **Drift Detection**  10) **Retraining Trigger**

Pipeline decomposition (MVP scope):
- **Training pipeline**: pull data  validate  build features  train  evaluate  register
- **Inference pipeline (batch)**: load production model  compute features  forecast next 7 days  write outputs
- **Drift detection pipeline**: compare live feature distributions vs training baseline  alert
- **Monitoring pipeline**: track data health + prediction health + business health
- **Retraining pipeline**: triggered by drift/perf drop; gated promotion to Production

Maturity target:
- Level 1 (pipeline automation) quickly, then Level 2 (CI/CD for ML). Level 3 (auto retraining) only if evidence demands it.

---

## Data Plan (LOCKED)

### Data source
We start with a fully public, API-accessible dataset so the project is reproducible end-to-end.

- **Dataset**: HHS / healthdata.gov  *COVID-19 Reported Patient Impact and Hospital Capacity* (Socrata)
- **Dataset page**: https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Cap/g62h-syeh
- **Dataset id**: `g62h-syeh`
- **API base**:
  - JSON: https://healthdata.gov/resource/g62h-syeh.json
  - CSV:  https://healthdata.gov/resource/g62h-syeh.csv

The locally downloaded snapshot (not committed to git):
- `data/raw/hhs_hospital_capacity_g62h-syeh.csv`

### Ingestion strategy (deterministic pull)
We will **avoid manual browser downloads**. Socrata supports deterministic data pulls via query parameters:
- `$select`  curated column subset
- `$where`  date filter
- `$order`  stable ordering (determinism)
- `$limit`  cap rows

We will pull by a fixed query shape (example pattern):

```
https://healthdata.gov/resource/g62h-syeh.csv?
  $select=<curated_columns>&
  $where=date>='2020-01-01T00:00:00.000'&
  $order=date%20ASC&
  $limit=5000000
```

### Data versioning (reproducibility without bloating git)
Raw CSVs can be large and change over time. We will **not** commit full raw snapshots to git by default.

Instead, every pull produces a small **pull manifest** (metadata) capturing:
- pull timestamp (UTC)
- dataset_id + source_url
- exact query (select/where/order/limit)
- row_count, date_min/date_max
- schema signature/hash (columns list + basic type inference)

Storage options:
- MVP: store manifests as ZenML artifacts (local artifact store)
- Optional: also commit manifests as JSON under `data/pulls/` (still no raw CSV in git)

### Data validation (quality gates)
We implement validation at the start of pipelines so bad data fails fast.

Hard checks (fail run):
- Required columns present (including target `staffed_adult_icu_bed_occupancy`)
- `date` parseable for all rows
- Numeric columns parseable (no non-numeric strings)
- Duplicate detection on `(state, date)` (warn or fail if severe)

Soft checks (warn/alert; may fail if severe):
- Missingness thresholds (baseline from audit):
  - ICU fields ~9% missing in our snapshot  alert if it jumps (e.g., >12%)
  - influenza fields ~14% missing  alert if it jumps (e.g., >20%)
- Range checks:
  - utilization roughly within [0, 1.5] (flag extremes)
  - occupancy should be  total staffed ICU beds + small tolerance (flag violations)
- Freshness checks:
  - record the latest available date; in production this becomes an SLA (e.g., latest date within N days)

### Data storage
- Raw snapshot: `data/raw/` (ignored by git)
- Processed features: `data/processed/` as Parquet (ignored by git)

Splitting strategy:
- **Temporal splits only** (no random split) to avoid leakage.
- Group-aware behavior: we model per state and avoid mixing future into past.

---

## Feature Engineering Plan (TODO  next)
LOCKED decisions:
- Forecasting strategy: **Direct multi-output** (predict t+1..t+7 in one shot; avoids recursive error compounding)
- Target: `staffed_adult_icu_bed_occupancy`
- Grain: `(state, date)`

### Target / horizon construction
- For each `(state, date=t)` row, create labels:
  - `y_t_plus_1` ... `y_t_plus_7` from future values of `staffed_adult_icu_bed_occupancy`
- Drop the last 7 days per state (no future labels).

### Feature set (past-only; no leakage)

Indexing:
- Sort by date per state.
- Detect duplicates on `(state, date)`; warn or fail depending on severity.

Lag features (examples):
- ICU occupancy: `occ_lag_1`, `occ_lag_7`, `occ_lag_14`
- Utilization: `adult_icu_bed_utilization_lag_1`, `_lag_7`
- Inpatient beds used: `inpatient_beds_used_lag_1`, `_lag_7`
- Admissions: lagged versions of prior-day admissions columns (COVID confirmed/suspected, influenza)

Rolling window features (computed using only past data):
- Rolling mean/std for windows 7/14/30 days for:
  - ICU occupancy
  - ICU utilization
  - inpatient beds used
  - admissions (confirmed/suspected; influenza where available)
- Trend proxy: `rolling_mean_7 - rolling_mean_14` (acceleration signal)

Calendar features (safe):
- day_of_week (06), month (112), weekend flag
- (Optional later) US federal holiday flags

Operational stress signals:
- Use staffing shortage indicators as numeric signals:
  - `critical_staffing_shortage_today_yes/no`
  - `critical_staffing_shortage_anticipated_within_week_yes/no`

### Missingness handling
Because ICU and influenza fields have non-trivial missingness, we will handle missingness explicitly:
- Add missingness indicators: `is_missing_<feature>`
- Impute numeric features with per-state median (fit on training only)

### Trainingserving parity (non-negotiable)
All preprocessing (imputation + missing indicators + optional scaling) must live inside a single serialized pipeline object (e.g., sklearn Pipeline) so training and inference compute identical features.

## Training & Evaluation Plan (TODO)
LOCKED approach: **two baselines**
- **Ridge regression** = "floor" model (fast, interpretable, sanity-check)
- **XGBoost regressor** = "strong baseline" (nonlinear; typically stronger on tabular)

### Split strategy (no leakage)
- Temporal splits only (no random split).
- Per-state ordering by date.
- Holdout is the most recent window (e.g., last `data.test_size_days` days) to approximate future performance.

### Baselines
- **Naive persistence baseline**: use yesterday's ICU occupancy as the forecast for all horizons (t+1..t+7).
- Compare Ridge and XGBoost against this baseline using the same temporal split.

### Metrics
- **Primary metric**: asymmetric RMSE (underprediction ×3 penalty) aggregated across horizons, plus per-horizon breakdown.
- **Guardrails**: MAE and underprediction rate (% of days where prediction < actual).

### Experiment tracking & reproducibility
- Use MLflow (via ZenML stack) to log:
  - parameters (model hyperparams, horizon, feature windows)
  - data pull manifest (query + date range + row count)
  - metrics per horizon + aggregated
  - model artifacts (full preprocessing+model pipeline)

### Temporal validation (optional after MVP)
- Rolling-origin evaluation (walk-forward) for more robust estimates.

## Deployment Plan (TODO)
LOCKED: **Batch inference** (daily)  no real-time API in MVP.

### How predictions are consumed
- Predictions are generated once per day and written to a durable artifact (CSV/Parquet).
- Downstream users (staffing ops) consume a simple table with forecast + uncertainty.

### Schedule
- Batch job runs daily (e.g., 06:00 local) to forecast horizons t+1..t+7.

### Input contract
- Uses the latest available raw snapshot (or incrementally pulled data) up to date=t.
- Applies the same feature pipeline as training (trainingserving parity).

### Output contract (recommended)
Write `data/outputs/icu_occupancy_forecast.parquet` (or CSV) with columns:
- state
- forecast_date (the date we ran the batch)
- target_date (the day being forecasted)
- horizon (1..7)
- y_pred (point forecast)
- y_pred_p50/p90 (optional later)
- model_version (from registry)

### Model promotion + rollback
- Models are registered with metrics attached.
- Promotion rule (MVP): promote the best run that beats naive baseline on the primary metric and passes underprediction-rate guardrail.
- Rollback: batch job can be pointed back to the previous Production model version; rollback must be <5 minutes.

## Monitoring & Drift Plan (TODO)
LOCKED (MVP): **Report-only monitoring** (no alerts/paging).

Why: avoids noisy alerting complexity while still preventing silent failures. Monitoring outputs become versioned artifacts you can review and show in a portfolio.

### What we monitor
Data health (run every batch):
- schema checks (required columns, types)
- missingness rates (watch for spikes vs baseline)
- duplicates on (state, date)
- freshness (latest date observed)

Drift (run daily/with each batch):
- feature distribution drift vs training baseline (e.g., PSI on key numeric features)
- target drift is only measurable once labels arrive; for MVP we track *prediction distribution* + later backtest error.

Prediction risk guardrails:
- underprediction rate (% days y_pred < y_true) on backtests
- asymmetric RMSE trend over time (when labels are available)

### Outputs (artifacts)
- Write drift + health reports to `data/reports/` (HTML/JSON/Parquet summaries).
- Keep a simple historical log so we can plot drift/health over time.

### Retraining trigger (deferred)
- MVP does not auto-retrain. We generate reports; retraining is a manual decision.
- We add auto-retraining only after we see consistent drift or sustained metric degradation.

## Versioning & Governance (TODO)
LOCKED: **Two-stage promotion flow** (Staging > Production)

### Model registry stages
- **Staging**: candidate models that passed training/eval and are ready for review
- **Production**: the currently-serving batch model version

### What gets versioned (the 4 sources of truth)
- **Code**: git commit SHA
- **Data**: pull manifest (dataset_id, query, date range, row count)
- **Model**: registered model version + artifacts (full preprocessing+model pipeline)
- **Config**: configs/config.yaml (and any per-env overrides)

### Promotion gates (must pass to move Staging > Production)
- Beats naive baseline on the **primary metric** (asymmetric RMSE under 13)
- Passes guardrail: underprediction rate below an agreed threshold (set later)
- Data validation passed (schema/type/range/missingness)
- Training run is reproducible (same inputs re-run yields similar metrics within tolerance)

### Rollback
- Production points to a single model version.
- Rollback = re-point batch inference to the previous Production model version.
- Rollback target time: <5 minutes.

## ZenML Stack Specification (TODO)
TBD in Phase 2H after requirements are set.

## Project Structure
See repo structure under `src/`, `configs/`, `docs/`, `tests/`.

## MVP Scope
MVP = reproducible training pipeline + batch inference + drift report + rollbackable model registry.

## Deferred Components
- External data sources (weather/ILI) until baseline pipeline is stable
- Auto retraining until drift evidence justifies it
