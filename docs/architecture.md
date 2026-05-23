# Architecture: Hospital Capacity Forecasting (ICU Occupancy) for Staffing

## MLOps Pipeline Overview

A production ML system is not a model — it is a system of pipelines. We design for the full lifecycle:

1. **Data Ingestion** → 2. **Data Validation** → 3. **Feature Engineering** → 4. **Model Training** → 5. **Model Evaluation** → 6. **Model Registry** → 7. **Deployment** → 8. **Monitoring** → 9. **Drift Detection** → 10. **Retraining Trigger**

### Pipeline Decomposition

- **Training pipeline:** pull data → validate → build features → train → evaluate → register
- **Inference pipeline (batch):** load production model → compute features → forecast next 7 days → write outputs
- **Drift detection pipeline:** compare live feature distributions vs training baseline → alert
- **Monitoring pipeline:** track data health + prediction health + business health
- **Retraining pipeline:** triggered by drift/perf drop; gated promotion to Production

### Maturity Target

- **Level 1 (pipeline automation):** reproducible pipelines, model versioning, data validation — target this quickly
- **Level 2 (CI/CD for ML):** automated testing, promotion gates, multiple environments — reach within months
- **Level 3 (full automation):** automated drift response, retraining, monitoring — only when scale demands it

---

## Data Plan

### Data Source

- **Dataset:** HHS / healthdata.gov — COVID-19 Reported Patient Impact and Hospital Capacity (Socrata)
- **Dataset page:** https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Cap/g62h-syeh
- **Dataset id:** `g62h-syeh`
- **API base:**
  - JSON: https://healthdata.gov/resource/g62h-syeh.json
  - CSV: https://healthdata.gov/resource/g62h-syeh.csv

### Ingestion Strategy (Deterministic Pull)

Socrata supports deterministic data pulls via query parameters:
- `$select` — curated column subset
- `$where` — date filter
- `$order` — stable ordering (determinism)
- `$limit` — cap rows

Example query pattern:
```
https://healthdata.gov/resource/g62h-syeh.csv?
  $select=<curated_columns>&
  $where=date>='2020-01-01T00:00:00.000'&
  $order=date%20ASC&
  $limit=5000000
```

### Data Versioning

Every pull produces a small **pull manifest** (metadata) capturing:
- pull timestamp (UTC)
- dataset_id + source_url
- exact query (select/where/order/limit)
- row_count, date_min/date_max
- schema signature/hash

Storage: ZenML artifacts (local artifact store for MVP).

### Data Validation (Quality Gates)

**Hard checks (fail run):**
- Required columns present (including target)
- `date` parseable for all rows
- Numeric columns parseable
- Duplicate detection on `(state, date)`

**Soft checks (warn/alert):**
- Missingness thresholds (ICU fields ~9% baseline, alert if >12%)
- Range checks (utilization within [0, 1.5])
- Freshness (latest date within N days)

### Data Storage

- Raw snapshot: `data/raw/` (ignored by git)
- Processed features: `data/processed/` as Parquet (ignored by git)
- Splitting strategy: **Temporal splits only** (no random split) to avoid leakage

---

## Feature Engineering Plan

### Target / Horizon Construction

- **Strategy:** Direct multi-output (predict t+1..t+7 in one shot; avoids recursive error compounding)
- **Target:** `staffed_adult_icu_bed_occupancy`
- **Grain:** `(state, date)`
- For each `(state, date=t)` row, create labels: `y_t_plus_1` ... `y_t_plus_7`
- Drop the last 7 days per state (no future labels)

### Feature Set (Past-Only; No Leakage)

**Lag features:**
- ICU occupancy: `occ_lag_1`, `occ_lag_7`, `occ_lag_14`
- Utilization: `adult_icu_bed_utilization_lag_1`, `_lag_7`
- Inpatient beds used: `inpatient_beds_used_lag_1`, `_lag_7`
- Admissions: lagged versions of prior-day admissions columns

**Rolling window features (computed using only past data):**
- Rolling mean/std for windows 7/14/30 days
- Trend proxy: `rolling_mean_7 - rolling_mean_14` (acceleration signal)

**Calendar features (safe):**
- `day_of_week` (0-6), `month` (1-12), `is_weekend` flag

**Operational stress signals:**
- `critical_staffing_shortage_today_yes/no`
- `critical_staffing_shortage_anticipated_within_week_yes/no`

**Missingness handling:**
- Missingness indicators: `is_missing_<feature>`
- Impute numeric features with per-state median (fit on training only)

### Training-Serving Parity (Non-Negotiable)

All preprocessing (imputation + missing indicators + optional scaling) lives inside a single serialized sklearn.Pipeline object. Training and inference compute identical features. This prevents the #1 production ML bug: training-serving skew.

---

## Training & Evaluation Plan

### Models

| Model | Role | Why |
|-------|------|-----|
| Naive persistence | Floor | "Yesterday's value" — simplest benchmark |
| Ridge Regression | Interpretable baseline | Linear, fast, stable coefficients |
| XGBoost Regressor | Production model | Strong tabular performance |

### Split Strategy (No Leakage)

- Temporal splits only (no random split)
- Per-state ordering by date
- Holdout = most recent window (e.g., last 30 days)

### Metrics

- **Primary:** Asymmetric RMSE (underprediction ×3) aggregated across horizons, plus per-horizon breakdown
- **Guardrails:** MAE, underprediction rate (% of days where prediction < actual)

### Experiment Tracking & Reproducibility

Use MLflow (via ZenML stack) to log:
- Parameters (model hyperparams, horizon, feature windows)
- Data pull manifest (query + date range + row count)
- Metrics per horizon + aggregated
- Model artifacts (full preprocessing+model pipeline)

---

## Deployment Plan

### How Predictions Are Consumed

- Predictions generated once per day, written to durable artifact (Parquet)
- Downstream users (staffing ops) consume a simple table with forecast + uncertainty

### Schedule

- Batch job runs daily (e.g., 06:00 local) to forecast horizons t+1..t+7

### Input Contract

- Uses latest available raw snapshot up to date=t
- Applies same feature pipeline as training (training-serving parity)

### Output Contract

`data/outputs/icu_occupancy_forecast.parquet` with columns:
- `state`, `forecast_date`, `target_date`, `horizon` (1..7), `y_pred`, `model_version`

### Model Promotion + Rollback

- Models registered with metrics attached
- Promotion rule: best run that beats naive baseline on primary metric and passes underprediction-rate guardrail
- Rollback: batch job pointed back to previous Production model version; < 5 minutes

---

## Monitoring & Drift Plan

**MVP: Report-only monitoring** (no alerts/paging). Avoids noisy alerting while preventing silent failures.

### What We Monitor

**Data health (every batch):**
- Schema checks (required columns, types)
- Missingness rates (watch for spikes vs baseline)
- Duplicates on `(state, date)`
- Freshness (latest date observed)

**Drift (daily/with each batch):**
- Feature distribution drift vs training baseline (PSI on key numeric features)
- Target drift measurable once labels arrive; for MVP we track prediction distribution + later backtest error

**Prediction risk guardrails:**
- Underprediction rate on backtests
- Asymmetric RMSE trend over time (when labels available)

### Outputs (Artifacts)

- Drift + health reports written to `data/reports/` (HTML/JSON/Parquet summaries)
- Historical log maintained for plotting drift/health over time

### Retraining Trigger (Deferred)

- MVP does not auto-retrain; reports generated, retraining is manual decision
- Auto-retraining added only after consistent drift or sustained metric degradation

---

## Versioning & Governance

### Two-Stage Promotion Flow: Staging → Production

**Model registry stages:**
- **Staging:** candidate models that passed training/eval, ready for review
- **Production:** currently-serving batch model version

### What Gets Versioned (4 Sources of Truth)

- **Code:** git commit SHA
- **Data:** pull manifest (dataset_id, query, date range, row count)
- **Model:** registered model version + artifacts (full preprocessing+model pipeline)
- **Config:** `configs/config.yaml`

### Promotion Gates (Must Pass to Move Staging → Production)

1. Beats naive baseline on primary metric (asymmetric RMSE)
2. Passes guardrail: underprediction rate below agreed threshold
3. Data validation passed (schema/type/range/missingness)
4. Training run is reproducible

### Rollback

- Production points to a single model version
- Rollback = re-point batch inference to previous Production model version
- Target: < 5 minutes

---

## ZenML Stack Specification

**Local-only ZenML stack (MVP):**

| Component | Choice | Why |
|-----------|--------|-----|
| Orchestrator | `default` (local) | Simplest execution on your machine |
| Artifact Store | Local artifact store | Stores pipeline artifacts locally |
| Experiment Tracker | MLflow (local) | Experiment history + metrics per run |
| Model Registry | MLflow model registry | Supports Staging → Production promotion |
| Data Validator | Evidently | Drift/health report artifacts |

---

## Project Structure

```
hospital-capacity-forecasting/
├── src/
│   ├── core/              # Pure Python logic (no framework imports)
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── evaluation.py
│   ├── data/              # Data loading + validation steps
│   ├── features/          # Feature engineering steps
│   ├── models/            # Model training + evaluation steps
│   ├── evaluation/        # Evaluation metrics
│   ├── serving/           # FastAPI inference endpoint
│   ├── monitoring/        # Drift detection + health checks
│   ├── dashboard/         # Streamlit monitoring dashboard
│   └── pipelines/         # ZenML pipeline definitions
├── tests/                 # Unit tests
├── configs/               # Environment-specific configuration
├── docs/                  # Problem statement, architecture, algorithm writeup
├── Dockerfile
├── docker-compose.yml
├── requirements.txt       # Lightweight deps for Streamlit Cloud
├── pyproject.toml         # Full project deps for local dev
└── README.md
```

---

## MVP Scope

### Training & Evaluation
- [x] Deterministic data pull + pull manifest stored
- [x] Data validation gates run before training
- [x] Feature builder creates leakage-safe lags/rollings + 7 horizon labels
- [x] Baselines computed: naive persistence, Ridge, XGBoost
- [x] Metrics logged per-horizon and aggregated (asymmetric RMSE, MAE, underprediction rate)
- [x] Best run registered to Staging with full metadata
- [x] Manual promotion Staging → Production after gates pass

### Batch Inference
- [x] Daily batch inference produces `data/outputs/icu_occupancy_forecast.parquet`
- [x] Output contract fields present
- [x] Rollback tested

### Monitoring (Report-Only)
- [x] Data health report written to `data/reports/` each run
- [x] Drift report (PSI + KS) written to `data/reports/` each run
- [x] Historical log of drift/health metrics maintained
- [x] Streamlit dashboard visualizes all reports

### CI/CD
- [x] GitHub Actions for test + lint + Docker build
- [x] CI badge in README

---

## Deferred Components

- External data sources (weather/ILI) until baseline pipeline is stable
- Alerts/paging (Slack/email) — keep report-only for MVP
- Auto-retraining until drift evidence justifies it
- Real-time API serving (batch is sufficient for staffing)

---

## Key Design Decisions Summary

| Decision | Rationale |
|----------|-----------|
| Asymmetric loss (×3 underprediction) | Understaffing is far worse than overstaffing |
| Direct multi-output forecasting | Avoids recursive error compounding |
| Temporal splits only | Random splits leak future info; temporal reflects reality |
| PSI + KS drift detection | Catches both gradual and abrupt distribution shifts |
| Report-only monitoring (MVP) | Avoids alert fatigue; manual retrain decision |
| Training-serving parity (sklearn.Pipeline) | Prevents #1 production ML bug |
| Rollback < 5 minutes | Production safety requirement |
| Two-stage promotion (Staging → Production) | Prevents broken models from reaching production |
