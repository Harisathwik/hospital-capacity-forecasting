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
TBD after we confirm: horizon definition (direct vs recursive forecasting), state-level modeling approach, lag/rolling features, and holiday features.

## Training & Evaluation Plan (TODO)
TBD: baseline model, candidate models, temporal CV strategy, experiment tracking in MLflow.

## Deployment Plan (TODO)
TBD: batch inference schedule, output format, model promotion workflow.

## Monitoring & Drift Plan (TODO)
TBD: what to monitor (data drift, prediction drift, underprediction rate), alert routes, retraining trigger.

## Versioning & Governance (TODO)
TBD: registry stages (staging/production), promotion gates, rollback strategy.

## ZenML Stack Specification (TODO)
TBD in Phase 2H after requirements are set.

## Project Structure
See repo structure under `src/`, `configs/`, `docs/`, `tests/`.

## MVP Scope
MVP = reproducible training pipeline + batch inference + drift report + rollbackable model registry.

## Deferred Components
- External data sources (weather/ILI) until baseline pipeline is stable
- Auto retraining until drift evidence justifies it
