# Hospital ICU Bed Demand Forecasting — Algorithm Writeup

## 1. Problem Statement

Predict daily ICU bed occupancy for each U.S. state for the next 7 days. Accurate forecasts enable hospital administrators to adjust staffing, cancel elective procedures, or open overflow units proactively.

**Business impact:** Overstaffing by 10 beds → ~$50K/month wasted. Understaffing by 10 beds → patient safety risk, regulatory exposure.

## 2. Data Sources

- **Primary:** HHS COVID-19 Reported Patient Impact and Hospital Capacity (Socrata) — dataset `g62h-syeh`
  - URL: https://healthdata.gov/resource/g62h-syeh.csv
  - Rows: ~81,713 | States: 54 | Date range: 2020-01-01 → 2024-04-27
  - Known missingness: ~9% ICU fields, ~14% influenza fields
- **Supplementary:** CDC ILINet (influenza-like illness), Weather API, Calendar data

## 3. ML Formulation

- **Type:** Time-series regression (temporal splits only — no random split)
- **Target:** `staffed_adult_icu_bed_occupancy` (ICU beds occupied)
- **Horizon:** 7 days ahead (t+1 through t+7)
- **Strategy:** Direct multi-output (one model per horizon, not recursive)
- **Grain:** `(state, date)`

## 4. Feature Engineering

### Lag Features
- ICU occupancy: `occ_lag_1`, `occ_lag_7`, `occ_lag_14`
- ICU utilization: `util_lag_1`, `util_lag_7`
- Inpatient beds used: `beds_lag_1`, `beds_lag_7`
- Admissions: lagged COVID confirmed/suspected, influenza

### Rolling Window Features (7/14/30 day windows)
- Rolling mean and standard deviation for each base column
- Trend proxy: `rolling_mean_7 - rolling_mean_14` (acceleration signal)

### Calendar Features
- `day_of_week` (0=Monday), `month` (1-12), `is_weekend` (binary)

### Operational Stress Signals
- `critical_staffing_shortage_today_yes/no` → binary numeric
- `critical_staffing_shortage_anticipated_within_week_yes/no` → binary numeric

### Missingness Handling
- Missingness indicators: `is_missing_<feature>` (binary)
- Imputation: per-state median (fit on training only)

### Training-Serving Parity
All preprocessing bundled in single sklearn.Pipeline object. Identical computation at train and inference time. This is the single most important design decision — training-serving skew is the #1 production ML bug.

## 5. Model Selection

| Model | Role | Why |
|-------|------|-----|
| Naive persistence | Floor | "Yesterday's value" — simplest possible forecast |
| Ridge Regression | Interpretable baseline | Linear, fast, stable coefficients |
| XGBoost Regressor | Production model | Strong tabular performance, handles non-linearity |

### Why Direct Multi-Output?
Recursive forecasting (predict t+1, feed as input for t+2) compounds errors. For 7-day horizons, small errors accumulate rapidly. Direct multi-output trains independent models per horizon — no error compounding.

### Temporal Splitting
- **Train:** Earliest 70% of data (per state, by date)
- **Validation:** Next 15% (model selection, hyperparameter tuning)
- **Test:** Last 15% (held-out, never touched during training)

No random split. Future never leaks into past. Random splits gave misleading metrics in development — temporal splits reflect real deployment conditions.

## 6. Evaluation Metrics

### Primary: Asymmetric RMSE (Underprediction ×3)

```
Loss = sqrt(mean((y_pred - y_true)² × w))
where w = 3 if y_pred < y_true (underprediction)
      w = 1 otherwise
```

Underprediction is 3× worse than overprediction because:
- Underprediction → not enough staff → patient safety risk
- Overprediction → extra staff on standby → wasted cost (acceptable)

### Guardrail Metrics
- **MAE:** Mean absolute error (symmetric, interpretable)
- **Underprediction Rate:** % of predictions where y_pred < y_true
- **Per-horizon breakdown:** Metrics reported for each day t+1 through t+7

### Results

| Model | Asymmetric RMSE (↓) | MAE (↓) | Underprediction Rate (↓) |
|-------|---------------------|---------|--------------------------|
| Naive (yesterday's value) | 1.000 (baseline) | 0.850 | 48% |
| Ridge Regression | 0.720 | 0.580 | 35% |
| **XGBoost (production)** | **0.540** | **0.410** | **22%** |

*Metrics on held-out test set (last 30 days), aggregated across 7-day horizon.*

## 7. Training Pipeline (ZenML)

```
Data Pull → Validate → Feature Engineer → Train → Evaluate → Register
```

1. **Pull:** Deterministic Socrata API query → CSV + pull manifest
2. **Validate:** Schema check, missingness thresholds, range checks, freshness SLA
3. **Feature Engineer:** Lags, rolling stats, calendar, ops signals, imputation
4. **Train:** Ridge + XGBoost with MLflow logging (params, metrics, artifacts)
5. **Evaluate:** Asymmetric RMSE, MAE, underprediction rate on held-out test
6. **Register:** Best model → MLflow Staging (with full metadata: code SHA, data manifest, config)

## 8. Deployment

- **Type:** Batch inference (daily at 06:00 local)
- **Output:** `data/outputs/icu_occupancy_forecast.parquet`
  - Columns: `state`, `forecast_date`, `target_date`, `horizon`, `y_pred`, `model_version`
- **Promotion:** Staging → Production if beats naive baseline on asymmetric RMSE + underprediction rate < threshold
- **Rollback:** Re-point batch job to previous Production model version (< 5 minutes)

## 9. Monitoring & Drift Detection

### Data Drift (PSI + KS Test)
- **PSI (Population Stability Index):** Per-feature distribution shift vs training baseline
  - PSI < 0.1: no drift | 0.1-0.2: moderate | > 0.2: significant
- **KS Test (Kolmogorov-Smirnov):** Distribution change detection (p < 0.05 = drift)

Combined, they catch both gradual degradation and sudden data pipeline failures.

### Data Health
- Schema validation (required columns present)
- Missingness rates (alert if >50% spike vs baseline)
- Duplicate detection on `(state, date)`
- Freshness SLA (latest date within 3 days)

### Alerting
- Report-only MVP (no paging)
- JSON reports → `data/reports/`
- Streamlit dashboard visualizes drift + health over time

## 10. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Asymmetric loss | Underprediction is 3× worse for staffing safety |
| Direct multi-output | Avoids recursive error compounding |
| Temporal splits only | Random splits leak future info |
| PSI + KS drift | Catches both gradual and abrupt distribution shifts |
| Report-only monitoring | Avoids alert fatigue; manual retrain decision |
| Rollback < 5 min | Production safety requirement |
| Training-serving parity | Bundled preprocessing prevents #1 production bug |

## 11. What I Learned

- **Silent failures are the real risk.** A model making confident wrong predictions is worse than one that refuses to predict. Drift detection prevents this.
- **Training-serving skew is the #1 production bug.** Bundling preprocessing inside the sklearn pipeline (not as a separate step) ensures identical computation at train and serve time.
- **Asymmetric metrics change model behavior.** The same XGBoost with asymmetric loss produces systematically different predictions — safer for staffing.
- **Temporal splits are non-negotiable.** Random splits gave AUC 0.95; temporal splits gave 0.78. The honest number is the one you deploy with.

## Links

- **GitHub:** https://github.com/Harisathwik/icuflow
- **Live Dashboard:** https://icuflow.streamlit.app/
- **Data Source:** https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Cap/g62h-syeh
