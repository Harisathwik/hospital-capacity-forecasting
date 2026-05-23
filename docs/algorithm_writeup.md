# Hospital ICU Bed Demand Forecasting Algorithm

## 1. Problem Statement
Predict the daily occupancy of staffed adult ICU beds for each U.S. state for the next 7 days. Accurate forecasts enable hospital administrators to adjust staffing, cancel elective procedures, or open overflow units proactively.

## 2. Data Sources
- **Primary**: CDC COVID‑19 Reported Patient Impact and Hospital Capacity (Socrata) – `g62h-syeh`.  
  *URL*: https://healthdata.gov/resource/g62h-syeh.csv
- **Supplementary**: CDC ILINet (influenza‑like illness), Weather API, Calendar data.

## 3. Pipeline Overview
1. **Ingestion** – Pull raw CSV from Socrata using deterministic query parameters (`$select`, `$where`, `$order`, `$limit`).  
2. **Validation** – Schema, missingness, range and freshness checks.  
3. **Feature Engine** – Lag/rolling windows, calendar, operational stress signals, missingness indicators.  
4. **Model Training** – Two baselines: Ridge regression & XGBoost regressor.  
5. **Evaluation** – Asymmetric RMSE, MAE, under‑prediction rate.  
6. **Registry & Promotion** – MLflow artifact, promotion rule (beat baseline and guardrails).  
7. **Batch Inference** – Daily job outputs CSV/Parquet to `data/outputs/icu_occupancy_forecast.parquet`.

## 4. Feature Engineering Details
| Feature | Description |
|---|---|
| `occ_lag_1`, `occ_lag_7`, `occ_lag_14` | Lagged ICU occupancy |
| `util_lag_1`, `util_lag_7` | Lagged ICU utilization |
| `admissions_lag_1`, `admissions_lag_7` | Lagged admissions (COVID confirmed/suspected, influenza) |
| Rolling windows (7/14/30) | Mean & std of past values |
| `rolling_mean_7 - rolling_mean_14` | Trend/acceleration |
| Calendar features | Day of week, month, weekend flag |
| Operational stress | Staffing shortage flags |
| Missingness indicators | `is_missing_<feature>` |

## 5. Model Training
* **Ridge** – baseline, interpretable.  
* **XGBoost** – strong baseline for tabular data.  

**Split**: Temporal, per‑state, holdout = most recent window.  

## 6. Evaluation Metrics
* **Primary**: Asymmetric RMSE (under‑prediction ×3).  
* **Guardrails**: MAE, under‑prediction rate.  

## 7. Deployment Plan
* Batch inference at 06:00 local each day.  
* Output contract: `state`, `forecast_date`, `target_date`, `horizon`, `y_pred`, `model_version`.  
* Promotion rule: Best run that beats naive persistence baseline on RMSE and meets guardrails.  
* Rollback: Switch to previous production model within 5 minutes.

---
*For more details see the architecture diagram and pipeline definitions in the `docs/` folder.*