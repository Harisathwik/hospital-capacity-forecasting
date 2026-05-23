# Hospital Capacity Forecasting

[![CI](https://github.com/Harisathwik/hospital-capacity-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Harisathwik/hospital-capacity-forecasting/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/🚀_Live_Dashboard-Streamlit-FF4B4B?logo=streamlit)](https://hospital-capacity-forecasting.streamlit.app/)

> Predict ICU bed demand for the next 7 days so hospital administrators can proactively adjust staffing, cancel elective surgeries, or open overflow units.

## 🚀 Live Demo

**Dashboard:** [hospital-capacity-forecasting.streamlit.app](https://hospital-capacity-forecasting.streamlit.app/)

The dashboard shows real-time drift detection, data health monitoring, and feature distribution analysis — all updated from the latest pipeline run.

## 💡 Why This Matters

Hospitals operate on thin margins. Overstaffing burns cash. Understaffing kills.

- **Overstaffing by 10 beds** → ~$50K/month wasted per hospital
- **Understaffing by 10 beds** → patient safety risk, staff burnout, regulatory exposure

This system forecasts ICU demand 7 days ahead with an **asymmetric loss function** that penalizes underprediction 3× more than overprediction. Because getting caught short is worse than having extra capacity.

**Projected impact:** 15-20% reduction in staffing cost variance while maintaining <5% understaffing risk.

## 📊 Model Performance

| Model | Asymmetric RMSE (↓) | MAE (↓) | Underprediction Rate (↓) |
|-------|---------------------|---------|--------------------------|
| Naive (yesterday's value) | 1.000 (baseline) | 0.850 | 48% |
| Ridge Regression | 0.720 | 0.580 | 35% |
| **XGBoost (production)** | **0.540** | **0.410** | **22%** |

*Metrics on held-out test set (last 30 days), aggregated across 7-day horizon. Asymmetric RMSE penalizes underprediction ×3.*

## 🏗️ Architecture

```
Data (HHS healthdata.gov)
  → ZenML Training Pipeline
    → Data Validation (schema, missingness, freshness)
    → Feature Engineering (lags, rolling stats, calendar)
    → Model Training (Ridge / XGBoost, MLflow tracking)
    → Evaluation (asymmetric RMSE, underprediction rate)
    → Model Registry (Staging → Production)
  → FastAPI Serving (:8002)
  → Drift Detection Pipeline (PSI + KS test vs baseline)
  → Monitoring Dashboard (Streamlit)
```

## 🛠️ Production Features

- **Automated Pipelines:** ZenML orchestration for training and inference
- **Model Control Plane:** MLflow for experiment tracking, model versioning, and Staging → Production promotion
- **Drift Detection:** PSI + KS-test based feature drift monitoring vs training baseline
- **Data Health:** Schema validation, missingness checks, duplicate detection, freshness SLA
- **Asymmetric Loss:** Underprediction penalized ×3 — aligned with staffing safety requirements
- **Serving:** FastAPI endpoint with `/predict` and `/health` routes
- **Rollback:** Switch to previous model version in < 5 minutes
- **CI/CD:** GitHub Actions for automated testing, linting, and Docker builds

## 📁 Project Structure

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
├── tests/                 # Unit tests for monitoring, data, features
├── configs/               # Environment-specific configuration
├── docs/                  # Problem statement, architecture, algorithm writeup
├── Dockerfile
├── docker-compose.yml
├── requirements.txt       # Lightweight deps for Streamlit Cloud
├── pyproject.toml         # Full project deps for local dev
└── README.md
```

## 📡 API Usage

```bash
# Health check
curl http://localhost:8002/health

# Prediction
curl -X POST http://localhost:8002/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"inpatient_beds_used_lag_1": 150, "inpatient_beds_used_lag_7": 145, "inpatient_beds_used_roll_mean_7": 148, "day_of_week": 2, "month": 6}}'
```

Response:
```json
{
  "predictions": [152.3, 149.8, 147.1, 144.5, 142.0, 140.2, 138.9],
  "target_names": ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6", "t+7"]
}
```

## 📈 Monitoring

The drift detection pipeline runs automatically and writes reports to `data/reports/`. The Streamlit dashboard visualizes:

- **PSI per feature** — Population Stability Index with thresholds (0.1 moderate, 0.2 significant)
- **KS test results** — Distribution shift detection
- **Data health** — Schema, completeness, duplicates, freshness
- **Alert log** — Severity-coded alerts (critical/warning/info)

## 🔧 Local Setup

```bash
git clone https://github.com/Harisathwik/hospital-capacity-forecasting.git
cd hospital-capacity-forecasting
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
PYTHONPATH=. pytest tests/ -v
```

Run dashboard locally:
```bash
PYTHONPATH=. streamlit run src/dashboard/app.py
```

## 📄 Data Source

[HHS COVID-19 Reported Patient Impact and Hospital Capacity](https://healthdata.gov/Hospital/COVID-19-Reported-Patient-Impact-and-Hospital-Cap/g62h-syeh) — public dataset via healthdata.gov Socrata API.

## 📜 License

MIT

## 👤 Author

**Harisathwik Veerla** — AI Engineer specializing in production ML systems, MLOps, and agentic AI.

- LinkedIn: https://www.linkedin.com/in/harisathwik-veerla/
- GitHub: https://github.com/Harisathwik
- Portfolio: https://harisathwik.github.io/
