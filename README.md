# Hospital Bed Demand Forecasting

[![CI](https://github.com/Harisathwik/hospital-bed-forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/Harisathwik/hospital-bed-forecasting/actions/workflows/ci.yml)

> Predict ICU bed demand for the next 7 days so hospital administrators can proactively adjust staffing, cancel elective surgeries, or open overflow units.

## Problem Overview

**Business Problem:** Hospitals need to know how many ICU beds they'll need in the next 7 days. Too few beds = patient safety risk. Too many = wasted resources.

**ML Formulation:** Time-series regression — predict daily ICU bed count using historical admissions, disease surveillance, weather, and calendar features.

**Primary Metric:** Asymmetric RMSE (underprediction ×3 penalty) — because understaffing is far worse than overstaffing.
**Guardrail Metrics:** MAE, Underprediction Rate, Prediction Interval Coverage.

## Project Structure

```
hospital-bed-forecasting/
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
│   ├── monitoring/        # Drift detection
│   └── pipelines/         # ZenML pipeline definitions
├── tests/
├── configs/
│   ├── config.yaml
│     └── config.prod.yaml
├── data/
│   ├── raw/               # Raw ingested data
│   ├── processed/         # Cleaned + feature-engineered
│   └── external/          # Weather, disease surveillance
├── notebooks/             # EDA notebooks
├── docs/
│   ├── problem_statement.md
│   ├── architecture.md
│   └── algorithm_writeup.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Data Sources

| Source | Data | Update Frequency |
|--------|------|-----------------|
| Hospital Records | Admissions, discharges, ICU transfers | Daily |
| CDC ILINet | Influenza-like illness rates | Weekly |
| Weather API | Temperature, humidity, precipitation | Daily |
| Calendar | Day of week, holidays, flu season flag | Static |

## Production Features

- **Automated Pipelines:** ZenML orchestration for training and inference.
- **Model Control Plane:** MLflow for experiment tracking, model versioning, and staging $\to$ production promotion.
- **Monitoring & Drift:** Integrated PSI and KS-test based drift detection with data health checks (schema, missingness, freshness) and JSON reporting.
- **Serving:** High-performance FastAPI endpoint for real-time predictions.
- **Safe Deployment:** Rollback capability to previous model versions in < 5 minutes.

## License

MIT

## Author

**Harisathwik Veerla** — AI Engineer specializing in production ML systems, MLOps, and agentic AI.

- LinkedIn: https://www.linkedin.com/in/harisathwik-veerla/
- GitHub: https://github.com/Harisathwik
- Portfolio: https://harisathwik.github.io/
