# Self-Healing Churn Prediction Pipeline

## Overview
A production-grade MLOps system that automatically detects data drift, triggers retraining, validates new models against business guardrails, and promotes them to production — all without human intervention.

## Architecture

```
[Kaggle Telco Churn Data]
        ↓
[ZenML Training Pipeline]
  ├── load_data       → Ingest & validate
  ├── preprocess      → Feature engineering
  ├── train_model     → XGBoost with hyperparameter tuning
  └── evaluate_model  → F2-score, PR-AUC, FPR guardrail
        ↓
[MLflow Model Registry]
  ├── Experiment tracking (params, metrics, artifacts)
  └── Model versioning + stage transitions
        ↓
[FastAPI Serving Layer]
  ├── /predict        → Real-time predictions
  ├── /health         → Health check
  └── /model-info     → Current production model metadata
        ↓
[Evidently Drift Monitor]
  ├── Compares live traffic vs training distribution
  ├── Triggers retraining when drift detected
  └── Promotion gate: new model must beat production
        ↓
[Streamlit Dashboard]
  ├── Live metrics (F2, PR-AUC, FPR)
  ├── Drift score visualization
  └── Model version history
```

## Dataset
- **Source:** [Kaggle Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 21 (demographics, services, account info)
- **Target:** `Churn` (Yes/No) — ~27% churn rate

## Tech Stack (all free)
| Component | Tool |
|-----------|------|
| Orchestration | ZenML |
| Experiment Tracking | MLflow |
| Model Registry | MLflow |
| Drift Detection | Evidently |
| Serving | FastAPI |
| Dashboard | Streamlit |
| CI/CD | GitHub Actions |
| Containerization | Docker |

## Setup

```bash
# 1. Clone and enter directory
git clone https://github.com/harisathwik-veerla/MLOps-Github.git
cd MLOps-Github

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize ZenML
zenml init
zenml integration install sklearn xgboost mlflow evidently -y
```

## Usage

### Run Training Pipeline
```bash
python -m src.pipelines.train_pipeline
```

### Start Serving API
```bash
uvicorn src.serving.app:app --reload --port 8000
```

### Open Dashboard
```bash
streamlit run dashboard/app.py
```

### Run Tests
```bash
pytest tests/ -v
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Get churn prediction for a customer |
| `/health` | GET | Health check |
| `/model-info` | GET | Current production model metadata |

### Example Prediction Request
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

## Project Structure
```
MLOps-Github/
├── .github/workflows/ci-cd.yml
├── configs/
│   └── config.yaml
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Data ingestion
│   │   └── validator.py        # Schema & quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py         # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Model training
│   │   └── evaluator.py        # Model evaluation
│   ├── serving/
│   │   ├── __init__.py
│   │   └── app.py              # FastAPI application
│   ├── monitoring/
│   │   ├── __init__.py
│   │   └── drift_detector.py   # Evidently drift detection
│   └── pipelines/
│       ├── __init__.py
│       └── train_pipeline.py   # ZenML pipeline definition
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_features.py
│   ├── test_models.py
│   └── test_serving.py
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml
└── README.md
```

## CI/CD Pipeline
On every push to `main`:
1. Run linting (flake8)
2. Run unit tests (pytest)
3. Build Docker image
4. Push to GitHub Container Registry

## License
MIT
