# System Design: Telco Customer Churn Prediction — Self-Healing MLOps Pipeline

## Requirements

### Functional Requirements

| ID | Requirement | Description |
|----|-------------|-------------|
| F1 | Batch churn scoring | Score all customers daily/weekly, output churn risk list to retention team |
| F2 | Real-time prediction API | Given customer features, return churn probability and risk tier in <100ms |
| F3 | Model training pipeline | Reproducible pipeline: data → features → train → evaluate → register |
| F4 | Drift detection | Monitor incoming feature distributions vs training baseline; flag when drift exceeds threshold |
| F5 | Auto-retrain trigger | When drift detected, automatically retrain, evaluate, and promote if new model beats production |
| F6 | Monitoring dashboard | Live view of model metrics, drift scores, prediction history, feature importance |
| F7 | Model registry | Versioned model storage with stage transitions (Staging → Production → Archived) |

### Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| **Scale** | ~100K customers scored daily (batch); ~10 QPS real-time | Mid-sized telecom; batch is primary, real-time is secondary |
| **Latency (batch)** | <30 min for full customer base | Overnight batch job; retention team reviews results next morning |
| **Latency (real-time)** | <100ms P95 per prediction | Customer service rep waiting on the phone; must be instant |
| **Availability** | 99.5% (real-time API) | Not safety-critical; brief downtime acceptable during retraining |
| **Prediction freshness** | Daily (batch), Real-time (on-demand) | Churn signals change slowly; daily refresh is sufficient |
| **Model update frequency** | Weekly scheduled + drift-triggered | Balance between freshness and stability |
| **Feature freshness** | Daily batch computation | Features derived from account data that updates daily |
| **Interpretability** | Per-customer risk factors | Retention team needs to know WHY a customer is flagged |
| **Cost** | $0 infrastructure (all free/open-source) | Local deployment; Docker for containerization |

---

## Estimation

| Metric | Calculation | Result |
|--------|-------------|--------|
| **Batch QPS** | 100K customers / 30 min = 100,000 / 1,800s | ~56 predictions/sec (batch) |
| **Real-time QPS** | ~10 concurrent users × 1 req/s | ~10 QPS peak |
| **Storage (models)** | ~5 MB/model × 12 versions/year | ~60 MB/year |
| **Storage (predictions)** | 100K rows × 200 bytes × 365 days | ~7.3 GB/year |
| **Storage (training data)** | 7K rows × 19 features × 8 bytes | ~1 MB (negligible) |
| **Training compute** | XGBoost on 7K rows, 19 features | ~5 seconds on CPU |
| **Drift detection** | Evidently on 100K rows, 19 features | ~10 seconds on CPU |
| **Latency budget (real-time)** | Network (10ms) + Preprocessing (5ms) + Prediction (2ms) + Response (5ms) | ~22ms total (well under 100ms) |

**Architecture implications**: Low QPS and small dataset mean we don't need distributed systems. A single FastAPI server + SQLite/PostgreSQL + local file storage is sufficient. No need for Kafka, Redis, or Kubernetes at this scale.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │  CSV Dataset │    │  Prediction  │    │  Training Data       │  │
│  │  (7K rows)   │    │  Logs (SQLite)│    │  Snapshot (versioned)│  │
│  └──────┬───────┘    └──────┬───────┘    └──────────┬───────────┘  │
└─────────┼───────────────────┼───────────────────────┼──────────────┘
          │                   │                       │
          ▼                   ▼                       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ZENML ORCHESTRATION LAYER                        │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   TRAINING PIPELINE                          │    │
│  │  load_data → validate → preprocess → train → evaluate →     │    │
│  │  register_model                                              │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │                 DRIFT DETECTION PIPELINE                     │    │
│  │  load_reference_profile → load_current_data → compute_drift │    │
│  │  → generate_report → evaluate_thresholds → trigger_retrain  │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                              │                                       │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │                 RETRAINING PIPELINE                          │    │
│  │  trigger_training → compare_with_production → promote_if_better│   │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      SERVING LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    FASTAPI REAL-TIME API                     │    │
│  │  POST /predict  → load_model → preprocess → predict →       │    │
│  │                  return {probability, risk_tier, factors}    │    │
│  │  GET  /health   → {status, model_version, uptime}           │    │
│  │  GET  /model-info → {version, metrics, training_date}       │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MONITORING LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  STREAMLIT DASHBOARD                         │    │
│  │  • Model metrics (F2, Recall, Precision, FPR) over time     │    │
│  │  • Drift score visualization (per-feature + overall)         │    │
│  │  • Prediction history & distribution                         │    │
│  │  • Feature importance (SHAP)                                 │    │
│  │  • Model version history                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### Component Descriptions

| Component | Purpose | Technology | Failure Mode |
|-----------|---------|------------|--------------|
| **Data Layer** | Store training data, prediction logs, model artifacts | CSV + SQLite + local files | Data corruption → retrain from snapshot |
| **ZenML Orchestrator** | Pipeline execution, artifact tracking, reproducibility | ZenML (local) | Pipeline step fails → retry or alert |
| **Training Pipeline** | End-to-end model training | ZenML steps + sklearn + XGBoost | Training fails → alert, keep production model |
| **Drift Detection** | Monitor data distribution shifts | Evidently | Drift missed → silent degradation |
| **Retraining Pipeline** | Auto-retrain + promote gate | ZenML + MLflow | Bad model promoted → rollback to previous |
| **FastAPI Serving** | Real-time predictions | FastAPI + uvicorn | API down → health check alerts |
| **MLflow Registry** | Model versioning + stage transitions | MLflow (local) | Registry corruption → rebuild from artifacts |
| **Streamlit Dashboard** | Monitoring + visualization | Streamlit | Dashboard down → API still works |

---

## Deep-Dive: Training Pipeline

The training pipeline is the core learning loop. It runs on schedule (weekly) or when triggered by drift detection.

### Pipeline Steps

```
Step 1: load_data
  Input:  CSV file path (versioned snapshot)
  Output: Raw DataFrame (7,043 rows × 21 columns)
  Validation: Schema check (column names, types), row count sanity check

Step 2: validate_data
  Input:  Raw DataFrame
  Output: Validated DataFrame (or raises error)
  Checks:  Null rate < 1%, target distribution within expected range,
          no duplicate rows, feature ranges within bounds

Step 3: preprocess
  Input:  Validated DataFrame
  Output: X_train, X_test, y_train, y_test (numpy arrays)
  Actions: Stratified 80/20 split (random_state=42)
           sklearn.Pipeline: StandardScaler (numeric) + OneHotEncoder (categorical)
           Fit on train only, transform both

Step 4: train_model
  Input:  X_train, y_train
  Output: Trained model (XGBoost), fitted preprocessing pipeline
  Config:  XGBClassifier with hyperparameters from algorithm_writeup.md
           MLflow logs: params, metrics, artifacts

Step 5: evaluate_model
  Input:  Trained model, X_test, y_test
  Output: Metrics dict {F2, Recall, Precision, FPR, PR-AUC, ROC-AUC, LogLoss}
  Actions: Compute metrics on test set (touched ONCE)
           Slice-level evaluation (by Contract, InternetService, Tenure)
           Bootstrap 95% confidence intervals (1,000 iterations)
           Generate confusion matrix + SHAP summary plot

Step 6: register_model
  Input:  Trained model, metrics dict
  Output: Model version in MLflow Registry (Staging)
  Gate:    Only register if F2 ≥ 0.55 AND Recall ≥ 0.60 AND FPR ≤ 0.40
           Otherwise, log as failed run and alert
```

### Reproducibility

Every run records:
- **Data**: CSV path + MD5 hash + row count
- **Code**: Git commit hash
- **Config**: All hyperparameters from config file
- **Environment**: Python 3.11, sklearn, xgboost, pandas versions
- **Artifacts**: Serialized model (.pkl), fitted pipeline (.pkl), confusion matrix (.png), SHAP plot (.png)

---

## Deep-Dive: Drift Detection Pipeline

### Why Drift Detection Matters

The model was trained on a snapshot of 7,043 customers. In production, the customer base changes — new customers join, plans get repriced, competitors launch offers. If the input feature distributions shift significantly, the model's predictions become unreliable. This is called **data drift**, and it's the #1 silent killer of production ML systems.

### How It Works

```
Step 1: load_reference_profile
  Input:  Training data statistics (mean, std, distribution per feature)
  Output: Reference profile (computed once, stored as artifact)

Step 2: load_current_data_window
  Input:  Recent prediction inputs (last 7 days of real-time requests)
  Output: Current data window (DataFrame)

Step 3: compute_drift
  Input:  Reference profile, current data window
  Output: Drift report (per-feature + overall)
  Tests:   Kolmogorov-Smirnov test (numeric features)
           Chi-squared test (categorical features)
           Population Stability Index (PSI)

Step 4: evaluate_thresholds
  Input:  Drift report
  Output: Drift decision (drifted / not drifted)
  Thresholds: KS test p-value < 0.05 → feature drifted
              PSI > 0.2 → significant drift
              >30% of features drifted → trigger retrain

Step 5: trigger_retrain (if drifted)
  Input:  Drift decision
  Output: Training pipeline trigger
  Action:  Fire ZenML training pipeline with latest data
           Notify via dashboard alert
```

### Drift Response

| Drift Level | Features Affected | Action |
|-------------|-------------------|--------|
| **None** | <10% features | Log only, no action |
| **Warning** | 10–30% features | Alert on dashboard, schedule retrain within 7 days |
| **Critical** | >30% features | Immediate retrain trigger, alert retention team |

---

## Deep-Dive: Real-Time Serving API

### API Contract

**POST /predict**
```json
// Request
{
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
}

// Response
{
  "churn_probability": 0.72,
  "risk_tier": "HIGH",
  "top_risk_factors": [
    {"feature": "Contract=Month-to-month", "impact": "+0.18"},
    {"feature": "tenure=12", "impact": "+0.12"},
    {"feature": "InternetService=Fiber optic", "impact": "+0.09"}
  ],
  "model_version": "v3",
  "prediction_id": "uuid-..."
}
```

**GET /health**
```json
{
  "status": "healthy",
  "model_version": "v3",
  "model_stage": "Production",
  "uptime_seconds": 86400,
  "last_training_date": "2025-05-20"
}
```

**GET /model-info**
```json
{
  "version": "v3",
  "stage": "Production",
  "training_date": "2025-05-20",
  "metrics": {
    "f2_score": 0.61,
    "recall": 0.67,
    "precision": 0.48,
    "fpr": 0.32
  },
  "training_data_hash": "md5:abc123..."
}
```

### Serving Architecture

```
Client Request
      │
      ▼
┌─────────────┐
│   FastAPI   │
│   (uvicorn) │
└──────┬──────┘
       │
       ├──→ Load serialized preprocessing pipeline (fitted on training data)
       ├──→ Load serialized model (from MLflow Registry, Production stage)
       ├──→ Transform input features (same pipeline as training)
       ├──→ Predict (model.predict_proba)
       ├──→ Compute SHAP values for top risk factors
       ├──→ Log prediction to SQLite (for drift detection + monitoring)
       └──→ Return JSON response
```

**Key design decision**: The preprocessing pipeline used at serving time is the **exact same fitted pipeline** from training. It's serialized as a `.pkl` file and loaded at startup. This eliminates training-serving skew — the #1 source of bugs in production ML.

---

## Tradeoffs

| Decision | Chosen | Alternative | Reason |
|----------|--------|-------------|--------|
| **Orchestration** | ZenML (local) | Airflow, Prefect | ZenML is purpose-built for ML pipelines; handles artifact tracking, experiment tracking, and model registry in one stack. Airflow is overkill for this scale. |
| **Serving** | FastAPI (real-time) | Batch only | Real-time needed for customer service reps. FastAPI is lightweight, async, and has automatic OpenAPI docs. |
| **Database** | SQLite (prediction logs) | PostgreSQL, MongoDB | SQLite is zero-config, file-based, and sufficient for ~10 QPS. PostgreSQL adds operational complexity without benefit at this scale. |
| **Drift detection** | Evidently | Custom statistical tests, WhyLabs | Evidently is open-source, generates ready-made reports, and integrates with ZenML. WhyLabs is paid. Custom tests require maintenance. |
| **Model selection** | XGBoost primary, LR baseline | Neural networks, ensemble | XGBoost is SOTA on tabular data with 7K rows. Neural networks need 10K+ rows to beat gradient boosting. Ensemble adds complexity without proportional gain. |
| **Deployment** | Local Docker | Cloud (AWS/GCP) | $0 cost. All tools run locally. Docker ensures reproducibility. Cloud adds cost and complexity without benefit for this scale. |
| **Monitoring** | Streamlit dashboard | Grafana, custom | Streamlit is Python-native, fast to build, and sufficient for a single-user dashboard. Grafana is overkill. |
| **Retraining trigger** | Drift-based + weekly schedule | Continuous, performance-based | Continuous retraining is expensive and risky. Drift-based catches distribution shifts. Weekly schedule ensures regular refresh even without drift. Performance-based requires ground truth labels which are delayed. |

---

## Operational Concerns

### Monitoring

**Four Golden Signals (per service):**

| Signal | What to Track | Alert Threshold |
|--------|---------------|-----------------|
| **Latency** | P50, P95, P99 response time | P95 > 100ms |
| **Traffic** | Requests/minute | Drop to 0 for 5 min |
| **Errors** | 5xx rate, prediction failures | Error rate > 1% |
| **Saturation** | CPU, memory, disk usage | CPU > 80% for 10 min |

**ML-Specific Metrics:**

| Metric | What to Track | Alert Threshold |
|--------|---------------|-----------------|
| **Prediction distribution** | Mean churn probability | Shift > 10% from baseline |
| **Feature drift** | KS test p-value per feature | p < 0.05 for >30% features |
| **Model performance** | F2-score (when labels available) | F2 < 0.50 |
| **Data quality** | Null rate, schema violations | Null rate > 1% |

### Deployment

**Strategy**: Blue-green deployment for model updates
1. New model registered in MLflow Staging
2. Evaluation gate: new model must beat production model on F2, Recall, FPR
3. If gate passes: promote to Production, restart FastAPI with new model
4. If gate fails: keep production model, alert team
5. Rollback: previous model version always available in registry; instant rollback by reassigning Production stage

**Database migrations**: Not applicable (SQLite is schema-less for prediction logs). Feature pipeline changes are backward-compatible by design.

### Failure Mode Analysis

| Component | Failure | Impact | Mitigation |
|-----------|---------|--------|------------|
| Training pipeline | Step fails | No new model | Retry 3×, then alert. Production model continues serving. |
| Drift detection | Missed drift | Silent model degradation | Weekly scheduled retrain as safety net |
| FastAPI server | Crash | Real-time predictions unavailable | Health check + auto-restart (Docker). Batch predictions still work. |
| MLflow registry | Corruption | Can't load models | Models also stored as `.pkl` files; can rebuild registry |
| Prediction DB | Disk full | Can't log predictions | Rotate logs older than 90 days; alert at 80% disk |
| Bad model promoted | Worse predictions | Business impact | Champion-challenger gate; automatic rollback if F2 drops >10% |

### Cost Estimation

| Component | Cost | Notes |
|-----------|------|-------|
| Compute (local) | $0 | Runs on existing Windows 10 machine |
| Storage | ~10 GB/year | Prediction logs + model artifacts |
| Docker | $0 | Open-source |
| All tools | $0 | ZenML, MLflow, Evidently, FastAPI, Streamlit, sklearn, XGBoost — all free |
| **Total** | **$0** | Entirely local, open-source stack |

### MLOps Maturity Level

**Current target: Level 2 (CI/CD for ML)**

| Level | Status | What's Needed |
|-------|--------|---------------|
| Level 0 — Manual | ✅ Past | Notebooks → pipelines |
| Level 1 — Pipeline Automation | ✅ Current | ZenML pipelines, MLflow tracking, model registry |
| Level 2 — CI/CD for ML | 🔄 In Progress | GitHub Actions for lint + test + Docker build; automated evaluation gates |
| Level 3 — Full Automation | ⏳ Future | Automated drift response, auto-retrain, auto-promote |

---

## MLOps Lifecycle: 10 Stages

| # | Stage | Status | Implementation |
|---|-------|--------|----------------|
| 1 | Data Ingestion | ✅ | ZenML step: load CSV, versioned snapshot |
| 2 | Data Validation | ✅ | ZenML step: schema, null, distribution checks |
| 3 | Feature Engineering | ✅ | sklearn.Pipeline: StandardScaler + OneHotEncoder |
| 4 | Model Training | ✅ | XGBoost via ZenML step, MLflow tracking |
| 5 | Model Evaluation | ✅ | Test set metrics + slice evaluation + bootstrap CI |
| 6 | Model Registry | ✅ | MLflow Registry with stage transitions |
| 7 | Deployment | ✅ | FastAPI serving endpoint, Docker containerization |
| 8 | Monitoring | ✅ | Streamlit dashboard + SQLite prediction logs |
| 9 | Drift Detection | ✅ | Evidently pipeline, daily schedule |
| 10 | Retraining Trigger | ✅ | Drift-based + weekly schedule, quality gates |

---

## Pipeline Decomposition

### Training Pipeline
```
load_data → validate_data → preprocess → train_model → evaluate_model → register_model
```
**Trigger**: Weekly schedule OR drift detection alert
**Output**: New model version in MLflow Staging (if quality gates pass)

### Inference Pipeline (Real-Time)
```
FastAPI /predict → load_model + pipeline → transform → predict → log → return
```
**Trigger**: HTTP request
**Latency**: <100ms P95

### Inference Pipeline (Batch)
```
ZenML step → load_all_customers → transform → predict → save_to_DB
```
**Trigger**: Daily schedule (cron)
**Output**: Churn risk scores for all customers

### Drift Detection Pipeline
```
load_reference → load_current_window → compute_drift → evaluate_thresholds → trigger_if_needed
```
**Trigger**: Daily schedule
**Output**: Evidently drift report + retrain trigger (if drift detected)

### Monitoring Pipeline
```
collect_prediction_logs → compute_metrics → update_dashboard → evaluate_alerts
```
**Trigger**: Continuous (dashboard refreshes every 60s)
**Output**: Streamlit dashboard + alerts

### Retraining Pipeline
```
trigger → run_training_pipeline → compare_with_production → promote_if_better → notify
```
**Trigger**: Drift detection OR weekly schedule
**Output**: New model in Production (if it beats current)
