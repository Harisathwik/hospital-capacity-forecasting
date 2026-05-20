# Problem Statement: Telco Customer Churn Prediction

## One-Sentence Formulation

Given a telecom customer's demographic profile, subscribed services, and account history, predict whether the customer will churn (leave the company) in the near term, for the retention/marketing team, at the point of decision before the customer actually leaves, to reduce customer acquisition costs and revenue loss by enabling proactive retention interventions.

---

## Business Context

### The Problem

Customer churn — when a subscriber leaves a telecom provider for a competitor or cancels service entirely — is one of the most expensive problems in the telecommunications industry. Telecom companies operate in a near-saturated market where acquiring a new customer costs **5–7x more** than retaining an existing one. Industry benchmarks put the average cost of acquiring a new telecom customer at $300–$400, while retaining an existing one through targeted intervention costs $50–$100.

For a mid-sized telecom provider with 100,000 subscribers and a 26.5% annual churn rate, that's roughly 26,500 customers leaving per year. Even a modest 10% reduction in churn (saving 2,650 customers) translates to **$795,000–$1,060,000 in saved acquisition costs annually** — not counting the preserved monthly recurring revenue from those retained customers.

### Why This Matters Now

Telecom churn is not random. Customers leave for identifiable reasons: poor service quality, better competitor pricing, lack of engagement, contract expiration, or life events. The key insight is that **churn is predictable** — customers exhibit behavioral and demographic signals before they leave. A model that can identify at-risk customers 30–60 days before they churn gives the retention team a critical window to intervene with targeted offers (discounts, plan upgrades, loyalty rewards).

### Current Baseline

Today, most telecom companies handle churn reactively:
- **No prediction**: Customers are only identified as "churned" after they've already left. The retention team has no early warning system.
- **Simple heuristics**: Some companies use basic rules like "flag customers whose contract expired" or "flag customers who called support 3+ times." These catch obvious cases but miss the majority of at-risk customers.
- **Blanket retention offers**: Without targeting, companies send retention offers to everyone — including customers who would have stayed anyway — wasting budget and training customers to expect discounts.

**The baseline is effectively random**: the company retains customers at the natural retention rate of ~73.5%, with no systematic ability to identify who is at risk. Any model that can predict churn better than random and enable targeted intervention will generate measurable ROI.

---

## ML Formulation

### Problem Type
**Binary Classification** — The target variable `Churn` has two values: `Yes` (customer left) or `No` (customer stayed).

### Target Variable
- **Name**: `Churn`
- **Definition**: Whether the customer left the telecom provider (Yes) or remained a subscriber (No) during the period captured in the dataset
- **Type**: Binary categorical (Yes/No)
- **Positive class**: `Yes` (churned) — this is the class we want to detect

### Class Imbalance
- **No (retained)**: 5,174 customers (73.46%)
- **Yes (churned)**: 1,869 customers (26.54%)
- **Imbalance ratio**: ~2.77:1 (retained:churned)

This is a **moderate imbalance** — not extreme enough to require aggressive resampling, but significant enough that accuracy is a misleading metric. A naive "predict No for everyone" model would achieve 73.5% accuracy while catching zero churners.

### Primary Metric: **F2-Score (F-beta with β=2)**

**Why F2?** Because in churn prediction, **false negatives are far more expensive than false positives**.

- **False Negative (missed churner)**: The model says "this customer will stay," but they actually leave. Cost: lost customer, lost lifetime revenue ($500–$2,000+ per customer), plus acquisition cost for a replacement.
- **False Positive (false alarm)**: The model says "this customer will churn," but they would have stayed anyway. Cost: unnecessary retention offer ($20–$50 discount or perk). Annoying but cheap.

The F2-score weights recall 2x more than precision, reflecting this asymmetry. We'd rather waste a retention offer on a happy customer than miss a churner entirely.

### Guardrail Metrics
1. **False Positive Rate (FPR)**: Must stay below 40%. If the model flags too many non-churners, the retention budget gets wasted and customers get spammed with unwanted offers.
2. **PR-AUC (Precision-Recall Area Under Curve)**: More informative than ROC-AUC for imbalanced data. Measures the model's ability to distinguish churners from non-churners across all thresholds.
3. **Recall (Sensitivity)**: Must be above 60%. We need to catch at least 60% of actual churners for the system to be worthwhile.

### Success Criteria
The model is considered production-worthy when:
- F2-score ≥ 0.55 on the held-out test set
- Recall ≥ 0.60 (catches 60%+ of churners)
- FPR ≤ 0.40 (no more than 40% false alarms)
- PR-AUC ≥ 0.60

These thresholds are intentionally modest for the first iteration. The goal is to beat the baseline (no model) and demonstrate value, then improve.

---

## Metric Ladder

| Level | Metric | Target | Why It Matters |
|-------|--------|--------|----------------|
| **Business Outcome** | Customer retention rate improvement | +5–10% relative reduction in churn | The north star — more customers staying means more revenue |
| **Product Metric** | Retention offer acceptance rate | >30% | If the model flags at-risk customers but they don't accept offers, the system fails |
| **Model Metric** | F2-score | ≥0.55 | Balances catching churners (recall) with not wasting budget (precision) |
| **Data Quality Metric** | Schema validity, null rate, distribution drift | <1% nulls, no schema changes | Garbage in, garbage out — the model is only as good as its input data |

---

## Data Summary

### Source
- **Dataset**: IBM Telco Customer Churn (publicly available on Kaggle)
- **URL**: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Rows**: 7,043 customers
- **Features**: 19 input features + 1 target + 1 customer ID
- **Time period**: Cross-sectional snapshot (single point in time per customer)

### Feature Breakdown

**Demographic Features (4):**
| Feature | Type | Values | Notes |
|---------|------|--------|-------|
| gender | Categorical | Male, Female | Binary |
| SeniorCitizen | Binary | 0, 1 | ~16% are seniors |
| Partner | Categorical | Yes, No | Has domestic partner |
| Dependents | Categorical | Yes, No | Has dependents |

**Service Features (9):**
| Feature | Type | Values | Notes |
|---------|------|--------|-------|
| PhoneService | Categorical | Yes, No | ~90% have phone |
| MultipleLines | Categorical | Yes, No, No phone service | Correlated with PhoneService |
| InternetService | Categorical | DSL, Fiber optic, No | ~44% Fiber, ~34% DSL, ~22% no internet |
| OnlineSecurity | Categorical | Yes, No, No internet service | Add-on service |
| OnlineBackup | Categorical | Yes, No, No internet service | Add-on service |
| DeviceProtection | Categorical | Yes, No, No internet service | Add-on service |
| TechSupport | Categorical | Yes, No, No internet service | Add-on service |
| StreamingTV | Categorical | Yes, No, No internet service | Add-on service |
| StreamingMovies | Categorical | Yes, No, No internet service | Add-on service |

**Account Features (5):**
| Feature | Type | Range | Notes |
|---------|------|-------|-------|
| tenure | Numeric (int) | 0–72 months | Mean 32.4 months. Strong churn predictor. |
| Contract | Categorical | Month-to-month, One year, Two year | **Strongest churn predictor**: 42.7% churn for month-to-month vs 2.8% for two-year |
| PaperlessBilling | Categorical | Yes, No | ~59% use paperless |
| PaymentMethod | Categorical | Electronic check, Mailed check, Bank transfer, Credit card | Electronic check users churn more |
| MonthlyCharges | Numeric (float) | $18.25–$118.75 | Mean $64.76. Higher charges → higher churn |
| TotalCharges | Numeric (string→float) | $18.80–$8,684.80 | 11 blank values (tenure=0 customers). Derived from tenure × MonthlyCharges. |

### Key Data Quality Issues
1. **TotalCharges is stored as string** — needs conversion to float. 11 blank values correspond to customers with tenure=0 (new customers who haven't been billed yet).
2. **No missing values** in other columns — dataset is clean.
3. **No duplicate rows** — each row is a unique customer.
4. **Derived feature risk**: TotalCharges ≈ tenure × MonthlyCharges, creating multicollinearity. May need to drop or transform.

### Churn Patterns (EDA Insights)

| Segment | Churn Rate | Insight |
|---------|-----------|---------|
| **Contract: Month-to-month** | 42.7% | Highest risk — no commitment |
| **Contract: One year** | 11.3% | Moderate risk |
| **Contract: Two year** | 2.8% | Very low — locked in |
| **Internet: Fiber optic** | 41.9% | High churn — likely price-sensitive |
| **Internet: DSL** | 19.0% | Moderate |
| **Internet: No** | 7.4% | Low — basic service, less to churn from |
| **Tenure: 0–12 months** | 47.7% | New customers are flight risks |
| **Tenure: 48–72 months** | 9.5% | Loyal, long-term customers |
| **MonthlyCharges: $65–95** | 35.9% | Mid-tier plans have highest churn |
| **MonthlyCharges: $0–35** | 10.9% | Basic plans — low churn |

**Key takeaway**: The strongest predictors of churn are contract type, tenure, and internet service type. Month-to-month customers with fiber optic internet and low tenure are the highest-risk segment.

---

## Constraints

### Latency
- **Batch prediction**: Primary mode. Run daily/weekly to score all customers and generate a "churn risk" list for the retention team.
- **Real-time API**: Secondary mode. On-demand scoring via FastAPI for customer service reps who want to check a specific customer's churn risk during a call. Target latency: <100ms per prediction.

### Interpretability
- **Required**: The retention team needs to understand *why* a customer is flagged as high-risk. "This customer is at risk because they're on a month-to-month contract, have fiber optic internet, and have only been with us for 3 months" is actionable. "The model says so" is not.
- **Approach**: Use SHAP values or feature importance to provide per-prediction explanations.

### Regulatory
- **No PII in model features**: The dataset contains no names, addresses, SSNs, or other regulated PII. `customerID` is excluded from training.
- **No regulatory compliance requirements** (HIPAA, GDPR) for this dataset since it's synthetic/public. But the system design should be extensible to handle such requirements.

### Budget
- **All tools are free/open-source**: ZenML, MLflow, Evidently, FastAPI, Streamlit, scikit-learn, XGBoost.
- **No cloud costs**: Everything runs locally. Docker for containerization.

---

## Framework

- **Orchestration**: ZenML — handles pipeline steps, artifact tracking, and reproducibility
- **Experiment Tracking**: MLflow — logs parameters, metrics, and artifacts for every training run
- **Model Registry**: MLflow — versioned model storage with stage transitions (Staging → Production)
- **Drift Detection**: Evidently — monitors feature distributions and triggers retraining
- **Serving**: FastAPI — REST API for real-time predictions
- **Dashboard**: Streamlit — monitoring and visualization

---

## Success Criteria

The project is considered complete when:

1. **Training pipeline** runs end-to-end via ZenML: data loading → preprocessing → training → evaluation → model registration
2. **Model meets thresholds**: F2 ≥ 0.55, Recall ≥ 0.60, FPR ≤ 0.40 on test set
3. **FastAPI serving layer** is functional: accepts customer features, returns churn probability and risk tier
4. **Drift detection** is operational: Evidently monitors incoming predictions and flags distribution shifts
5. **Streamlit dashboard** displays: model metrics, drift scores, prediction history, feature importance
6. **CI/CD pipeline** runs on GitHub Actions: lint → test → build Docker image
7. **Documentation** is complete: README, architecture diagram, API docs, setup instructions

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model doesn't beat baseline | Wasted effort | Start with logistic regression baseline; if it doesn't beat random, the features may not have signal |
| Training-serving skew | Model performs worse in production | Use sklearn.Pipeline to bundle preprocessing; serialize and load the same pipeline at serving time |
| Data drift degrades model | Silent failure | Evidently monitors distributions; auto-retrain trigger when drift exceeds threshold |
| Class imbalance hurts recall | Missed churners | Use F2-score (weights recall higher); consider SMOTE or class weights if needed |
| Overfitting to training set | Poor generalization | Hold-out test set (80/20 stratified split); cross-validation during training |

---

## Deferred (Out of Scope for v1)

- **A/B testing framework**: Compare model-driven retention vs. current approach in production
- **Feature store**: Not needed for a single static dataset; add if data sources multiply
- **Multi-model ensemble**: Start with XGBoost; add ensemble if single model isn't enough
- **Real-time streaming**: Batch predictions are sufficient for v1
- **Customer lifetime value (CLV) integration**: Would improve retention offer targeting but requires additional data
- **Automated retraining in production**: Manual retrain trigger for v1; automate in v2
