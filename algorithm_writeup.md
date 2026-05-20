# Algorithm Writeup: Telco Customer Churn Prediction

## 1. Overview

This document describes the algorithms, training strategy, and evaluation approach for the Telco Customer Churn Prediction system. It covers the baseline model, candidate models, hyperparameter strategy, class imbalance handling, and evaluation methodology.

**Problem type**: Binary classification (Churn: Yes/No)
**Primary metric**: F2-score (recall-weighted F-beta)
**Dataset size**: 7,043 rows, 19 features
**Class imbalance**: 26.54% positive (Yes) / 73.46% negative (No) — ~2.77:1 ratio

---

## 2. Baseline Model — Logistic Regression

### Why Logistic Regression First?

Every ML project needs a baseline — the simplest model that establishes a performance floor. Logistic regression is the right baseline for this problem because:

1. **It's fast to train** — seconds, not minutes. You get a result immediately.
2. **It's interpretable** — coefficients directly tell you which features push toward churn vs. retention. The retention team can understand "customers with month-to-month contracts are 3x more likely to churn."
3. **It reveals signal strength** — if logistic regression gets F2=0.50, the features have signal. If it gets F2=0.20, the features may not be predictive enough and you need better data, not better models.
4. **It sets the floor** — any complex model (XGBoost, random forest) must beat this. If it can't, the added complexity isn't justified.

### Configuration

```
Model: LogisticRegression (sklearn)
Penalty: L2 (Ridge regularization)
C: 1.0 (default regularization strength)
class_weight: "balanced" — compensates for 2.77:1 imbalance by up-weighting minority class errors
solver: "lbfgs" — efficient for small datasets
max_iter: 1000 — ensure convergence
random_state: 42 — reproducibility
```

### Preprocessing for Logistic Regression

Logistic regression is sensitive to feature scale and requires numeric inputs only:

| Feature Type | Transformation | Reason |
|-------------|---------------|--------|
| Numeric (tenure, MonthlyCharges, TotalCharges) | StandardScaler | LR with regularization assumes comparable feature scales. Without scaling, MonthlyCharges ($18–$119) dominates tenure (0–72). |
| Binary categorical (gender, Partner, Dependents, PhoneService, PaperlessBilling) | OrdinalEncoder (0/1) | Binary features don't need one-hot; 0/1 encoding is sufficient. |
| Multi-class categorical (InternetService, Contract, PaymentMethod, MultipleLines, service add-ons) | OneHotEncoder (drop='first') | Avoids multicollinearity from dummy variable trap. Low cardinality (2–4 values) so feature explosion is minimal. |

**All preprocessing bundled in a single sklearn.Pipeline** to prevent training-serving skew. The fitted pipeline is serialized and loaded at serving time.

### Expected Performance

Based on published benchmarks for this dataset:
- F2-score: ~0.50–0.58
- Recall: ~0.55–0.65
- Precision: ~0.40–0.50
- PR-AUC: ~0.55–0.65

If the baseline meets the success thresholds (F2 ≥ 0.55, Recall ≥ 0.60), it could be the production model. Always try the simple thing first.

---

## 3. Candidate Model 1 — XGBoost (Primary)

### Why XGBoost?

XGBoost (eXtreme Gradient Boosting) is the primary candidate because:

1. **State-of-the-art on tabular data** — XGBoost consistently wins Kaggle competitions on structured/tabular problems. It's the default "strong model" for tabular ML.
2. **Handles mixed feature types natively** — works with numeric and encoded categorical features without requiring the same preprocessing as linear models.
3. **Built-in regularization** — L1 (alpha) and L2 (lambda) regularization prevent overfitting, critical for a 7K-row dataset.
4. **Handles moderate class imbalance** — the `scale_pos_weight` parameter directly adjusts for the 2.77:1 ratio without needing SMOTE.
5. **Feature importance** — built-in feature importance (gain, weight, cover) provides interpretability for the retention team.
6. **Missing value handling** — XGBoost handles missing values natively (learns the best direction to send missing values during splits). This is useful for the 11 blank TotalCharges values.

### Configuration

```
Model: XGBClassifier (xgboost)
n_estimators: 200 (number of boosting rounds/trees)
max_depth: 5 (moderate depth to prevent overfitting on small dataset)
learning_rate: 0.1 (standard default; lower = more robust but slower)
subsample: 0.8 (use 80% of rows per tree — reduces overfitting)
colsample_bytree: 0.8 (use 80% of features per tree — reduces overfitting)
min_child_weight: 3 (minimum sum of instance weight in a child — prevents splits on tiny groups)
gamma: 0.1 (minimum loss reduction for a split — regularization)
reg_alpha: 0.1 (L1 regularization on weights)
reg_lambda: 1.0 (L2 regularization on weights)
scale_pos_weight: 2.77 (ratio of negative to positive class — handles imbalance)
objective: "binary:logistic" (outputs probability)
eval_metric: "logloss" (proper scoring rule for probability calibration)
random_state: 42
n_jobs: -1 (use all CPU cores)
```

### Preprocessing for XGBoost

XGBoost is tree-based, so it's **scale-invariant** — no need for StandardScaler. But it still needs numeric inputs:

| Feature Type | Transformation | Reason |
|-------------|---------------|--------|
| Numeric (tenure, MonthlyCharges, TotalCharges) | Pass through as-is (or RobustScaler if outliers are severe) | Trees split on rank, not magnitude. Scaling doesn't change split quality. |
| All categoricals | OrdinalEncoder (integer encoding) | XGBoost handles integer-encoded categoricals well. One-hot encoding is unnecessary and creates sparse splits. |
| Missing values (TotalCharges) | Leave as NaN | XGBoost handles missing values natively. |

**Note**: We'll test both with and without StandardScaler for XGBoost to confirm it doesn't matter. If results are identical, we skip scaling to keep the pipeline simpler.

### Hyperparameter Tuning Strategy

**Phase 1 — Manual tuning (this writeup):** Start with the configuration above based on best practices for small tabular datasets. Train, evaluate, check if thresholds are met.

**Phase 2 — Automated tuning (if Phase 1 is close but not enough):** Use `RandomizedSearchCV` with 5-fold stratified cross-validation over:

```
n_estimators: [100, 200, 300, 500]
max_depth: [3, 4, 5, 6, 7]
learning_rate: [0.01, 0.05, 0.1, 0.2]
subsample: [0.7, 0.8, 0.9]
colsample_bytree: [0.7, 0.8, 0.9]
min_child_weight: [1, 3, 5]
gamma: [0, 0.1, 0.2]
reg_alpha: [0, 0.1, 0.5]
reg_lambda: [0.5, 1.0, 2.0]
scale_pos_weight: [1.0, 2.0, 2.77, 3.5, 5.0]
```

Search budget: 100 iterations, 5-fold stratified CV, optimized for F2-score. This explores ~100 combinations out of ~19,000 possible — enough to find a good region without exhaustive search.

**Phase 3 — Threshold tuning (always):** After selecting the best model, tune the classification threshold on the validation set. The default 0.5 threshold is almost never optimal for imbalanced data. We search thresholds from 0.1 to 0.9 and pick the one that maximizes F2-score.

### Expected Performance

Based on published benchmarks for this dataset:
- F2-score: ~0.58–0.65
- Recall: ~0.60–0.72
- Precision: ~0.45–0.55
- PR-AUC: ~0.60–0.72

XGBoost should meaningfully beat the logistic regression baseline, especially on recall, because boosting iteratively focuses on hard-to-classify examples (which are often the minority class).

---

## 4. Candidate Model 2 — Random Forest (Comparison)

### Why Random Forest?

Random Forest serves as a comparison model to validate that XGBoost's performance gain is real and not a fluke:

1. **Different inductive bias** — bagging (random forest) vs. boosting (XGBoost). If both agree on performance, the result is more trustworthy.
2. **Less prone to overfitting** — averaging many decorrelated trees is inherently regularized.
3. **Fast to train** — parallelizable, no sequential dependency like boosting.
4. **Feature importance** — provides an alternative importance ranking to cross-validate XGBoost's.

### Configuration

```
Model: RandomForestClassifier (sklearn)
n_estimators: 300
max_depth: 8
min_samples_split: 10
min_samples_leaf: 5
max_features: "sqrt" (standard for classification)
class_weight: "balanced"
random_state: 42
n_jobs: -1
```

### Preprocessing

Same as XGBoost — OrdinalEncoder for categoricals, no scaling needed.

### Expected Performance

- F2-score: ~0.55–0.62
- Typically slightly below XGBoost on this dataset, but the gap should be small. If Random Forest beats XGBoost, it suggests XGBoost is overfitting and needs more regularization.

---

## 5. Class Imbalance Strategy

### The Four-Factor Decision

1. **Minority class ratio**: 26.54% — this is moderate imbalance, not severe. The minority class has 1,869 examples, which is plenty to learn from.
2. **Chosen metric**: F2-score — recall-based, so imbalance matters. The model needs to prioritize catching churners.
3. **Model type**: XGBoost (tree-based) — handles moderate imbalance natively. Less sensitive than logistic regression.
4. **Threshold tuning first**: Always try this before modifying the training data.

### Strategy (In Preference Order)

**Step 1: Threshold tuning (always applied)**
- Train on the original data distribution (no resampling).
- After training, sweep classification thresholds from 0.1 to 0.9 on the validation set.
- Pick the threshold that maximizes F2-score.
- This is the simplest, safest approach — no synthetic data, no leakage risk.

**Step 2: Class weights (applied by default)**
- LogisticRegression: `class_weight="balanced"` — automatically adjusts weights inversely proportional to class frequencies.
- XGBoost: `scale_pos_weight=2.77` — ratio of negative to positive examples. This makes the model treat each churner error as 2.77x more important than each non-churner error.

**Step 3: SMOTE (only if Steps 1+2 are insufficient)**
- If F2 < 0.55 after threshold tuning and class weights, apply SMOTE to the training set only.
- Apply AFTER train/test split — never before. Applying before splitting creates synthetic examples that leak into the test set, inflating metrics.
- Target: oversample minority class to achieve 1:1 ratio (or 1.5:1 if 1:1 causes overfitting).
- Use `SMOTE(random_state=42, k_neighbors=5)` from imbalanced-learn.
- Log the number of synthetic samples created.

**We do NOT plan to use undersampling** — with only 7,043 rows, throwing away data is counterproductive.

---

## 6. Training Strategy

### Data Splitting

```
Method: Stratified train/test split
Train: 80% (5,634 rows) — preserves 26.54% churn rate
Test:  20% (1,409 rows) — preserves 26.54% churn rate
random_state: 42
```

Stratified splitting ensures both sets have the same churn proportion. A random split could accidentally give the test set 30% churn and the training set 23%, making metrics incomparable.

### Cross-Validation (During Tuning)

```
Method: Stratified K-Fold
K: 5
shuffle: True
random_state: 42
```

5-fold stratified CV gives 5 performance estimates per configuration. We use the mean F2-score across folds to compare models, and the standard deviation to assess stability. A model with mean F2=0.60 ± 0.02 is more trustworthy than one with mean F2=0.62 ± 0.08.

### Experiment Tracking (MLflow)

Every training run logs:

| Category | What's Logged |
|----------|--------------|
| **Data** | Dataset path, train/test split sizes, random seed, class distribution |
| **Preprocessing** | Pipeline steps, fitted parameters (scaler mean/std, encoder categories) |
| **Model** | Algorithm name, all hyperparameters, training duration |
| **Metrics** | F2, Recall, Precision, FPR, PR-AUC, ROC-AUC, LogLoss (on test set) |
| **Artifacts** | Serialized model (.pkl), fitted preprocessing pipeline (.pkl), confusion matrix (.png), SHAP summary (.png) |
| **Code** | Git commit hash, config file version |
| **Environment** | Python version, library versions (sklearn, xgboost, pandas) |

Each run gets a unique MLflow run ID. The best run is tagged as `candidate` and registered in the MLflow Model Registry as version N in the `Staging` stage.

### Reproducibility Checklist

- [x] Random seeds fixed: `random_state=42` everywhere
- [x] Data snapshot: specific CSV file, not "latest"
- [x] Library versions pinned in `requirements.txt`
- [x] Config parameters in a separate config file, not hardcoded
- [x] Git commit hash logged with each run
- [x] Environment recorded (Python 3.11, Windows 10)

---

## 7. Evaluation Methodology

### Primary Evaluation (Test Set)

The test set is touched **once** — after all tuning is complete. It provides the unbiased performance estimate.

**Metrics computed:**
1. **F2-score** (primary) — harmonic mean of precision and recall, weighted 2x toward recall
2. **Recall** (guardrail: ≥0.60) — % of actual churners caught
3. **Precision** — % of flagged customers who actually churn
4. **FPR** (guardrail: ≤0.40) — % of non-churners incorrectly flagged
5. **PR-AUC** (guardrail: ≥0.60) — area under precision-recall curve
6. **ROC-AUC** — supplementary; less informative for imbalanced data
7. **LogLoss** — measures probability calibration quality

### Confidence Intervals (Bootstrapping)

A single number without uncertainty is not evidence. We compute 95% confidence intervals via bootstrapping:

```
Procedure:
1. Take test set predictions and true labels
2. Resample with replacement to create a bootstrap sample of the same size (1,409 rows)
3. Compute F2, Recall, Precision, FPR on the bootstrap sample
4. Repeat 1,000 times
5. Report 2.5th and 97.5th percentiles as the 95% CI
```

Example output: "F2 = 0.61 (95% CI: 0.56–0.66)" — this means we're 95% confident the true F2 is between 0.56 and 0.66.

### Slice-Level Evaluation

Global metrics hide segment failures. We evaluate performance across business-relevant slices:

| Slice | Categories | Why It Matters |
|-------|-----------|----------------|
| Contract | Month-to-month, One year, Two year | Contract type is the strongest churn predictor. The model must work for all contract types, not just month-to-month. |
| InternetService | DSL, Fiber optic, No | Fiber optic users churn at 41.9% — the model must perform well here. |
| Tenure | 0–12, 12–24, 24–48, 48–72 months | New customers (0–12) churn at 47.7%. The model must catch these. |
| MonthlyCharges | $0–35, $35–65, $65–95, $95–120 | Mid-tier plans have highest churn. |
| SeniorCitizen | 0, 1 | Fairness check — model shouldn't discriminate against seniors. |
| Gender | Male, Female | Fairness check — model should perform equally across genders. |

**Red flag**: If F2 for month-to-month customers is 0.65 but for two-year customers is 0.20, the model is only working for the easy cases. This needs to be flagged and addressed.

### Confusion Matrix Analysis

For the final model, we produce a confusion matrix at the chosen threshold:

```
                    Predicted
                  No      Yes
Actual No    [  TN  |  FP  ]  → FPR = FP / (FP + TN)
Actual Yes   [  FN  |  TP  ] → Recall = TP / (TP + FN)
```

We analyze:
- **FN count**: How many churners did we miss? At what cost?
- **FP count**: How many false alarms? At what cost?
- **Cost-weighted analysis**: FN cost ($500–$2,000) × FN count vs. FP cost ($20–$50) × FP count

---

## 8. Model Comparison Framework

### How We Compare Models

| Criterion | Weight | How Measured |
|-----------|--------|-------------|
| F2-score (primary) | 40% | Higher is better; must be ≥0.55 |
| Recall (guardrail) | 25% | Must be ≥0.60; higher is better |
| FPR (guardrail) | 15% | Must be ≤0.40; lower is better |
| PR-AUC | 10% | Higher is better; measures ranking quality |
| Training time | 5% | Lower is better; relevant for retraining frequency |
| Interpretability | 5% | Higher is better; can the retention team understand it? |

### Decision Rules

1. **If no model meets all guardrails**: Do not ship. Go back to feature engineering or collect more data.
2. **If only the baseline meets thresholds**: Ship the baseline. Simple and working beats complex and broken.
3. **If XGBoost beats baseline by >0.05 F2**: Ship XGBoost. The complexity is justified.
4. **If XGBoost beats baseline by <0.05 F2**: Consider shipping the baseline. The marginal gain may not justify the operational complexity.
5. **If Random Forest beats XGBoost**: Investigate why. XGBoost may be overfitting. Increase regularization or reduce n_estimators.

---

## 9. Interpretability Plan

The retention team needs to understand *why* a customer is flagged. We provide:

### Global Interpretability
- **Feature importance plot** (XGBoost gain-based): Shows which features matter most overall.
- **SHAP summary plot**: Shows the direction and magnitude of each feature's effect on churn probability. E.g., "month-to-month contract increases churn probability by +0.15; two-year contract decreases by -0.12."

### Local Interpretability (Per-Customer)
- **SHAP force plot** for individual predictions: "This customer has a 72% churn probability. Top risk factors: month-to-month contract (+0.18), tenure=3 months (+0.12), fiber optic internet (+0.09). Protective factors: OnlineSecurity=Yes (-0.05)."
- This is exposed via the FastAPI `/predict` endpoint as a `top_risk_factors` field in the response.

---

## 10. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Overfitting to training set (7K rows is small) | High | Model fails in production | Regularization (max_depth=5, subsample=0.8), early stopping, cross-validation |
| Training-serving skew | Medium | Silent production failure | sklearn.Pipeline bundles preprocessing; serialize and load at serving time |
| Class imbalance causes low recall | Medium | Miss churners | scale_pos_weight=2.77, threshold tuning, F2 optimization |
| TotalCharges multicollinearity (≈ tenure × MonthlyCharges) | Low | Unstable coefficients | Drop TotalCharges or create derived feature (avg_monthly_charge = TotalCharges/tenure) |
| Model doesn't beat baseline | Low | Wasted effort | If this happens, the features lack signal — need better data, not better models |
| Feature distribution shift in production | Medium | Silent degradation | Evidently drift detection monitors feature distributions; triggers retrain |

---

## 11. Deferred (Out of Scope for v1)

- **Neural networks / deep learning**: Overkill for 7K rows and 19 features. Tabular deep learning (TabNet, FT-Transformer) needs 10K+ rows to beat gradient boosting.
- **Ensemble of ensembles**: Stacking XGBoost + Random Forest + Logistic Regression. Marginal gain, high complexity. Add in v2 if single models plateau.
- **Automated feature engineering** (Featuretools, autoFE): Manual feature engineering is sufficient for 19 features. AutoFE adds complexity without proportional gain.
- **Bayesian hyperparameter optimization** (Optuna, Hyperopt): RandomizedSearchCV is sufficient for v1. Bayesian optimization is more efficient but adds dependency.
- **Cost-sensitive learning with custom loss**: The cost ratio (FN:FP ≈ 10–40x) is approximated by F2-score and scale_pos_weight. A custom loss function could be more precise but adds complexity.
- **Online/incremental learning**: Batch retraining is sufficient. Online learning adds complexity and risk without clear benefit for this use case.
