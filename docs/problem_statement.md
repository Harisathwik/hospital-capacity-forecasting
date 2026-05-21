# Problem Statement: Hospital Bed Demand Forecasting

## Business Context

Hospitals need to know how many ICU beds they'll need in the next 7 days. Getting this wrong has immediate, visible consequences:
- **Too few beds:** Patient safety risk, emergency diversions, staff burnout
- **Too many beds:** Wasted resources, idle staff, unnecessary costs

Currently, most hospitals use simple rules of thumb ("yesterday's count + 10%") or manual estimates by experienced administrators. These break down during flu season, COVID waves, or unexpected surges.

## ML Formulation

- **Problem type:** Time-series regression
- **Target variable:** `icu_bed_count` — number of ICU beds needed per day
- **Prediction horizon:** 7 days ahead
- **Primary metric:** RMSE — because large errors (missing 20+ beds) are disproportionately worse than many small errors
- **Guardrail metrics:** MAE, MAPE, Prediction Interval Coverage (95%)
- **Current baseline:** "Yesterday's count" naive forecast

## Data Summary

| Source | Features | Frequency | Rows |
|--------|----------|-----------|------|
| Hospital Records | Admissions, discharges, ICU transfers, bed count | Daily | ~1,000 days |
| CDC ILINet | Influenza-like illness rates by region | Weekly | ~140 weeks |
| Weather API | Temperature, humidity, precipitation | Daily | ~1,000 days |
| Calendar | Day of week, holidays, flu season flag | Static | 365/year |

**Known issues:**
- Missing data during system outages
- ILINet data is weekly (needs interpolation for daily features)
- Hospital data may have reporting delays of 1-2 days

## Feature Engineering

**Lag features:** bed count at 1, 7, 14 days ago
**Rolling statistics:** 7-day and 14-day rolling mean, std, min, max
**Calendar features:** day of week, month, is_holiday, is_flu_season
**External features:** temperature, humidity, ILI rate (interacted with flu season)

## Constraints

- **Latency:** Batch prediction daily at 6 AM (not real-time)
- **Interpretability:** Hospital administrators need to understand why the model predicts a surge
- **Regulatory:** Patient data must be de-identified; model decisions affect patient safety

## Success Criteria

The model is in production when:
1. RMSE is at least 20% better than the naive baseline
2. 95% prediction intervals actually contain the true value 93-97% of the time
3. Drift detection is operational and alerts are routed to the right people
4. A rollback procedure is tested and takes under 5 minutes
