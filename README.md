# Loan Default Risk Prediction

Predicts whether a loan applicant will default within 2 years, and explains why — delivered as a live API, not just a notebook.

## Problem

Lenders often rely on manual credit checks and rigid rule-based cutoffs (fixed income or credit score thresholds) that are blunt instruments — they either reject too many creditworthy applicants or approve too many risky ones. This project builds a model that scores default probability from applicant data, giving lenders a faster, more calibrated approve/reject decision — with the tradeoff between catching defaulters and rejecting good borrowers made explicit and tunable, rather than hidden inside a black-box accuracy number.

## Dataset

[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) (Kaggle, 2011) — 150,000 borrowers, target `SeriousDlqin2yrs` (1 = serious delinquency within 2 years). ~6.7% positive class — a genuinely imbalanced, realistic dataset, not a toy 50/50 split.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.635 | 0.053 | 0.098 | 0.812 |
| Logistic Regression (class-weighted) | 0.183 | 0.746 | 0.294 | 0.833 |
| Logistic Regression (SMOTE) | 0.180 | 0.748 | 0.290 | 0.826 |
| XGBoost (default params) | 0.233 | 0.704 | 0.350 | 0.849 |
| **XGBoost (tuned, final model)** | 0.210 | 0.790 | 0.340 | **0.869** |

The baseline model's 93% accuracy was misleading — it caught only 5% of real defaulters, since it could hit high accuracy just by predicting "safe" for nearly everyone on this imbalanced dataset. Every subsequent step addresses that directly.

**Final decision threshold: 0.3** (not the default 0.5). At this threshold, the tuned model catches **~90% of real defaulters**, accepting a lower precision in return — because in this problem, missing a defaulter is assumed to cost more than an unnecessary manual review of a safe applicant.

## What actually mattered to the model

SHAP analysis on the final model revealed:
- **`TotalTimesPastDue`** (an engineered feature combining three raw delinquency columns) was the single most predictive feature — despite showing only weak linear correlation (0.116) in EDA. This confirmed that XGBoost was capturing a non-linear signal a simple correlation check couldn't detect.
- **`RevolvingUtilizationOfUnsecuredLines`** was the second most important feature — but only *after* cleaning. Its correlation with default was -0.002 (meaningless) in the raw data, and jumped to 0.28 once extreme outlier values (up to 50,708 on a ratio that should rarely exceed ~1) were capped. This is direct evidence that the data cleaning step had a measurable, not just cosmetic, impact.
- The three original delinquency columns became redundant once `TotalTimesPastDue` existed — dropping them cost less than 0.002 ROC-AUC, so the simplified 9-feature model was kept over the original 12-feature version.

## Approach

1. EDA — found severe class imbalance (93.3%/6.7%), age and dependents as real risk signals, mostly weak individual feature correlations
2. Data cleaning — traced broken values (an impossible `age=0`, `DebtRatio` values over 300,000) back to their root cause (missing income data) rather than blindly capping everything the same way
3. Feature engineering — combined three delinquency columns into `TotalTimesPastDue`
4. Compared 4 models: baseline Logistic Regression, class-weighted LR, SMOTE-resampled LR, and XGBoost
5. Hyperparameter tuning via `RandomizedSearchCV`
6. Threshold tuning, justified against the stated business cost asymmetry (not left at the default 0.5)
7. SHAP explainability — verified both global feature importance and used it to justify simplifying the feature set
8. Deployed as a FastAPI backend with a `/predict` endpoint returning a risk score, recommendation, and the top 5 factors behind each individual prediction

## Deployment

Built as a **FastAPI** backend rather than a UI-only demo, since the intended real-world use case is another system (a bank's loan processing pipeline) calling this programmatically — not a human clicking through a page.

**Run it:**
```bash
cd app
uvicorn main:app --reload
```

Interactive docs (test it directly in your browser): `http://127.0.0.1:8000/docs`

**Example request/response:**

Request:
```json
{
  "age": 24, "monthly_income": 1800, "dependents": 3,
  "open_credit_lines": 2, "revolving_utilization": 1.4,
  "debt_ratio": 0.9, "real_estate_loans": 0, "total_times_past_due": 6
}
```

Response:
```json
{
  "risk_score": 0.975,
  "recommendation": "Reject / Flag for Review",
  "threshold_used": 0.3,
  "top_factors": [
    {"feature": "TotalTimesPastDue", "impact": 2.1964},
    {"feature": "RevolvingUtilizationOfUnsecuredLines", "impact": 0.9509},
    {"feature": "age", "impact": 0.2016}
  ]
}
```

## What I'd improve for a real deployment

This project demonstrates the core technical pipeline behind a lending risk model, but several things would be non-negotiable before any real institution could use it:

- **Fairness/bias auditing** — this dataset has no demographic fields to check against, but a real deployment legally requires checking for disparate impact across protected classes (e.g., under US ECOA fair lending law). This wasn't possible here and would be a hard requirement, not optional polish.
- **Temporal validation, not just a random split** — this project uses an 80/20 random split, which assumes future applicants resemble past ones. A more rigorous approach trains on older data and tests on more recent data (out-of-time validation) to properly measure generalization as conditions drift — not possible here since the dataset lacks reliable timestamps.
- **Missing-data handling at inference time** — the API currently requires all fields; a production system needs a defined behavior for applicants who don't disclose income, not just a rejected request.
- **Model monitoring** — tracking prediction distributions over time in production to catch data drift before performance silently degrades.
- **Interpretability tradeoff** — XGBoost + SHAP explains predictions after the fact; a regulated lending environment might prefer an inherently transparent model (like a scorecard), even at some accuracy cost, for easier regulatory defense.
- **Dataset recency** — this data is from 2011; economic conditions and lending patterns have shifted since, and a real model would need retraining on current data.

## Tech stack

Python, Pandas, NumPy, Scikit-learn, imbalanced-learn (SMOTE), XGBoost, SHAP, FastAPI, Pydantic, Joblib

## Project structure

```
├── data/
│   ├── raw/            # original dataset (not committed - see Dataset section for source)
│   └── processed/
├── notebooks/
│   └── 01_loan_default_analysis.ipynb   # full EDA -> model -> evaluation pipeline
├── models/              # saved trained model, feature columns, threshold
├── app/
│   └── main.py           # FastAPI backend
└── requirements.txt
```