import os
import joblib
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, '..', 'models')

model = joblib.load(os.path.join(MODELS_DIR, 'xgb_loan_default_model.pkl'))
feature_columns = joblib.load(os.path.join(MODELS_DIR, 'feature_columns.pkl'))
threshold = joblib.load(os.path.join(MODELS_DIR, 'chosen_threshold.pkl'))

class Applicant(BaseModel):
    age: int = Field(..., ge = 18, le = 110)
    monthly_income: float = Field(..., ge = 0)
    dependents: int = Field(..., ge=0, le=20)
    open_credit_lines: int = Field(..., ge=0)
    revolving_utilization: float = Field(..., ge=0, le=5)
    debt_ratio: float = Field(..., ge=0)
    real_estate_loans: int = Field(..., ge=0)
    total_times_past_due: int = Field(..., ge=0)

app = FastAPI(title="Loan Default Risk Prediction API")

explainer = shap.TreeExplainer(model)

def build_input_row(applicant: Applicant) -> pd.DataFrame:
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': applicant.revolving_utilization,
        'age': applicant.age,
        'DebtRatio': applicant.debt_ratio,
        'MonthlyIncome': applicant.monthly_income,
        'NumberOfOpenCreditLinesAndLoans': applicant.open_credit_lines,
        'NumberRealEstateLoansOrLines': applicant.real_estate_loans,
        'NumberOfDependents': applicant.dependents,
        'income_was_missing': 0,
        'TotalTimesPastDue': applicant.total_times_past_due,
    }
    return pd.DataFrame([input_dict])[feature_columns]

@app.post("/predict")
def predict(applicant: Applicant):
    input_df = build_input_row(applicant)

    risk_score = float(model.predict_proba(input_df)[:, 1][0])
    recommendation = "Reject / Flag for Review" if risk_score >= threshold else "Approve"

    shap_values = explainer(input_df)
    contributions = list(zip(feature_columns, shap_values.values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    top_factors = [
        {"feature": name, "impact": round(float(value), 4)}
        for name, value in contributions[:5]
    ]

    return {
        "risk_score": round(risk_score, 4),
        "recommendation": recommendation,
        "threshold_used": threshold,
        "top_factors": top_factors
    }