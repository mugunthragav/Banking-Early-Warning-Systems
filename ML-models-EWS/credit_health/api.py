from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

app = FastAPI()

# Load the trained models and scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
pd_model = pickle.load(open('models/pd_model_xgboost.pkl', 'rb'))
lgd_model = pickle.load(open('models/lgd_model_dt.pkl', 'rb'))
ead_model = pickle.load(open('models/ead_model_gbm.pkl', 'rb'))
pd_importance = pickle.load(open('models/pd_feature_importance.pkl', 'rb'))
lgd_importance = pickle.load(open('models/lgd_feature_importance.pkl', 'rb'))
ead_importance = pickle.load(open('models/ead_feature_importance.pkl', 'rb'))

# Define the expected features based on config.yaml
EXPECTED_FEATURES = [
    'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim', 'installment',
    'mths_since_last_delinq', 'mths_since_last_record', 'term_int', 'emp_length_int',
    'mths_since_issue_d', 'mths_since_earliest_cr_line', 'revol_bal', 'revol_util',
    'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
    'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
    'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical',
    'purpose_moving', 'purpose_other', 'purpose_small_business', 'purpose_vacation',
    'purpose_wedding',
    'initial_list_status_f', 'initial_list_status_w'
]


# Define the input model for the API
class LoanInput(BaseModel):
    loan_id: int
    funded_amnt: float
    int_rate: float
    grade: str
    dti: float
    home_ownership: str
    purpose: str
    initial_list_status: str
    term: str
    annual_inc: float
    emp_length: str
    delinq_2yrs: int
    inq_last_6mths: int
    open_acc: int
    pub_rec: int
    total_acc: int
    acc_now_delinq: int
    total_rev_hi_lim: int
    installment: float
    mths_since_last_delinq: int
    mths_since_last_record: int
    revol_bal: int
    revol_util: float
    issue_d: str  # Format: YYYY-MM-DD
    earliest_cr_line: str  # Format: YYYY-MM-DD


# Define the prediction endpoint
@app.post("/predict")
async def predict_loan_risk(loan: LoanInput):
    # Convert input to dictionary
    input_data = loan.dict()

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature engineering (match src/data_preprocessing.py exactly)
    # 1. Convert term to term_int
    input_df['term_int'] = input_df['term'].str.extract('(\d+)').astype(int)

    # 2. Convert emp_length to emp_length_int
    emp_length_mapping = {
        '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
        '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
    }
    input_df['emp_length_int'] = input_df['emp_length'].map(emp_length_mapping)

    # 3. Calculate months since issue_d and earliest_cr_line using the current date
    current_date = pd.to_datetime(datetime.today().date())  # Use today's date dynamically
    input_df['issue_d'] = pd.to_datetime(input_df['issue_d'])
    input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'])
    input_df['mths_since_issue_d'] = ((current_date - input_df['issue_d']) / pd.Timedelta(days=30)).astype(int)
    input_df['mths_since_earliest_cr_line'] = (
                (current_date - input_df['earliest_cr_line']) / pd.Timedelta(days=30)).astype(int)

    # 4. Handle missing values (match training)
    numerical_cols = [
        'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
        'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim', 'installment',
        'mths_since_last_delinq', 'mths_since_last_record', 'revol_bal', 'revol_util',
        'term_int', 'emp_length_int', 'mths_since_issue_d', 'mths_since_earliest_cr_line'
    ]
    median_values = {
        'funded_amnt': 15000, 'int_rate': 12.0, 'annual_inc': 60000, 'dti': 18.0,
        'delinq_2yrs': 0, 'inq_last_6mths': 0, 'open_acc': 10, 'pub_rec': 0,
        'total_acc': 20, 'acc_now_delinq': 0, 'total_rev_hi_lim': 30000, 'installment': 400,
        'mths_since_last_delinq': 30, 'mths_since_last_record': 60, 'term_int': 36,
        'emp_length_int': 5, 'mths_since_issue_d': 60, 'mths_since_earliest_cr_line': 120,
        'revol_bal': 10000, 'revol_util': 50.0
    }
    for col in numerical_cols:
        input_df[col] = input_df[col].fillna(median_values[col])

    # 5. Create dummy variables (match training exactly)
    categorical_cols = ['grade', 'home_ownership', 'purpose', 'initial_list_status']
    input_df = pd.get_dummies(input_df, columns=categorical_cols, dummy_na=False)

    # 6. Ensure all expected dummy variables are present
    for feature in EXPECTED_FEATURES:
        if feature not in input_df.columns and feature.startswith(
                ('grade_', 'home_ownership_', 'purpose_', 'initial_list_status_')):
            input_df[feature] = 0

    # 7. Select only the expected features
    input_df = input_df[EXPECTED_FEATURES]

    # 8. Scale the input data
    input_scaled = scaler.transform(input_df)

    # 9. Make predictions
    pd_pred = pd_model.predict_proba(input_scaled)[:, 1][0]  # Probability of default
    lgd_pred = lgd_model.predict(input_scaled)[0]  # Loss given default
    ead_pred = ead_model.predict(input_scaled)[0]  # Exposure at default (CCF)
    ead_pred_amount = ead_pred * loan.funded_amnt  # Convert CCF to amount

    # 10. Calculate expected loss
    expected_loss = pd_pred * lgd_pred * ead_pred_amount

    # 11. Confidence scores (approximated intervals)
    pd_conf = (max(0, pd_pred - 0.1), min(1, pd_pred + 0.1))
    lgd_conf = (max(0, lgd_pred - 0.05), min(1, lgd_pred + 0.05))
    ead_conf = (ead_pred_amount * 0.9, ead_pred_amount * 1.1)

    # 12. Decision factors (top 3 features)
    pd_top = sorted(pd_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    lgd_top = sorted(lgd_importance.items(), key=lambda x: x[1], reverse=True)[:3]
    ead_top = sorted(ead_importance.items(), key=lambda x: x[1], reverse=True)[:3]

    # 13. Save predictions to CSV
    predictions = pd.DataFrame({
        'loan_id': [loan.loan_id],
        'pd': [pd_pred],
        'lgd': [lgd_pred],
        'ead': [ead_pred_amount],
        'expected_loss': [expected_loss]
    })
    if not os.path.exists('results'):
        os.makedirs('results')
    predictions_file = 'results/loan_risk_predictions.csv'
    if os.path.exists(predictions_file):
        predictions.to_csv(predictions_file, mode='a', header=False, index=False)
    else:
        predictions.to_csv(predictions_file, mode='w', header=True, index=False)

    # Prepare response
    response = {
        "loan_id": loan.loan_id,
        "borrower_will_default": "Yes" if pd_pred > 0.5 else "No",
        "probability_of_default": {
            "value": float(pd_pred),
            "confidence": [float(pd_conf[0]), float(pd_conf[1])]
        },
        "potential_loss": {
            "percentage": float(lgd_pred),
            "amount": float(loan.funded_amnt * lgd_pred),
            "confidence": [float(lgd_conf[0]), float(lgd_conf[1])]
        },
        "exposure_at_default": {
            "amount": float(ead_pred_amount),
            "confidence": [float(ead_conf[0]), float(ead_conf[1])]
        },
        "expected_loss": float(expected_loss),
        "decision_factors": {
            "pd_factors": [{"feature": feature, "importance": float(importance)} for feature, importance in pd_top],
            "lgd_factors": [{"feature": feature, "importance": float(importance)} for feature, importance in lgd_top],
            "ead_factors": [{"feature": feature, "importance": float(importance)} for feature, importance in ead_top]
        },
        "predictions_saved_to": predictions_file
    }

    return response


# Endpoint to retrieve Basel III report
@app.get("/basel-report")
async def get_basel_report():
    basel_report_file = 'basel_report.txt'
    if not os.path.exists(basel_report_file):
        raise HTTPException(status_code=404, detail="Basel III report not found. Please generate the report first.")

    with open(basel_report_file, 'r') as f:
        basel_report_content = f.read()

    # Parse the report content into a structured response
    lines = basel_report_content.split('\n')
    report_data = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().replace(' ', '_').lower()
            try:
                # Clean and convert value to float
                value = value.replace('$', '').replace(',', '').strip()
                report_data[key] = float(value) if value.replace('.', '').replace('-', '').isdigit() else value
            except ValueError:
                report_data[key] = value

    return {
        "basel_iii_report": report_data
    }


# Endpoint to retrieve portfolio metrics
@app.get("/portfolio-metrics")
async def get_portfolio_metrics():
    predictions_file = 'data/loan_data_with_predictions.csv'
    if not os.path.exists(predictions_file):
        raise HTTPException(status_code=404,
                            detail="Portfolio predictions file not found. Please generate the Basel III report first.")

    portfolio_df = pd.read_csv(predictions_file)
    metrics = {
        "total_loans_analyzed": int(len(portfolio_df)),
        "average_probability_of_default": float(portfolio_df['PD_Probability'].mean()),
        "average_loss_given_default": float(portfolio_df['LGD_Prediction'].mean()),
        "average_exposure_at_default": float(portfolio_df['EAD_Prediction'].mean()),
        "total_expected_loss": float(portfolio_df['Expected_Loss'].sum()),
        "total_risk_weighted_assets": float(portfolio_df['RWA'].sum()),
        "capital_requirement": float(portfolio_df['RWA'].sum() * 0.08)
    }

    return {
        "portfolio_metrics": metrics
    }


# Endpoint to generate Basel III report
@app.post("/generate-basel-report")
async def generate_basel_report_endpoint():
    try:
        from generate_basel_report import generate_basel_report
        generate_basel_report()
        return {"status": "success", "message": "Basel III report generated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating Basel III report: {str(e)}")