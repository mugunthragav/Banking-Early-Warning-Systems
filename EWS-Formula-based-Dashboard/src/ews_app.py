import dash
from dash import dcc, html, Input, Output
import pandas as pd
import numpy as np
import QuantLib as ql
import plotly.graph_objs as go
from flask import Flask, request, jsonify
import pickle
import os
from sklearn.preprocessing import LabelEncoder

# Ensure data and models directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set QuantLib evaluation date
ql.Settings.instance().evaluationDate = ql.Date(24, 6, 2025)

# Load datasets with error handling
try:
    credit_risk_df = pd.read_csv('data/credit_risk_dataset.csv')
    liquidity_risk_df = pd.read_csv('data/liquidity_risk_dataset.csv')
    market_risk_df = pd.read_csv('data/market_risk_dataset.csv')
    capital_compliance_df = pd.read_csv('data/capital_compliance_dataset.csv')
except FileNotFoundError as e:
    raise FileNotFoundError(f"Dataset not found: {e}. Ensure 'data/' contains all CSV files.")

# Load ML models
try:
    with open('models/ml_models.pkl', 'rb') as f:
        ml_models = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("ML models not found. Run 'train_ml_models.py' to generate 'models/ml_models.pkl'.")

# Initialize LabelEncoders for categorical columns
le_rate = LabelEncoder().fit(capital_compliance_df['Rate_Type'])
le_stress = LabelEncoder().fit(capital_compliance_df['Stress_Scenario_ID'])

# Initialize Flask app
flask_app = Flask(__name__)

# Initialize Dash app with Flask server
dash_app = dash.Dash(__name__, server=flask_app, routes_pathname_prefix='/')

# Formula-based calculations with interpretations
def calculate_npl_ratio(df):
    npl = df[df['Days_Past_Due'] > 90]['Principal_Amount'].sum()
    total_loans = df['Principal_Amount'].sum()
    value = (npl / total_loans * 100) if total_loans > 0 else 0
    status = "High Risk" if value > 5 else "Acceptable"
    return value, status, "NPL Ratio > 5% indicates high credit risk (Basel III)."

def calculate_pcr(df):
    provisions = df['Provision_Amount'].sum()
    npl = df[df['Days_Past_Due'] > 90]['Principal_Amount'].sum()
    value = (provisions / npl * 100) if npl > 0 else 0
    status = "Non-Compliant" if value < 70 else "Compliant"
    return value, status, "PCR ≥ 70% required for IFRS 9 compliance."

def calculate_ecl(df):
    pd = df['Credit_Score'].apply(lambda x: 0.05 if x < 600 else 0.01)
    lgd = (1 - df['Collateral_Value'] / df['Principal_Amount']).clip(0, 1)
    ead = df['Principal_Amount'] + df['Commitment_Amount']
    value = (pd * lgd * ead).sum()
    status = "High ECL" if value > df['Principal_Amount'].sum() * 0.1 else "Manageable"
    return value, status, "High ECL (>10% of principal) indicates significant credit loss risk."

def calculate_pd(df):
    value = df['Credit_Score'].apply(lambda x: 0.05 if x < 600 else 0.01).mean()
    status = "High Risk" if value > 0.03 else "Low Risk"
    return value, status, "PD > 3% signals elevated default risk (Basel III IRB)."

def calculate_lgd(df):
    value = (1 - df['Collateral_Value'] / df['Principal_Amount']).clip(0, 1).mean()
    status = "High Risk" if value > 0.5 else "Acceptable"
    return value, status, "LGD > 50% indicates low collateral coverage (Basel III)."

def calculate_ead(df):
    value = (df['Principal_Amount'] + df['Commitment_Amount']).sum()
    status = "High Exposure" if value > 1e7 else "Manageable"
    return value, status, "EAD > $10M indicates significant exposure (Basel III)."

def calculate_lcr(df):
    hqla = df[df['Asset_Type'].isin(['Cash', 'Government Bond'])]['Asset_Value'].sum()
    outflows = df['Outflow_Amount'].sum()
    value = (hqla / outflows * 100) if outflows > 0 else 0
    status = "Non-Compliant" if value < 100 else "Compliant"
    return value, status, "LCR ≥ 100% required for Basel III/CRR compliance."

def calculate_nsfr(df):
    asf = (df['Funding_Amount'] * df['ASF_Weight'] / 100).sum()
    rsf = (df['Asset_Value'] * df['RSF_Weight'] / 100).sum()
    value = (asf / rsf * 100) if rsf > 0 else 0
    status = "Non-Compliant" if value < 100 else "Compliant"
    return value, status, "NSFR ≥ 100% required for Basel III/CRR compliance."

def calculate_var(df):
    portfolio_value = (df['Quantity'] * df['Closing_Price']).sum()
    volatilities = df['Return_Volatility'].values
    weights = (df['Quantity'] * df['Closing_Price'] / portfolio_value).values
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.diag(volatilities), weights)))
    value = portfolio_value * portfolio_vol * np.sqrt(10) * 1.645
    status = "High Risk" if value > portfolio_value * 0.05 else "Acceptable"
    return value, status, "VaR > 5% of portfolio value signals high market risk (FRTB)."

def calculate_es(df):
    portfolio_value = (df['Quantity'] * df['Closing_Price']).sum()
    volatilities = df['Return_Volatility'].values
    weights = (df['Quantity'] * df['Closing_Price'] / portfolio_value).values
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.diag(volatilities), weights)))
    value = portfolio_value * portfolio_vol * np.sqrt(10) * 2.241
    status = "High Risk" if value > portfolio_value * 0.075 else "Acceptable"
    return value, status, "ES > 7.5% of portfolio value signals extreme market risk (FRTB)."

def calculate_tier1_ratio(df):
    tier1 = df[df['Capital_ID'].str.contains('Equity|Reserves', case=False, na=False)]['Capital_Amount'].sum() - df['Deduction_Amount'].sum()
    rwa = (df['Exposure_Amount'] * df['Risk_Weight'] / 100).sum()
    value = (tier1 / rwa * 100) if rwa > 0 else 0
    status = "Non-Compliant" if value < 6 else "Compliant"
    return value, status, "Tier 1 Ratio ≥ 6% required for Basel III/CRR."

def calculate_cet1_ratio(df):
    cet1 = df[df['Capital_ID'].str.contains('Common Equity', case=False, na=False)]['Capital_Amount'].sum() - df['Deduction_Amount'].sum()
    rwa = (df['Exposure_Amount'] * df['Risk_Weight'] / 100).sum()
    value = (cet1 / rwa * 100) if rwa > 0 else 0
    status = "Non-Compliant" if value < 4.5 else "Compliant"
    return value, status, "CET1 Ratio ≥ 4.5% required for Basel III/TLAC."

def calculate_aml_compliance(df):
    valid = df[df['AML_Flag'] == 'Pass']['Transaction_Amount'].count()
    total = df['Transaction_Amount'].count()
    value = (valid / total * 100) if total > 0 else 0
    status = "Non-Compliant" if value < 95 else "Compliant"
    return value, status, "AML Compliance > 95% required for FATF/AML Directive."

def calculate_libor_exposure(df):
    libor = df[df['Rate_Type'] == 'LIBOR']['Contract_ID'].count()
    total = df['Contract_ID'].count()
    value = (libor / total * 100) if total > 0 else 0
    status = "Non-Compliant" if value > 0 else "Compliant"
    return value, status, "LIBOR Exposure should be 0% post-Jun 2023 (FCA)."

def calculate_scb(df):
    value = df[df['Stress_Scenario_ID'] == 'S1']['Loss_Amount'].sum()
    status = "High Stress Impact" if value > 1e6 else "Manageable"
    return value, status, "SCB > $1M indicates significant stress losses (CCAR)."

def calculate_ccar_readiness(df):
    cr_score = calculate_npl_ratio(credit_risk_df)[0]
    lr_score = calculate_lcr(liquidity_risk_df)[0]
    value = (100 - cr_score + lr_score) / 2
    status = "Not Ready" if value < 80 else "Ready"
    return value, status, "CCAR Readiness ≥ 80% for Apr 2026 compliance."

def calculate_basel_readiness(df):
    cr_score = calculate_npl_ratio(credit_risk_df)[0]
    lr_score = calculate_lcr(liquidity_risk_df)[0]
    mr_score = calculate_var(market_risk_df)[0]
    value = (100 - cr_score + lr_score + 100 - mr_score) / 3
    status = "Not Ready" if value < 80 else "Ready"
    return value, status, "Basel III Readiness ≥ 80% for Jul 2025 compliance."

def calculate_compliance_score(df):
    cr_score = calculate_npl_ratio(credit_risk_df)[0]
    lr_score = calculate_lcr(liquidity_risk_df)[0]
    mr_score = calculate_var(market_risk_df)[0]
    value = 0.4 * (100 - cr_score) + 0.3 * lr_score + 0.3 * (100 - mr_score)
    status = "Non-Compliant" if value < 80 else "Compliant"
    return value, status, "Compliance Score > 80% for Basel III/CRR/Dodd-Frank."

def calculate_operational_rwa(df):
    bi = df['Revenue_Amount'].sum() * 0.12
    ilm = df['Loss_Amount'].count() / 1000
    value = bi * ilm
    status = "High Risk" if value > 1e6 else "Acceptable"
    return value, status, "Operational RWA > $1M indicates high operational risk (Basel III AMA)."

def calculate_composite_risk_index(df_cr, df_lr, df_mr):
    cr_metric = calculate_npl_ratio(df_cr)[0]
    lr_metric = calculate_lcr(df_lr)[0]
    mr_metric = calculate_var(df_mr)[0]
    value = (cr_metric + (100 - lr_metric) + mr_metric) / 3
    status = "High Risk" if value > 0.7 else "Acceptable"
    return value, status, "Composite Risk Index > 0.7 indicates elevated overall risk."

# Flask API endpoints for calculations
@flask_app.route('/api/npl_ratio', methods=['POST'])
def npl_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_npl_ratio(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/pcr', methods=['POST'])
def pcr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_pcr(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/ecl', methods=['POST'])
def ecl():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_ecl(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/cr_metrics', methods=['POST'])
def cr_metrics():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        pd_val, pd_status, pd_interp = calculate_pd(df)
        lgd_val, lgd_status, lgd_interp = calculate_lgd(df)
        ead_val, ead_status, ead_interp = calculate_ead(df)
        return jsonify({
            "pd": {"value": pd_val, "status": pd_status, "interpretation": pd_interp},
            "lgd": {"value": lgd_val, "status": lgd_status, "interpretation": lgd_interp},
            "ead": {"value": ead_val, "status": ead_status, "interpretation": ead_interp}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/ifrs9_cr', methods=['POST'])
def ifrs9_cr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_ecl(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/irb', methods=['POST'])
def irb():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        pd_val, _, _ = calculate_pd(df)
        lgd_val, _, _ = calculate_lgd(df)
        ead_val, _, _ = calculate_ead(df)
        value = pd_val * lgd_val * ead_val * 12.5
        status = "High Risk" if value > 1e6 else "Acceptable"
        return jsonify({"value": value, "status": status, "interpretation": "IRB Risk Weight > $1M indicates high risk (Basel III)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/lcr', methods=['POST'])
def lcr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_lcr(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/nsfr', methods=['POST'])
def nsfr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_nsfr(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/lr_metrics', methods=['POST'])
def lr_metrics():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        lcr_val, lcr_status, lcr_interp = calculate_lcr(df)
        nsfr_val, nsfr_status, nsfr_interp = calculate_nsfr(df)
        return jsonify({
            "lcr": {"value": lcr_val, "status": lcr_status, "interpretation": lcr_interp},
            "nsfr": {"value": nsfr_val, "status": nsfr_status, "interpretation": nsfr_interp}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/var', methods=['POST'])
def var():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_var(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/es', methods=['POST'])
def es():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_es(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/mr_metrics', methods=['POST'])
def mr_metrics():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        var_val, var_status, var_interp = calculate_var(df)
        es_val, es_status, es_interp = calculate_es(df)
        return jsonify({
            "var": {"value": var_val, "status": var_status, "interpretation": var_interp},
            "es": {"value": es_val, "status": es_status, "interpretation": es_interp}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/tier1_ratio', methods=['POST'])
def tier1_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_tier1_ratio(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/cet1_ratio', methods=['POST'])
def cet1_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_cet1_ratio(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/aml_compliance', methods=['POST'])
def aml_compliance():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_aml_compliance(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/libor_exposure', methods=['POST'])
def libor_exposure():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_libor_exposure(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/scb', methods=['POST'])
def scb():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_scb(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/ccar_readiness', methods=['POST'])
def ccar_readiness():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_ccar_readiness(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/basel_readiness', methods=['POST'])
def basel_readiness():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_basel_readiness(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/compliance_score', methods=['POST'])
def compliance_score():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_compliance_score(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/operational_rwa', methods=['POST'])
def operational_rwa():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_operational_rwa(df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/composite_risk', methods=['POST'])
def composite_risk():
    try:
        data = request.get_json()
        df_cr = pd.DataFrame(data['data'])
        value, status, interpretation = calculate_composite_risk_index(df_cr, liquidity_risk_df, market_risk_df)
        return jsonify({"value": value, "status": status, "interpretation": interpretation})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Flask API endpoints for ML predictions
@flask_app.route('/api/predict/npl_ratio', methods=['POST'])
def predict_npl_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Principal_Amount', 'Days_Past_Due', 'Credit_Score']]
        prediction = ml_models['npl_ratio'].predict(X)[0]
        status = "High Risk" if prediction > 5 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted NPL Ratio > 5% indicates high credit risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/pcr', methods=['POST'])
def predict_pcr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Provision_Amount', 'Principal_Amount', 'Days_Past_Due']]
        prediction = ml_models['pcr'].predict(X)[0]
        status = "Non-Compliant" if prediction < 70 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted PCR ≥ 70% required for IFRS 9."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/ecl', methods=['POST'])
def predict_ecl():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Credit_Score', 'Collateral_Value', 'Principal_Amount', 'Commitment_Amount']]
        prediction = ml_models['ecl'].predict(X)[0]
        status = "High ECL" if prediction > df['Principal_Amount'].sum() * 0.1 else "Manageable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted ECL > 10% of principal indicates significant loss risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/pd', methods=['POST'])
def predict_pd():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Credit_Score', 'Days_Past_Due']]
        prediction = ml_models['pd'].predict(X)[0]
        status = "High Risk" if prediction > 0.03 else "Low Risk"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted PD > 3% signals elevated default risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/lgd', methods=['POST'])
def predict_lgd():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Collateral_Value', 'Principal_Amount']]
        prediction = ml_models['lgd'].predict(X)[0]
        status = "High Risk" if prediction > 0.5 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted LGD > 50% indicates low collateral coverage."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/ead', methods=['POST'])
def predict_ead():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Principal_Amount', 'Commitment_Amount']]
        prediction = ml_models['ead'].predict(X)[0]
        status = "High Exposure" if prediction > 1e7 else "Manageable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted EAD > $10M indicates significant exposure."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/ifrs9_stage', methods=['POST'])
def predict_ifrs9_stage():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Credit_Score', 'Days_Past_Due']]
        prediction = ml_models['ifrs9_stage'].predict(X)[0]
        status = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}.get(int(prediction), "Unknown")
        return jsonify({"prediction": int(prediction), "status": status, "interpretation": f"Predicted IFRS 9 Stage: {status}."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/lcr', methods=['POST'])
def predict_lcr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Asset_Value', 'Outflow_Amount', 'ASF_Weight']]
        prediction = ml_models['lcr'].predict(X)[0]
        status = "Non-Compliant" if prediction < 100 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted LCR ≥ 100% required for Basel III."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/nsfr', methods=['POST'])
def predict_nsfr():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Funding_Amount', 'Asset_Value', 'ASF_Weight', 'RSF_Weight']]
        prediction = ml_models['nsfr'].predict(X)[0]
        status = "Non-Compliant" if prediction < 100 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted NSFR ≥ 100% required for Basel III."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/var', methods=['POST'])
def predict_var():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Quantity', 'Closing_Price', 'Return_Volatility']]
        prediction = ml_models['var'].predict(X)[0]
        status = "High Risk" if prediction > df['Quantity'].sum() * df['Closing_Price'].mean() * 0.05 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted VaR > 5% of portfolio value signals high market risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/es', methods=['POST'])
def predict_es():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Quantity', 'Closing_Price', 'Return_Volatility']]
        prediction = ml_models['es'].predict(X)[0]
        status = "High Risk" if prediction > df['Quantity'].sum() * df['Closing_Price'].mean() * 0.075 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted ES > 7.5% of portfolio value signals extreme market risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/tier1_ratio', methods=['POST'])
def predict_tier1_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']]
        prediction = ml_models['tier1_ratio'].predict(X)[0]
        status = "Non-Compliant" if prediction < 6 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted Tier 1 Ratio ≥ 6% required for Basel III."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/cet1_ratio', methods=['POST'])
def predict_cet1_ratio():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']]
        prediction = ml_models['cet1_ratio'].predict(X)[0]
        status = "Non-Compliant" if prediction < 4.5 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted CET1 Ratio ≥ 4.5% required for Basel III."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/aml_compliance', methods=['POST'])
def predict_aml_compliance():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Transaction_Amount', 'Risk_Score']]
        prediction = ml_models['aml_compliance'].predict(X)[0]
        status = "Pass" if prediction == 'Pass' else "Fail"
        return jsonify({"prediction": str(prediction), "status": status, "interpretation": "Predicted AML Compliance: Pass indicates low risk (FATF)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/libor_exposure', methods=['POST'])
def predict_libor_exposure():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        df['Rate_Type_Encoded'] = le_rate.transform(df['Rate_Type'])
        X = df[['Transaction_Amount', 'Rate_Type_Encoded']]
        prediction = ml_models['libor_exposure'].predict(X)[0]
        status = "Non-Compliant" if prediction == 1 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted LIBOR Exposure: 0 required post-Jun 2023 (FCA)."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/scb', methods=['POST'])
def predict_scb():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        df['Stress_Scenario_ID_Encoded'] = le_stress.transform(df['Stress_Scenario_ID'])
        X = df[['Capital_Amount', 'Loss_Amount', 'Stress_Scenario_ID_Encoded']]
        prediction = ml_models['scb'].predict(X)[0]
        status = "High Stress Impact" if prediction > 1e6 else "Manageable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted SCB > $1M indicates significant stress losses."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/ccar_readiness', methods=['POST'])
def predict_ccar_readiness():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
        prediction = ml_models['ccar_readiness'].predict(X)[0]
        status = "Not Ready" if prediction < 80 else "Ready"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted CCAR Readiness ≥ 80% for Apr 2026."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/basel_readiness', methods=['POST'])
def predict_basel_readiness():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
        prediction = ml_models['basel_readiness'].predict(X)[0]
        status = "Not Ready" if prediction < 80 else "Ready"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted Basel III Readiness ≥ 80% for Jul 2025."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/compliance_score', methods=['POST'])
def predict_compliance_score():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
        prediction = ml_models['compliance_score'].predict(X)[0]
        status = "Non-Compliant" if prediction < 80 else "Compliant"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted Compliance Score > 80% for Basel III/CRR."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/op_rwa', methods=['POST'])
def predict_op_rwa():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Revenue_Amount', 'Loss_Amount']]
        prediction = ml_models['op_rwa'].predict(X)[0]
        status = "High Risk" if prediction > 1e6 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted Operational RWA > $1M indicates high risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@flask_app.route('/api/predict/composite_risk', methods=['POST'])
def predict_composite_risk():
    try:
        data = request.get_json()
        df = pd.DataFrame(data['data'])
        X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
        prediction = ml_models['composite_risk'].predict(X)[0]
        status = "High Risk" if prediction > 0.7 else "Acceptable"
        return jsonify({"prediction": float(prediction), "status": status, "interpretation": "Predicted Composite Risk Index > 0.7 indicates elevated risk."})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Dash layout with enhanced dashboard and AI analytics
dash_app.layout = html.Div(className='container mx-auto p-6 bg-gray-100 min-h-screen', children=[
    html.Link(href='https://cdn.tailwindcss.com', rel='stylesheet'),
    html.H1('Early Warning System (EWS)', className='text-4xl font-extrabold text-blue-800 mb-6 text-center'),
    dcc.Tabs([
        dcc.Tab(label='Dashboards', className='text-lg font-semibold', children=[
            html.H2('Risk Dashboards', className='text-2xl font-bold mt-4 mb-4 text-blue-700'),
            html.Div([
                html.H3('Credit Risk', className='text-xl font-semibold mb-2 text-gray-800'),
                html.Div(className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', children=[
                    html.Div([
                        html.H4(f'{comp["name"]} ({comp["standard"]})', className='text-lg font-medium text-gray-900'),
                        html.P(f'Formula: {comp["formula"]}', className='text-sm text-gray-600'),
                        dcc.Graph(id=f'{comp["id"]}-graph', style={'height': '300px'}),
                        html.P(id=f'{comp["id"]}-value', className='text-sm mt-2'),
                        html.P(id=f'{comp["id"]}-status', className='text-sm font-semibold'),
                        html.P(id=f'{comp["id"]}-interpretation', className='text-sm text-gray-600')
                    ], className='p-4 bg-white rounded-lg shadow-md hover:shadow-lg') for comp in [
                        {"name": "NPL Ratio", "standard": "Basel III IRB", "formula": "(Non-Performing Loans / Total Loans) * 100", "id": "npl_ratio"},
                        {"name": "Provision Coverage Ratio", "standard": "IFRS 9", "formula": "(Loan Loss Provisions / NPL) * 100", "id": "pcr"},
                        {"name": "Expected Credit Loss", "standard": "IFRS 9", "formula": "PD * LGD * EAD", "id": "ecl"},
                        {"name": "Credit Risk Metrics", "standard": "Basel III IRB", "formula": "PD, LGD, EAD", "id": "cr_metrics"},
                        {"name": "IRB Risk Weights", "standard": "Basel III IRB", "formula": "f(PD, LGD, EAD)", "id": "irb"},
                        {"name": "IFRS 9 Credit Risk", "standard": "IFRS 9", "formula": "PD * LGD * EAD", "id": "ifrs9_cr"}
                    ]
                ]),
                html.H3('Liquidity Risk', className='text-xl font-semibold mt-6 mb-2 text-gray-800'),
                html.Div(className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', children=[
                    html.Div([
                        html.H4(f'{comp["name"]} ({comp["standard"]})', className='text-lg font-medium text-gray-900'),
                        html.P(f'Formula: {comp["formula"]}', className='text-sm text-gray-600'),
                        dcc.Graph(id=f'{comp["id"]}-graph', style={'height': '300px'}),
                        html.P(id=f'{comp["id"]}-value', className='text-sm mt-2'),
                        html.P(id=f'{comp["id"]}-status', className='text-sm font-semibold'),
                        html.P(id=f'{comp["id"]}-interpretation', className='text-sm text-gray-600')
                    ], className='p-4 bg-white rounded-lg shadow-md hover:shadow-lg') for comp in [
                        {"name": "Liquidity Coverage Ratio", "standard": "Basel III, CRR", "formula": "(HQLA / Net Cash Outflows) * 100", "id": "lcr"},
                        {"name": "Net Stable Funding Ratio", "standard": "Basel III, CRR", "formula": "(ASF / RSF) * 100", "id": "nsfr"},
                        {"name": "Liquidity Risk Metrics", "standard": "Basel III, CRR", "formula": "LCR, NSFR", "id": "lr_metrics"}
                    ]
                ]),
                html.H3('Market Risk', className='text-xl font-semibold mt-6 mb-2 text-gray-800'),
                html.Div(className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', children=[
                    html.Div([
                        html.H4(f'{comp["name"]} ({comp["standard"]})', className='text-lg font-medium text-gray-900'),
                        html.P(f'Formula: {comp["formula"]}', className='text-sm text-gray-600'),
                        dcc.Graph(id=f'{comp["id"]}-graph', style={'height': '300px'}),
                        html.P(id=f'{comp["id"]}-value', className='text-sm mt-2'),
                        html.P(id=f'{comp["id"]}-status', className='text-sm font-semibold'),
                        html.P(id=f'{comp["id"]}-interpretation', className='text-sm text-gray-600')
                    ], className='p-4 bg-white rounded-lg shadow-md hover:shadow-lg') for comp in [
                        {"name": "Value at Risk", "standard": "FRTB, Basel 3.1", "formula": "Portfolio Loss at 95% Confidence", "id": "var"},
                        {"name": "Expected Shortfall", "standard": "FRTB, Basel 3.1", "formula": "Average Loss Beyond 97.5% Confidence", "id": "es"},
                        {"name": "Market Risk Metrics", "standard": "FRTB, Basel 3.1", "formula": "VaR, ES", "id": "mr_metrics"}
                    ]
                ]),
                html.H3('Capital & Compliance', className='text-xl font-semibold mt-6 mb-2 text-gray-800'),
                html.Div(className='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6', children=[
                    html.Div([
                        html.H4(f'{comp["name"]} ({comp["standard"]})', className='text-lg font-medium text-gray-900'),
                        html.P(f'Formula: {comp["formula"]}', className='text-sm text-gray-600'),
                        dcc.Graph(id=f'{comp["id"]}-graph', style={'height': '300px'}),
                        html.P(id=f'{comp["id"]}-value', className='text-sm mt-2'),
                        html.P(id=f'{comp["id"]}-status', className='text-sm font-semibold'),
                        html.P(id=f'{comp["id"]}-interpretation', className='text-sm text-gray-600')
                    ], className='p-4 bg-white rounded-lg shadow-md hover:shadow-lg') for comp in [
                        {"name": "Tier 1 Capital Ratio", "standard": "Basel III, CRR", "formula": "(Tier 1 Capital / RWA) * 100", "id": "tier1_ratio"},
                        {"name": "CET1 Ratio", "standard": "Basel III, TLAC", "formula": "(Common Equity Tier 1 / RWA) * 100", "id": "cet1_ratio"},
                        {"name": "Stress Capital Buffer", "standard": "CCAR, Dodd-Frank", "formula": "Max Loss under CCAR Scenarios", "id": "scb"},
                        {"name": "AML/KYC Compliance", "standard": "FATF, AML Directive", "formula": "(Valid Transactions / Total Transactions) * 100", "id": "aml_compliance"},
                        {"name": "LIBOR Transition", "standard": "FCA", "formula": "(LIBOR Contracts / Total Contracts) * 100", "id": "libor_exposure"},
                        {"name": "CCAR Readiness", "standard": "CCAR, Dodd-Frank", "formula": "(CR + LR Compliance) / 2", "id": "ccar_readiness"},
                        {"name": "Basel III Readiness", "standard": "Basel III", "formula": "(CR + LR + MR Compliance) / 3", "id": "basel_readiness"},
                        {"name": "Compliance Score", "standard": "Basel III, CRR, Dodd-Frank", "formula": "0.4 * CR_Score + 0.3 * LR_Score + 0.3 * MR_Score", "id": "compliance_score"},
                        {"name": "Operational RWA", "standard": "Basel III AMA", "formula": "BI * ILM", "id": "op_rwa"},
                        {"name": "Composite Risk Index", "standard": "Internal", "formula": "(CR + LR + MR) / 3", "id": "composite_risk"},
                        {"name": "NPL Dashboard", "standard": "Basel III", "formula": "(NPL / Total Loans) * 100", "id": "npl_dash"},
                        {"name": "LCR Dashboard", "standard": "Basel III", "formula": "(HQLA / Net Cash Outflows) * 100", "id": "lcr_dash"},
                        {"name": "NSFR Dashboard", "standard": "Basel III", "formula": "(ASF / RSF) * 100", "id": "nsfr_dash"},
                        {"name": "VaR Dashboard", "standard": "FRTB, Basel 3.1", "formula": "Portfolio Loss at 95% Confidence", "id": "var_dash"},
                        {"name": "Tier 1 Capital Ratio Dashboard", "standard": "Basel III, CRR", "formula": "(Tier 1 Capital / RWA) * 100", "id": "tier1_dash"},
                        {"name": "IFRS 9 Analysis", "standard": "IFRS 9", "formula": "PD * LGD * EAD", "id": "ifrs9_analysis"}
                    ]
                ])
            ])
        ]),
        dcc.Tab(label='AI Analytics', className='text-lg font-semibold', children=[
            html.H2('AI-Powered Risk Predictions', className='text-2xl font-bold mt-4 mb-4 text-blue-700'),
            html.Div([
                html.H3('Test ML Model Predictions', className='text-xl font-semibold mb-4 text-gray-800'),
                html.Div(className='grid grid-cols-1 md:grid-cols-2 gap-6', children=[
                    html.Div([
                        html.H4(f'{model["name"]}', className='text-lg font-medium text-gray-900'),
                        html.P(f'{model["description"]}', className='text-sm text-gray-600'),
                        dcc.Textarea(id=f'{model["id"]}-input', placeholder='Enter JSON data (e.g., [{"column1": value, ...}])', className='w-full p-2 border rounded mb-2'),
                        html.Button('Predict', id=f'{model["id"]}-button', className='bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700'),
                        html.P(id=f'{model["id"]}-prediction', className='text-sm mt-2'),
                        html.P(id=f'{model["id"]}-prediction-status', className='text-sm font-semibold')
                    ], className='p-4 bg-white rounded-lg shadow-md hover:shadow-lg') for model in [
                        {"name": "NPL Ratio Prediction", "description": "Predict Non-Performing Loans Ratio", "id": "npl_ratio"},
                        {"name": "PCR Prediction", "description": "Predict Provision Coverage Ratio", "id": "pcr"},
                        {"name": "ECL Prediction", "description": "Predict Expected Credit Loss", "id": "ecl"},
                        {"name": "PD Prediction", "description": "Predict Probability of Default", "id": "pd"},
                        {"name": "LGD Prediction", "description": "Predict Loss Given Default", "id": "lgd"},
                        {"name": "EAD Prediction", "description": "Predict Exposure at Default", "id": "ead"},
                        {"name": "IFRS 9 Staging", "description": "Classify loans into IFRS 9 stages", "id": "ifrs9_stage"},
                        {"name": "LCR Forecast", "description": "Forecast Liquidity Coverage Ratio", "id": "lcr"},
                        {"name": "NSFR Forecast", "description": "Forecast Net Stable Funding Ratio", "id": "nsfr"},
                        {"name": "VaR Estimation", "description": "Estimate Value at Risk", "id": "var"},
                        {"name": "ES Estimation", "description": "Estimate Expected Shortfall", "id": "es"},
                        {"name": "Tier 1 Ratio Prediction", "description": "Predict Tier 1 Capital Ratio", "id": "tier1_ratio"},
                        {"name": "CET1 Ratio Prediction", "description": "Predict Common Equity Tier 1 Ratio", "id": "cet1_ratio"},
                        {"name": "AML Detection", "description": "Detect AML compliance issues", "id": "aml_compliance"},
                        {"name": "LIBOR Exposure", "description": "Monitor LIBOR exposure risks", "id": "libor_exposure"},
                        {"name": "SCB Prediction", "description": "Predict Stress Capital Buffer", "id": "scb"},
                        {"name": "CCAR Readiness", "description": "Predict CCAR compliance readiness", "id": "ccar_readiness"},
                        {"name": "Basel III Readiness", "description": "Predict Basel III compliance readiness", "id": "basel_readiness"},
                        {"name": "Compliance Score", "description": "Predict overall compliance score", "id": "compliance_score"},
                        {"name": "Operational RWA", "description": "Predict Operational Risk-Weighted Assets", "id": "op_rwa"},
                        {"name": "Composite Risk Index", "description": "Predict composite risk index", "id": "composite_risk"}
                    ]
                ])
            ])
        ])
    ])
])

# Callbacks for dashboards with gauge charts and color-coded status
def create_gauge_chart(value, max_value, title, status, status_colors={'Compliant': 'green', 'Non-Compliant': 'red', 'High Risk': 'red', 'Acceptable': 'green', 'High Exposure': 'red', 'Manageable': 'green', 'Not Ready': 'red', 'Ready': 'green', 'High Stress Impact': 'red', 'Low Risk': 'green', 'Stage 1': 'green', 'Stage 2': 'orange', 'Stage 3': 'red', 'Pass': 'green', 'Fail': 'red'}):
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': status_colors.get(status, 'gray')},
            'steps': [
                {'range': [0, max_value * 0.5], 'color': 'lightgreen'},
                {'range': [max_value * 0.5, max_value * 0.75], 'color': 'yellow'},
                {'range': [max_value * 0.75, max_value], 'color': 'red'}
            ],
            'threshold': {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': value}
        }
    ))

@dash_app.callback(
    [Output('npl_ratio-graph', 'figure'), Output('npl_ratio-value', 'children'), Output('npl_ratio-status', 'children'), Output('npl_ratio-interpretation', 'children')],
    Input('npl_ratio-graph', 'id')
)
def update_npl_ratio(_):
    value, status, interp = calculate_npl_ratio(credit_risk_df)
    fig = create_gauge_chart(value, 100, 'NPL Ratio (%)', status)
    return fig, f'NPL Ratio: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('pcr-graph', 'figure'), Output('pcr-value', 'children'), Output('pcr-status', 'children'), Output('pcr-interpretation', 'children')],
    Input('pcr-graph', 'id')
)
def update_pcr(_):
    value, status, interp = calculate_pcr(credit_risk_df)
    fig = create_gauge_chart(value, 200, 'PCR (%)', status)
    return fig, f'PCR: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('ecl-graph', 'figure'), Output('ecl-value', 'children'), Output('ecl-status', 'children'), Output('ecl-interpretation', 'children')],
    Input('ecl-graph', 'id')
)
def update_ecl(_):
    value, status, interp = calculate_ecl(credit_risk_df)
    fig = create_gauge_chart(value, credit_risk_df['Principal_Amount'].sum() * 0.5, 'ECL ($)', status)
    return fig, f'ECL: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('cr_metrics-graph', 'figure'), Output('cr_metrics-value', 'children'), Output('cr_metrics-status', 'children'), Output('cr_metrics-interpretation', 'children')],
    Input('cr_metrics-graph', 'id')
)
def update_cr_metrics(_):
    pd_val, pd_status, pd_interp = calculate_pd(credit_risk_df)
    lgd_val, lgd_status, lgd_interp = calculate_lgd(credit_risk_df)
    ead_val, ead_status, ead_interp = calculate_ead(credit_risk_df)
    fig = go.Figure(data=[
        go.Bar(x=['PD', 'LGD', 'EAD'], y=[pd_val * 100, lgd_val * 100, ead_val / 1e6], marker_color=['blue', 'orange', 'green'])
    ], layout={'title': 'Credit Risk Metrics', 'yaxis': {'title': 'Value (% or $M)'}})
    return fig, f'PD: {pd_val:.2%}, LGD: {lgd_val:.2%}, EAD: ${ead_val:.2f}', f'Status: PD-{pd_status}, LGD-{lgd_status}, EAD-{ead_status}', f'{pd_interp} {lgd_interp} {ead_interp}'

@dash_app.callback(
    [Output('ifrs9_cr-graph', 'figure'), Output('ifrs9_cr-value', 'children'), Output('ifrs9_cr-status', 'children'), Output('ifrs9_cr-interpretation', 'children')],
    Input('ifrs9_cr-graph', 'id')
)
def update_ifrs9_cr(_):
    value, status, interp = calculate_ecl(credit_risk_df)
    fig = create_gauge_chart(value, credit_risk_df['Principal_Amount'].sum() * 0.5, 'ECL ($)', status)
    return fig, f'ECL: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('irb-graph', 'figure'), Output('irb-value', 'children'), Output('irb-status', 'children'), Output('irb-interpretation', 'children')],
    Input('irb-graph', 'id')
)
def update_irb(_):
    pd_val, _, _ = calculate_pd(credit_risk_df)
    lgd_val, _, _ = calculate_lgd(credit_risk_df)
    ead_val, _, _ = calculate_ead(credit_risk_df)
    value = pd_val * lgd_val * ead_val * 12.5
    status = "High Risk" if value > 1e6 else "Acceptable"
    fig = create_gauge_chart(value, 5e6, 'IRB Risk Weight ($)', status)
    return fig, f'IRB Risk Weight: ${value:.2f}', f'Status: {status}', "IRB Risk Weight > $1M indicates high risk (Basel III)."

@dash_app.callback(
    [Output('lcr-graph', 'figure'), Output('lcr-value', 'children'), Output('lcr-status', 'children'), Output('lcr-interpretation', 'children')],
    Input('lcr-graph', 'id')
)
def update_lcr(_):
    value, status, interp = calculate_lcr(liquidity_risk_df)
    fig = create_gauge_chart(value, 200, 'LCR (%)', status)
    return fig, f'LCR: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('nsfr-graph', 'figure'), Output('nsfr-value', 'children'), Output('nsfr-status', 'children'), Output('nsfr-interpretation', 'children')],
    Input('nsfr-graph', 'id')
)
def update_nsfr(_):
    value, status, interp = calculate_nsfr(liquidity_risk_df)
    fig = create_gauge_chart(value, 200, 'NSFR (%)', status)
    return fig, f'NSFR: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('lr_metrics-graph', 'figure'), Output('lr_metrics-value', 'children'), Output('lr_metrics-status', 'children'), Output('lr_metrics-interpretation', 'children')],
    Input('lr_metrics-graph', 'id')
)
def update_lr_metrics(_):
    lcr_val, lcr_status, lcr_interp = calculate_lcr(liquidity_risk_df)
    nsfr_val, nsfr_status, nsfr_interp = calculate_nsfr(liquidity_risk_df)
    fig = go.Figure(data=[
        go.Bar(x=['LCR', 'NSFR'], y=[lcr_val, nsfr_val], marker_color=['blue', 'orange'])
    ], layout={'title': 'Liquidity Risk Metrics', 'yaxis': {'title': 'Value (%)'}})
    return fig, f'LCR: {lcr_val:.2f}%, NSFR: {nsfr_val:.2f}%', f'Status: LCR-{lcr_status}, NSFR-{nsfr_status}', f'{lcr_interp} {nsfr_interp}'

@dash_app.callback(
    [Output('var-graph', 'figure'), Output('var-value', 'children'), Output('var-status', 'children'), Output('var-interpretation', 'children')],
    Input('var-graph', 'id')
)
def update_var(_):
    value, status, interp = calculate_var(market_risk_df)
    fig = create_gauge_chart(value, market_risk_df['Quantity'].sum() * market_risk_df['Closing_Price'].mean() * 0.2, 'VaR ($)', status)
    return fig, f'VaR: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('es-graph', 'figure'), Output('es-value', 'children'), Output('es-status', 'children'), Output('es-interpretation', 'children')],
    Input('es-graph', 'id')
)
def update_es(_):
    value, status, interp = calculate_es(market_risk_df)
    fig = create_gauge_chart(value, market_risk_df['Quantity'].sum() * market_risk_df['Closing_Price'].mean() * 0.3, 'ES ($)', status)
    return fig, f'ES: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('mr_metrics-graph', 'figure'), Output('mr_metrics-value', 'children'), Output('mr_metrics-status', 'children'), Output('mr_metrics-interpretation', 'children')],
    Input('mr_metrics-graph', 'id')
)
def update_mr_metrics(_):
    var_val, var_status, var_interp = calculate_var(market_risk_df)
    es_val, es_status, es_interp = calculate_es(market_risk_df)
    fig = go.Figure(data=[
        go.Bar(x=['VaR', 'ES'], y=[var_val, es_val], marker_color=['blue', 'orange'])
    ], layout={'title': 'Market Risk Metrics', 'yaxis': {'title': 'Value ($)'}})
    return fig, f'VaR: ${var_val:.2f}, ES: ${es_val:.2f}', f'Status: VaR-{var_status}, ES-{es_status}', f'{var_interp} {es_interp}'

@dash_app.callback(
    [Output('tier1_ratio-graph', 'figure'), Output('tier1_ratio-value', 'children'), Output('tier1_ratio-status', 'children'), Output('tier1_ratio-interpretation', 'children')],
    Input('tier1_ratio-graph', 'id')
)
def update_tier1_ratio(_):
    value, status, interp = calculate_tier1_ratio(capital_compliance_df)
    fig = create_gauge_chart(value, 20, 'Tier 1 Ratio (%)', status)
    return fig, f'Tier 1 Ratio: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('cet1_ratio-graph', 'figure'), Output('cet1_ratio-value', 'children'), Output('cet1_ratio-status', 'children'), Output('cet1_ratio-interpretation', 'children')],
    Input('cet1_ratio-graph', 'id')
)
def update_cet1_ratio(_):
    value, status, interp = calculate_cet1_ratio(capital_compliance_df)
    fig = create_gauge_chart(value, 20, 'CET1 Ratio (%)', status)
    return fig, f'CET1 Ratio: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('aml_compliance-graph', 'figure'), Output('aml_compliance-value', 'children'), Output('aml_compliance-status', 'children'), Output('aml_compliance-interpretation', 'children')],
    Input('aml_compliance-graph', 'id')
)
def update_aml_compliance(_):
    value, status, interp = calculate_aml_compliance(capital_compliance_df)
    fig = create_gauge_chart(value, 100, 'AML Compliance (%)', status)
    return fig, f'AML Compliance: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('libor_exposure-graph', 'figure'), Output('libor_exposure-value', 'children'), Output('libor_exposure-status', 'children'), Output('libor_exposure-interpretation', 'children')],
    Input('libor_exposure-graph', 'id')
)
def update_libor_exposure(_):
    value, status, interp = calculate_libor_exposure(capital_compliance_df)
    fig = create_gauge_chart(value, 100, 'LIBOR Exposure (%)', status)
    return fig, f'LIBOR Exposure: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('scb-graph', 'figure'), Output('scb-value', 'children'), Output('scb-status', 'children'), Output('scb-interpretation', 'children')],
    Input('scb-graph', 'id')
)
def update_scb(_):
    value, status, interp = calculate_scb(capital_compliance_df)
    fig = create_gauge_chart(value, 5e6, 'SCB ($)', status)
    return fig, f'SCB: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('ccar_readiness-graph', 'figure'), Output('ccar_readiness-value', 'children'), Output('ccar_readiness-status', 'children'), Output('ccar_readiness-interpretation', 'children')],
    Input('ccar_readiness-graph', 'id')
)
def update_ccar_readiness(_):
    value, status, interp = calculate_ccar_readiness(capital_compliance_df)
    fig = create_gauge_chart(value, 100, 'CCAR Readiness (%)', status)
    return fig, f'CCAR Readiness: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('basel_readiness-graph', 'figure'), Output('basel_readiness-value', 'children'), Output('basel_readiness-status', 'children'), Output('basel_readiness-interpretation', 'children')],
    Input('basel_readiness-graph', 'id')
)
def update_basel_readiness(_):
    value, status, interp = calculate_basel_readiness(capital_compliance_df)
    fig = create_gauge_chart(value, 100, 'Basel III Readiness (%)', status)
    return fig, f'Basel III Readiness: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('compliance_score-graph', 'figure'), Output('compliance_score-value', 'children'), Output('compliance_score-status', 'children'), Output('compliance_score-interpretation', 'children')],
    Input('compliance_score-graph', 'id')
)
def update_compliance_score(_):
    value, status, interp = calculate_compliance_score(capital_compliance_df)
    fig = create_gauge_chart(value, 100, 'Compliance Score (%)', status)
    return fig, f'Compliance Score: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('op_rwa-graph', 'figure'), Output('op_rwa-value', 'children'), Output('op_rwa-status', 'children'), Output('op_rwa-interpretation', 'children')],
    Input('op_rwa-graph', 'id')
)
def update_op_rwa(_):
    value, status, interp = calculate_operational_rwa(capital_compliance_df)
    fig = create_gauge_chart(value, 5e6, 'Operational RWA ($)', status)
    return fig, f'Operational RWA: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('composite_risk-graph', 'figure'), Output('composite_risk-value', 'children'), Output('composite_risk-status', 'children'), Output('composite_risk-interpretation', 'children')],
    Input('composite_risk-graph', 'id')
)
def update_composite_risk(_):
    value, status, interp = calculate_composite_risk_index(credit_risk_df, liquidity_risk_df, market_risk_df)
    fig = create_gauge_chart(value, 2, 'Composite Risk Index', status)
    return fig, f'Composite Risk Index: {value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('npl_dash-graph', 'figure'), Output('npl_dash-value', 'children'), Output('npl_dash-status', 'children'), Output('npl_dash-interpretation', 'children')],
    Input('npl_dash-graph', 'id')
)
def update_npl_dash(_):
    value, status, interp = calculate_npl_ratio(credit_risk_df)
    fig = create_gauge_chart(value, 100, 'NPL Ratio (%)', status)
    return fig, f'NPL Ratio: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('lcr_dash-graph', 'figure'), Output('lcr_dash-value', 'children'), Output('lcr_dash-status', 'children'), Output('lcr_dash-interpretation', 'children')],
    Input('lcr_dash-graph', 'id')
)
def update_lcr_dash(_):
    value, status, interp = calculate_lcr(liquidity_risk_df)
    fig = create_gauge_chart(value, 200, 'LCR (%)', status)
    return fig, f'LCR: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('nsfr_dash-graph', 'figure'), Output('nsfr_dash-value', 'children'), Output('nsfr_dash-status', 'children'), Output('nsfr_dash-interpretation', 'children')],
    Input('nsfr_dash-graph', 'id')
)
def update_nsfr_dash(_):
    value, status, interp = calculate_nsfr(liquidity_risk_df)
    fig = create_gauge_chart(value, 200, 'NSFR (%)', status)
    return fig, f'NSFR: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('var_dash-graph', 'figure'), Output('var_dash-value', 'children'), Output('var_dash-status', 'children'), Output('var_dash-interpretation', 'children')],
    Input('var_dash-graph', 'id')
)
def update_var_dash(_):
    value, status, interp = calculate_var(market_risk_df)
    fig = create_gauge_chart(value, market_risk_df['Quantity'].sum() * market_risk_df['Closing_Price'].mean() * 0.2, 'VaR ($)', status)
    return fig, f'VaR: ${value:.2f}', f'Status: {status}', interp

@dash_app.callback(
    [Output('tier1_dash-graph', 'figure'), Output('tier1_dash-value', 'children'), Output('tier1_dash-status', 'children'), Output('tier1_dash-interpretation', 'children')],
    Input('tier1_dash-graph', 'id')
)
def update_tier1_dash(_):
    value, status, interp = calculate_tier1_ratio(capital_compliance_df)
    fig = create_gauge_chart(value, 20, 'Tier 1 Ratio (%)', status)
    return fig, f'Tier 1 Ratio: {value:.2f}%', f'Status: {status}', interp

@dash_app.callback(
    [Output('ifrs9_analysis-graph', 'figure'), Output('ifrs9_analysis-value', 'children'), Output('ifrs9_analysis-status', 'children'), Output('ifrs9_analysis-interpretation', 'children')],
    Input('ifrs9_analysis-graph', 'id')
)
def update_ifrs9_analysis(_):
    value, status, interp = calculate_ecl(credit_risk_df)
    fig = create_gauge_chart(value, credit_risk_df['Principal_Amount'].sum() * 0.5, 'ECL ($)', status)
    return fig, f'ECL: ${value:.2f}', f'Status: {status}', interp

# Callbacks for AI Analytics predictions
@dash_app.callback(
    [Output('npl_ratio-prediction', 'children'), Output('npl_ratio-prediction-status', 'children')],
    [Input('npl_ratio-button', 'n_clicks')],
    [dash.dependencies.State('npl_ratio-input', 'value')]
)
def update_npl_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Principal_Amount', 'Days_Past_Due', 'Credit_Score']]
            prediction = ml_models['npl_ratio'].predict(X)[0]
            status = "High Risk" if prediction > 5 else "Acceptable"
            return f'Predicted NPL Ratio: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('pcr-prediction', 'children'), Output('pcr-prediction-status', 'children')],
    [Input('pcr-button', 'n_clicks')],
    [dash.dependencies.State('pcr-input', 'value')]
)
def update_pcr_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Provision_Amount', 'Principal_Amount', 'Days_Past_Due']]
            prediction = ml_models['pcr'].predict(X)[0]
            status = "Non-Compliant" if prediction < 70 else "Compliant"
            return f'Predicted PCR: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('ecl-prediction', 'children'), Output('ecl-prediction-status', 'children')],
    [Input('ecl-button', 'n_clicks')],
    [dash.dependencies.State('ecl-input', 'value')]
)
def update_ecl_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Credit_Score', 'Collateral_Value', 'Principal_Amount', 'Commitment_Amount']]
            prediction = ml_models['ecl'].predict(X)[0]
            status = "High ECL" if prediction > df['Principal_Amount'].sum() * 0.1 else "Manageable"
            return f'Predicted ECL: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''


@dash_app.callback(
    [Output('pd-prediction', 'children'), Output('pd-prediction-status', 'children')],
    [Input('pd-button', 'n_clicks')],
    [dash.dependencies.State('pd-input', 'value')]
)
def update_pd_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Credit_Score', 'Days_Past_Due']]
            prediction = ml_models['pd'].predict(X)[0]
            status = "High Risk" if prediction > 0.03 else "Low Risk"
            return f'Predicted PD: {prediction:.2%}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('lgd-prediction', 'children'), Output('lgd-prediction-status', 'children')],
    [Input('lgd-button', 'n_clicks')],
    [dash.dependencies.State('lgd-input', 'value')]
)
def update_lgd_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Collateral_Value', 'Principal_Amount']]
            prediction = ml_models['lgd'].predict(X)[0]
            status = "High Risk" if prediction > 0.5 else "Acceptable"
            return f'Predicted LGD: {prediction:.2%}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('ead-prediction', 'children'), Output('ead-prediction-status', 'children')],
    [Input('ead-button', 'n_clicks')],
    [dash.dependencies.State('ead-input', 'value')]
)
def update_ead_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Principal_Amount', 'Commitment_Amount']]
            prediction = ml_models['ead'].predict(X)[0]
            status = "High Exposure" if prediction > 1e7 else "Manageable"
            return f'Predicted EAD: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('ifrs9_stage-prediction', 'children'), Output('ifrs9_stage-prediction-status', 'children')],
    [Input('ifrs9_stage-button', 'n_clicks')],
    [dash.dependencies.State('ifrs9_stage-input', 'value')]
)
def update_ifrs9_stage_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Credit_Score', 'Days_Past_Due']]
            prediction = ml_models['ifrs9_stage'].predict(X)[0]
            status = {1: "Stage 1", 2: "Stage 2", 3: "Stage 3"}.get(int(prediction), "Unknown")
            return f'Predicted IFRS 9 Stage: {prediction}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('lcr-prediction', 'children'), Output('lcr-prediction-status', 'children')],
    [Input('lcr-button', 'n_clicks')],
    [dash.dependencies.State('lcr-input', 'value')]
)
def update_lcr_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Asset_Value', 'Outflow_Amount', 'ASF_Weight']]
            prediction = ml_models['lcr'].predict(X)[0]
            status = "Non-Compliant" if prediction < 100 else "Compliant"
            return f'Predicted LCR: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('nsfr-prediction', 'children'), Output('nsfr-prediction-status', 'children')],
    [Input('nsfr-button', 'n_clicks')],
    [dash.dependencies.State('nsfr-input', 'value')]
)
def update_nsfr_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Funding_Amount', 'Asset_Value', 'ASF_Weight', 'RSF_Weight']]
            prediction = ml_models['nsfr'].predict(X)[0]
            status = "Non-Compliant" if prediction < 100 else "Compliant"
            return f'Predicted NSFR: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('var-prediction', 'children'), Output('var-prediction-status', 'children')],
    [Input('var-button', 'n_clicks')],
    [dash.dependencies.State('var-input', 'value')]
)
def update_var_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Quantity', 'Closing_Price', 'Return_Volatility']]
            prediction = ml_models['var'].predict(X)[0]
            status = "High Risk" if prediction > df['Quantity'].sum() * df['Closing_Price'].mean() * 0.05 else "Acceptable"
            return f'Predicted VaR: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('es-prediction', 'children'), Output('es-prediction-status', 'children')],
    [Input('es-button', 'n_clicks')],
    [dash.dependencies.State('es-input', 'value')]
)
def update_es_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Quantity', 'Closing_Price', 'Return_Volatility']]
            prediction = ml_models['es'].predict(X)[0]
            status = "High Risk" if prediction > df['Quantity'].sum() * df['Closing_Price'].mean() * 0.075 else "Acceptable"
            return f'Predicted ES: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('tier1_ratio-prediction', 'children'), Output('tier1_ratio-prediction-status', 'children')],
    [Input('tier1_ratio-button', 'n_clicks')],
    [dash.dependencies.State('tier1_ratio-input', 'value')]
)
def update_tier1_ratio_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']]
            prediction = ml_models['tier1_ratio'].predict(X)[0]
            status = "Non-Compliant" if prediction < 6 else "Compliant"
            return f'Predicted Tier 1 Ratio: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('cet1_ratio-prediction', 'children'), Output('cet1_ratio-prediction-status', 'children')],
    [Input('cet1_ratio-button', 'n_clicks')],
    [dash.dependencies.State('cet1_ratio-input', 'value')]
)
def update_cet1_ratio_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']]
            prediction = ml_models['cet1_ratio'].predict(X)[0]
            status = "Non-Compliant" if prediction < 4.5 else "Compliant"
            return f'Predicted CET1 Ratio: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('aml_compliance-prediction', 'children'), Output('aml_compliance-prediction-status', 'children')],
    [Input('aml_compliance-button', 'n_clicks')],
    [dash.dependencies.State('aml_compliance-input', 'value')]
)
def update_aml_compliance_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Transaction_Amount', 'Risk_Score']]
            prediction = ml_models['aml_compliance'].predict(X)[0]
            status = "Pass" if prediction == 'Pass' else "Fail"
            return f'Predicted AML Compliance: {prediction}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('libor_exposure-prediction', 'children'), Output('libor_exposure-prediction-status', 'children')],
    [Input('libor_exposure-button', 'n_clicks')],
    [dash.dependencies.State('libor_exposure-input', 'value')]
)
def update_libor_exposure_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            df['Rate_Type_Encoded'] = le_rate.transform(df['Rate_Type'])
            X = df[['Transaction_Amount', 'Rate_Type_Encoded']]
            prediction = ml_models['libor_exposure'].predict(X)[0]
            status = "Non-Compliant" if prediction == 1 else "Compliant"
            return f'Predicted LIBOR Exposure: {prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('scb-prediction', 'children'), Output('scb-prediction-status', 'children')],
    [Input('scb-button', 'n_clicks')],
    [dash.dependencies.State('scb-input', 'value')]
)
def update_scb_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            df['Stress_Scenario_ID_Encoded'] = le_stress.transform(df['Stress_Scenario_ID'])
            X = df[['Capital_Amount', 'Loss_Amount', 'Stress_Scenario_ID_Encoded']]
            prediction = ml_models['scb'].predict(X)[0]
            status = "High Stress Impact" if prediction > 1e6 else "Manageable"
            return f'Predicted SCB: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('ccar_readiness-prediction', 'children'), Output('ccar_readiness-prediction-status', 'children')],
    [Input('ccar_readiness-button', 'n_clicks')],
    [dash.dependencies.State('ccar_readiness-input', 'value')]
)
def update_ccar_readiness_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
            prediction = ml_models['ccar_readiness'].predict(X)[0]
            status = "Not Ready" if prediction < 80 else "Ready"
            return f'Predicted CCAR Readiness: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('basel_readiness-prediction', 'children'), Output('basel_readiness-prediction-status', 'children')],
    [Input('basel_readiness-button', 'n_clicks')],
    [dash.dependencies.State('basel_readiness-input', 'value')]
)
def update_basel_readiness_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
            prediction = ml_models['basel_readiness'].predict(X)[0]
            status = "Not Ready" if prediction < 80 else "Ready"
            return f'Predicted Basel III Readiness: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('compliance_score-prediction', 'children'), Output('compliance_score-prediction-status', 'children')],
    [Input('compliance_score-button', 'n_clicks')],
    [dash.dependencies.State('compliance_score-input', 'value')]
)
def update_compliance_score_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
            prediction = ml_models['compliance_score'].predict(X)[0]
            status = "Non-Compliant" if prediction < 80 else "Compliant"
            return f'Predicted Compliance Score: {prediction:.2f}%', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('op_rwa-prediction', 'children'), Output('op_rwa-prediction-status', 'children')],
    [Input('op_rwa-button', 'n_clicks')],
    [dash.dependencies.State('op_rwa-input', 'value')]
)
def update_op_rwa_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Revenue_Amount', 'Loss_Amount']]
            prediction = ml_models['op_rwa'].predict(X)[0]
            status = "High Risk" if prediction > 1e6 else "Acceptable"
            return f'Predicted Operational RWA: ${prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

@dash_app.callback(
    [Output('composite_risk-prediction', 'children'), Output('composite_risk-prediction-status', 'children')],
    [Input('composite_risk-button', 'n_clicks')],
    [dash.dependencies.State('composite_risk-input', 'value')]
)
def update_composite_risk_prediction(n_clicks, input_data):
    if n_clicks and input_data:
        try:
            df = pd.DataFrame(eval(input_data))
            X = df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']]
            prediction = ml_models['composite_risk'].predict(X)[0]
            status = "High Risk" if prediction > 0.7 else "Acceptable"
            return f'Predicted Composite Risk Index: {prediction:.2f}', f'Status: {status}'
        except Exception as e:
            return f'Error: {str(e)}', ''
    return 'Enter JSON data and click Predict', ''

# Run Flask app
if __name__ == '__main__':
    flask_app.run(host='0.0.0.0', port=8000) 
