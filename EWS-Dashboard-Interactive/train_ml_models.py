import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Load datasets with error handling
try:
    credit_risk_df = pd.read_csv('data/credit_risk_dataset.csv')
    liquidity_risk_df = pd.read_csv('data/liquidity_risk_dataset.csv')
    market_risk_df = pd.read_csv('data/market_risk_dataset.csv')
    capital_compliance_df = pd.read_csv('data/capital_compliance_dataset.csv')
except FileNotFoundError as e:
    raise FileNotFoundError(f"Dataset not found: {e}. Ensure 'data/' contains all CSV files.")

# Preprocess categorical columns
le_rate = LabelEncoder()
capital_compliance_df['Rate_Type_Encoded'] = le_rate.fit_transform(capital_compliance_df['Rate_Type'])
le_stress = LabelEncoder()
capital_compliance_df['Stress_Scenario_ID_Encoded'] = le_stress.fit_transform(capital_compliance_df['Stress_Scenario_ID'])

# Helper functions for calculations (used for target variables)
def calculate_npl_ratio(df):
    npl = df[df['Days_Past_Due'] > 90]['Principal_Amount'].sum()
    total_loans = df['Principal_Amount'].sum()
    return (npl / total_loans * 100) if total_loans > 0 else 0

def calculate_pcr(df):
    provisions = df['Provision_Amount'].sum()
    npl = df[df['Days_Past_Due'] > 90]['Principal_Amount'].sum()
    return (provisions / npl * 100) if npl > 0 else 0

def calculate_ecl(df):
    pd = df['Credit_Score'].apply(lambda x: 0.05 if x < 600 else 0.01)
    lgd = (1 - df['Collateral_Value'] / df['Principal_Amount']).clip(0, 1)
    ead = df['Principal_Amount'] + df['Commitment_Amount']
    return (pd * lgd * ead).sum()

def calculate_pd(df):
    return df['Credit_Score'].apply(lambda x: 0.05 if x < 600 else 0.01).mean()

def calculate_lgd(df):
    return (1 - df['Collateral_Value'] / df['Principal_Amount']).clip(0, 1).mean()

def calculate_ead(df):
    return (df['Principal_Amount'] + df['Commitment_Amount']).sum()

def calculate_lcr(df):
    hqla = df[df['Asset_Type'].isin(['Cash', 'Government Bond'])]['Asset_Value'].sum()
    outflows = df['Outflow_Amount'].sum()
    return (hqla / outflows * 100) if outflows > 0 else 0

def calculate_nsfr(df):
    asf = (df['Funding_Amount'] * df['ASF_Weight'] / 100).sum()
    rsf = (df['Asset_Value'] * df['RSF_Weight'] / 100).sum()
    return (asf / rsf * 100) if rsf > 0 else 0

def calculate_var(df):
    portfolio_value = (df['Quantity'] * df['Closing_Price']).sum()
    volatilities = df['Return_Volatility'].values
    weights = (df['Quantity'] * df['Closing_Price'] / portfolio_value).values
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.diag(volatilities), weights)))
    return portfolio_value * portfolio_vol * np.sqrt(10) * 1.645

def calculate_es(df):
    portfolio_value = (df['Quantity'] * df['Closing_Price']).sum()
    volatilities = df['Return_Volatility'].values
    weights = (df['Quantity'] * df['Closing_Price'] / portfolio_value).values
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(np.diag(volatilities), weights)))
    return portfolio_value * portfolio_vol * np.sqrt(10) * 2.241

def calculate_tier1_ratio(df):
    tier1 = df[df['Capital_ID'].str.contains('Equity|Reserves', case=False, na=False)]['Capital_Amount'].sum() - df['Deduction_Amount'].sum()
    rwa = (df['Exposure_Amount'] * df['Risk_Weight'] / 100).sum()
    return (tier1 / rwa * 100) if rwa > 0 else 0

def calculate_cet1_ratio(df):
    cet1 = df[df['Capital_ID'].str.contains('Common Equity', case=False, na=False)]['Capital_Amount'].sum() - df['Deduction_Amount'].sum()
    rwa = (df['Exposure_Amount'] * df['Risk_Weight'] / 100).sum()
    return (cet1 / rwa * 100) if rwa > 0 else 0

def calculate_aml_compliance(df):
    valid = df[df['AML_Flag'] == 'Pass']['Transaction_Amount'].count()
    total = df['Transaction_Amount'].count()
    return (valid / total * 100) if total > 0 else 0

def calculate_libor_exposure(df):
    libor = df[df['Rate_Type'] == 'LIBOR']['Contract_ID'].count()
    total = df['Contract_ID'].count()
    return (libor / total * 100) if total > 0 else 0

def calculate_scb(df):
    return df[df['Stress_Scenario_ID'] == 'S1']['Loss_Amount'].sum()

def calculate_ccar_readiness(df):
    cr_score = calculate_npl_ratio(credit_risk_df)
    lr_score = calculate_lcr(liquidity_risk_df)
    return (100 - cr_score + lr_score) / 2

def calculate_basel_readiness(df):
    cr_score = calculate_npl_ratio(credit_risk_df)
    lr_score = calculate_lcr(liquidity_risk_df)
    mr_score = calculate_var(market_risk_df)
    return (100 - cr_score + lr_score + 100 - mr_score) / 3

def calculate_compliance_score(df):
    cr_score = calculate_npl_ratio(credit_risk_df)
    lr_score = calculate_lcr(liquidity_risk_df)
    mr_score = calculate_var(market_risk_df)
    return 0.4 * (100 - cr_score) + 0.3 * lr_score + 0.3 * (100 - mr_score)

def calculate_operational_rwa(df):
    bi = df['Revenue_Amount'].sum() * 0.12
    ilm = df['Loss_Amount'].count() / 1000
    return bi * ilm

def calculate_composite_risk_index(df_cr, df_lr, df_mr):
    cr_metric = calculate_npl_ratio(df_cr)
    lr_metric = calculate_lcr(df_lr)
    mr_metric = calculate_var(df_mr)
    return (cr_metric + (100 - lr_metric) + mr_metric) / 3

# ML Models dictionary
ml_models = {}

# Models for all 28 components
ml_models['npl_ratio'] = RandomForestRegressor().fit(
    credit_risk_df[['Principal_Amount', 'Days_Past_Due', 'Credit_Score']],
    np.repeat(calculate_npl_ratio(credit_risk_df), len(credit_risk_df))
)
ml_models['pcr'] = RandomForestRegressor().fit(
    credit_risk_df[['Provision_Amount', 'Principal_Amount', 'Days_Past_Due']],
    np.repeat(calculate_pcr(credit_risk_df), len(credit_risk_df))
)
ml_models['ecl'] = RandomForestRegressor().fit(
    credit_risk_df[['Credit_Score', 'Collateral_Value', 'Principal_Amount', 'Commitment_Amount']],
    np.repeat(calculate_ecl(credit_risk_df), len(credit_risk_df))
)
ml_models['pd'] = RandomForestRegressor().fit(
    credit_risk_df[['Credit_Score', 'Days_Past_Due']],
    credit_risk_df['Credit_Score'].apply(lambda x: 0.05 if x < 600 else 0.01)
)
ml_models['lgd'] = RandomForestRegressor().fit(
    credit_risk_df[['Collateral_Value', 'Principal_Amount']],
    (1 - credit_risk_df['Collateral_Value'] / credit_risk_df['Principal_Amount']).clip(0, 1)
)
ml_models['ead'] = RandomForestRegressor().fit(
    credit_risk_df[['Principal_Amount', 'Commitment_Amount']],
    credit_risk_df['Principal_Amount'] + credit_risk_df['Commitment_Amount']
)
ml_models['ifrs9_stage'] = RandomForestClassifier().fit(
    credit_risk_df[['Credit_Score', 'Days_Past_Due']],
    credit_risk_df['Days_Past_Due'].apply(lambda x: 1 if x <= 30 else 2 if x <= 90 else 3)
)
ml_models['lcr'] = RandomForestRegressor().fit(
    liquidity_risk_df[['Asset_Value', 'Outflow_Amount', 'ASF_Weight']],
    np.repeat(calculate_lcr(liquidity_risk_df), len(liquidity_risk_df))
)
ml_models['nsfr'] = RandomForestRegressor().fit(
    liquidity_risk_df[['Funding_Amount', 'Asset_Value', 'ASF_Weight', 'RSF_Weight']],
    np.repeat(calculate_nsfr(liquidity_risk_df), len(liquidity_risk_df))
)
ml_models['var'] = RandomForestRegressor().fit(
    market_risk_df[['Quantity', 'Closing_Price', 'Return_Volatility']],
    np.repeat(calculate_var(market_risk_df), len(market_risk_df))
)
ml_models['es'] = RandomForestRegressor().fit(
    market_risk_df[['Quantity', 'Closing_Price', 'Return_Volatility']],
    np.repeat(calculate_es(market_risk_df), len(market_risk_df))
)
ml_models['tier1_ratio'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']],
    np.repeat(calculate_tier1_ratio(capital_compliance_df), len(capital_compliance_df))
)
ml_models['cet1_ratio'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Deduction_Amount', 'Exposure_Amount', 'Risk_Weight']],
    np.repeat(calculate_cet1_ratio(capital_compliance_df), len(capital_compliance_df))
)
ml_models['aml_compliance'] = RandomForestClassifier().fit(
    capital_compliance_df[['Transaction_Amount', 'Risk_Score']],
    capital_compliance_df['AML_Flag']
)
ml_models['libor_exposure'] = RandomForestClassifier().fit(
    capital_compliance_df[['Transaction_Amount', 'Rate_Type_Encoded']],
    capital_compliance_df['Rate_Type'].apply(lambda x: 1 if x == 'LIBOR' else 0)
)
ml_models['scb'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Loss_Amount', 'Stress_Scenario_ID_Encoded']],
    np.repeat(calculate_scb(capital_compliance_df), len(capital_compliance_df))
)
ml_models['ccar_readiness'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']],
    np.repeat(calculate_ccar_readiness(capital_compliance_df), len(capital_compliance_df))
)
ml_models['basel_readiness'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']],
    np.repeat(calculate_basel_readiness(capital_compliance_df), len(capital_compliance_df))
)
ml_models['compliance_score'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']],
    np.repeat(calculate_compliance_score(capital_compliance_df), len(capital_compliance_df))
)
ml_models['op_rwa'] = RandomForestRegressor().fit(
    capital_compliance_df[['Revenue_Amount', 'Loss_Amount']],
    np.repeat(calculate_operational_rwa(capital_compliance_df), len(capital_compliance_df))
)
ml_models['composite_risk'] = RandomForestRegressor().fit(
    capital_compliance_df[['Capital_Amount', 'Loss_Amount', 'Risk_Score']],
    np.repeat(calculate_composite_risk_index(credit_risk_df, liquidity_risk_df, market_risk_df), len(capital_compliance_df))
)

# Save models
with open('models/ml_models.pkl', 'wb') as f:
    pickle.dump(ml_models, f)
