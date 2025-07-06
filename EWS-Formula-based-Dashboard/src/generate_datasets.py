import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

fake = Faker()
np.random.seed(42)

# Dataset 1: Credit Risk Dataset (20,000 rows)
credit_risk_data = {
    'Loan_ID': [f'L{i:06d}' for i in range(20000)],
    'Principal_Amount': np.random.uniform(10000, 1000000, 20000),
    'Days_Past_Due': np.random.choice([0, 30, 60, 90, 120, 180], 20000, p=[0.7, 0.15, 0.05, 0.05, 0.025, 0.025]),
    'Loan_Status': ['Active'] * 20000,
    'Credit_Score': np.random.randint(300, 850, 20000),
    'Collateral_Value': np.random.uniform(0, 800000, 20000),
    'Commitment_Amount': np.random.uniform(0, 500000, 20000),
    'Provision_Amount': np.random.uniform(0, 100000, 20000)
}
credit_risk_data['Loan_Status'] = ['Delinquent' if d > 90 else 'Active' for d in credit_risk_data['Days_Past_Due']]
credit_risk_df = pd.DataFrame(credit_risk_data)
credit_risk_df.to_csv('data/credit_risk_dataset.csv', index=False)

# Dataset 2: Liquidity Risk Dataset (20,000 rows)
liquidity_risk_data = {
    'Asset_ID': [f'A{i:06d}' for i in range(20000)],
    'Asset_Type': np.random.choice(['Cash', 'Government Bond', 'Corporate Bond'], 20000, p=[0.4, 0.4, 0.2]),
    'Asset_Value': np.random.uniform(10000, 5000000, 20000),
    'Outflow_Amount': np.random.uniform(5000, 2000000, 20000),
    'Funding_ID': [f'F{i:06d}' for i in range(20000)],
    'Funding_Amount': np.random.uniform(10000, 5000000, 20000),
    'Maturity_Date': [fake.date_between(start_date='-5y', end_date='+5y').strftime('%Y-%m-%d') for _ in range(20000)],
    'ASF_Weight': np.random.uniform(50, 100, 20000),
    'RSF_Weight': np.random.uniform(10, 100, 20000)
}
liquidity_risk_df = pd.DataFrame(liquidity_risk_data)
liquidity_risk_df.to_csv('data/liquidity_risk_dataset.csv', index=False)

# Dataset 3: Market Risk Dataset (20,000 rows)
market_risk_data = {
    'Asset_ID': [f'M{i:06d}' for i in range(20000)],
    'Quantity': np.random.randint(1, 1000, 20000),
    'Closing_Price': np.random.uniform(10, 1000, 20000),
    'Return_Volatility': np.random.uniform(0.01, 0.1, 20000),
    'Return_Date': [fake.date_between(start_date='-1y', end_date='today').strftime('%Y-%m-%d') for _ in range(20000)]
}
market_risk_df = pd.DataFrame(market_risk_data)
market_risk_df.to_csv('data/market_risk_dataset.csv', index=False)

# Dataset 4: Capital and Compliance Dataset (20,000 rows)
capital_compliance_data = {
    'Capital_ID': [f'C{i:06d}' for i in range(20000)],
    'Capital_Amount': np.random.uniform(100000, 10000000, 20000),
    'Deduction_Amount': np.random.uniform(0, 1000000, 20000),
    'Exposure_ID': [f'E{i:06d}' for i in range(20000)],
    'Exposure_Amount': np.random.uniform(10000, 5000000, 20000),
    'Risk_Weight': np.random.uniform(20, 150, 20000),
    'Transaction_ID': [f'T{i:06d}' for i in range(20000)],
    'Transaction_Amount': np.random.uniform(1000, 1000000, 20000),
    'AML_Flag': np.random.choice(['Pass', 'Fail'], 20000, p=[0.95, 0.05]),
    'Contract_ID': [f'CON{i:06d}' for i in range(20000)],
    'Rate_Type': np.random.choice(['LIBOR', 'SOFR'], 20000, p=[0.05, 0.95]),
    'Loss_Amount': np.random.uniform(0, 500000, 20000),
    'Risk_Score': np.random.uniform(50, 100, 20000),
    'Revenue_Amount': np.random.uniform(1000000, 100000000, 20000),
    'Loss_Event_ID': [f'LE{i:06d}' for i in range(20000)],
    'Stress_Scenario_ID': np.random.choice(['S1', 'S2', 'S3'], 20000)
}
capital_compliance_df = pd.DataFrame(capital_compliance_data)
capital_compliance_df.to_csv('data/capital_compliance_dataset.csv', index=False)
