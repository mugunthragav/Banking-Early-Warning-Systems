import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Setup
os.makedirs('data', exist_ok=True)
np.random.seed(123)
random.seed(123)

# Constants
BANKS = ['Citigroup', 'HSBC', 'Deutsche Bank', 'Barclays']
CURRENCIES = ['USD', 'EUR', 'GBP', 'CAD']
RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']
RATING_PROBS = [0.12, 0.18, 0.22, 0.25, 0.15, 0.05, 0.02, 0.01]  # Sums to 1.0
SECTORS = ['Finance', 'Tech', 'Energy', 'Retail']

def generate_dates(start, end, size):
    """Generate random business dates"""
    start_dt = datetime.strptime(start, '%Y-%m-%d')
    end_dt = datetime.strptime(end, '%Y-%m-%d')
    days_diff = (end_dt - start_dt).days
    dates = []
    for _ in range(size):
        random_days = random.randint(0, days_diff)
        date = start_dt + timedelta(days=random_days)
        while date.weekday() >= 5:  # Skip weekends
            date += timedelta(days=1)
        dates.append(date.strftime('%Y-%m-%d'))
    return dates

# Credit Risk Dataset
print("Generating Credit Risk Dataset...")
credit_data = {
    'Loan_ID': [f'LN{i:06d}' for i in range(5000)],
    'Bank': np.random.choice(BANKS, 5000),
    'Client_Type': np.random.choice(['Corporate', 'Individual'], 5000, p=[0.65, 0.35]),
    'Loan_Amount': np.random.lognormal(mean=10, sigma=1, size=5000),
    'Balance': None,
    'Rate': np.random.normal(5, 2, 5000),
    'Term_Months': np.random.choice([12, 36, 60, 120], 5000, p=[0.3, 0.3, 0.2, 0.2]),
    'Start_Date': generate_dates('2022-01-01', '2024-12-31', 5000),
    'End_Date': None,
    'Days_Overdue': np.random.choice([0, 30, 60, 90], 5000, p=[0.8, 0.12, 0.05, 0.03]),
    'Rating': np.random.choice(RATINGS, 5000, p=RATING_PROBS),
    'Default_Prob': np.random.beta(1, 25, 5000),
    'Loss_Given_Default': np.random.beta(2, 4, 5000),
}

credit_df = pd.DataFrame(credit_data)
credit_df['Balance'] = credit_df['Loan_Amount'] * np.random.uniform(0.7, 1.0, 5000)
credit_df['Rate'] = np.clip(credit_df['Rate'], 1, 10)
credit_df['Start_Date'] = pd.to_datetime(credit_df['Start_Date'])
credit_df['End_Date'] = credit_df['Start_Date'] + pd.to_timedelta(credit_df['Term_Months'] * 30, unit='D')
credit_df['Status'] = credit_df['Days_Overdue'].apply(
    lambda x: 'Default' if x >= 90 else 'Delinquent' if x > 0 else 'Performing')
credit_df.to_csv('data/credit_risk_data.csv', index=False)

# Liquidity Risk Dataset
print("Generating Liquidity Risk Dataset...")
liquidity_data = {
    'Asset_ID': [f'AS{i:06d}' for i in range(5000)],
    'Bank': np.random.choice(BANKS, 5000),
    'Asset_Category': np.random.choice(['Cash', 'Bonds', 'Stocks'], 5000, p=[0.5, 0.3, 0.2]),
    'Asset_Amount': np.random.lognormal(mean=11, sigma=1.2, size=5000),
    'Currency': np.random.choice(CURRENCIES, 5000),
    'Rating': np.random.choice(RATINGS, 5000, p=RATING_PROBS),
    'Buffer_Amount': np.random.lognormal(mean=10, sigma=1, size=5000),
}

liquidity_df = pd.DataFrame(liquidity_data)
liquidity_df['LCR_Compliance'] = (liquidity_df['Buffer_Amount'] / liquidity_df['Asset_Amount'] * 100) >= 100
liquidity_df.to_csv('data/liquidity_risk_data.csv', index=False)

# Market Risk Dataset
print("Generating Market Risk Dataset...")
market_data = {
    'Trade_ID': [f'TR{i:06d}' for i in range(5000)],
    'Bank': np.random.choice(BANKS, 5000),
    'Instrument': np.random.choice(['Stock', 'Bond', 'FX'], 5000, p=[0.4, 0.35, 0.25]),
    'Value': np.random.lognormal(mean=10, sigma=1.3, size=5000),
    'Currency': np.random.choice(CURRENCIES, 5000),
    'Return': np.random.normal(0, 0.01, 5000),
    'VaR_95': None,
}

market_df = pd.DataFrame(market_data)
market_df['VaR_95'] = market_df['Value'] * 0.01 * 1.65  # Simplified VaR
market_df.to_csv('data/market_risk_data.csv', index=False)

# Capital and Compliance Dataset
print("Generating Capital and Compliance Dataset...")
capital_data = {
    'Bank': np.random.choice(BANKS, 5000),
    'Capital_ID': [f'CP{i:06d}' for i in range(5000)],
    'Capital_Amount': np.random.lognormal(mean=14, sigma=1.2, size=5000),
    'Capital_Ratio': np.random.uniform(9, 16, 5000),
    'RWA': np.random.lognormal(mean=15, sigma=1.3, size=5000),
    'Rating': np.random.choice(RATINGS, 5000, p=RATING_PROBS),
}

capital_df = pd.DataFrame(capital_data)
capital_df.to_csv('data/capital_compliance_data.csv', index=False)

print("\nData Generation Complete!")
print("Generated datasets: credit_risk_data.csv, liquidity_risk_data.csv, market_risk_data.csv, capital_compliance_data.csv")