import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)

# Define parameters
start_date = datetime(1975, 4, 20)
end_date = datetime(2025, 4, 20)
n_rows = (end_date - start_date).days + 1  # 50 years of daily data

dates = [start_date + timedelta(days=x) for x in range(n_rows)]

# Generate synthetic data
data = {
    'date': dates,
    'hqla_value': np.random.uniform(10000, 15000, n_rows),  # High-Quality Liquid Assets
    'outflows': np.random.uniform(200, 400, n_rows),
    'inflows': np.random.uniform(100, 200, n_rows),
    'transaction_type': np.random.choice(['Deposit', 'Withdrawal'], n_rows),
    'asset_type': np.random.choice(['Cash', 'Bonds'], n_rows),
    'liability_type': np.random.choice(['Short-term Loan', 'Revolving Credit'], n_rows),
    'maturity_days': np.random.randint(30, 90, n_rows),
    'market_value': np.random.uniform(9000, 14000, n_rows),
    'leverage_ratio': np.random.uniform(4.5, 6.0, n_rows),
    'rwa': np.random.uniform(12000, 18000, n_rows),  # Risk-Weighted Assets
    'stable_funding': np.random.uniform(18000, 30000, n_rows),
    'required_funding': np.random.uniform(15000, 22000, n_rows),
    'maturity_period': np.random.choice(['1Y', '2Y', '3Y', '4Y'], n_rows),
    'deposit_type': np.random.choice(['Retail', 'Wholesale'], n_rows),
    'loan_type': np.random.choice(['Mortgage', 'Commercial'], n_rows),
    'capital_ratio': np.random.uniform(12.0, 13.0, n_rows),
    'funding_cost': np.random.uniform(2.1, 2.3, n_rows),
    'asset_quality': np.random.choice(['Low', 'Medium', 'High'], n_rows),
    'credit_rating': np.random.choice(['A', 'B', 'AA', 'BBB'], n_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure directory exists
os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'), exist_ok=True)

# Save to CSV with explicit full timestamp format
df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'transaction_data.csv'), index=False)
print(f"Generated transaction_data.csv with {n_rows} rows, date range: {df['date'].min()} to {df['date'].max()}")