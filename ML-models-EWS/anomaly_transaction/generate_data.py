import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n_transactions = 500000
n_customers = 10000

start_date = datetime(2024, 1, 1)
end_date = datetime(2025, 4, 20)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
dates = np.random.choice(date_range, n_transactions, p=np.linspace(0.1, 1, len(date_range))/sum(np.linspace(0.1, 1, len(date_range))))

customer_ids = [f"C{i:05d}" for i in range(n_customers)]
customer_assignments = np.random.choice(customer_ids, n_transactions, p=np.array([1/n_customers]*n_customers))

account_types = ['Savings', 'Checking', 'Credit']
countries = ['USA', 'UK', 'India', 'Germany']
transaction_types = ['Deposit', 'Withdrawal', 'Transfer']
currencies = {'USA': 'USD', 'UK': 'GBP', 'India': 'INR', 'Germany': 'EUR'}
merchant_categories = ['Retail', 'Online', 'Travel']
channels = ['Online', 'ATM', 'In-Person']
cities = {'USA': ['New York', 'Los Angeles'], 'UK': ['London', 'Manchester'], 'India': ['Mumbai', 'Delhi'], 'Germany': ['Berlin', 'Munich']}

data = {
    'Transaction ID': [f"T{i:06d}" for i in range(n_transactions)],
    'Date': dates,
    'Account Type': np.random.choice(account_types, n_transactions, p=[0.5, 0.3, 0.2]),
    'Country': np.random.choice(countries, n_transactions, p=[0.4, 0.2, 0.2, 0.2]),
    'Transaction Type': np.random.choice(transaction_types, n_transactions, p=[0.4, 0.3, 0.3]),
    'Amount': np.random.lognormal(mean=6.9, sigma=2, size=n_transactions),
    'Currency': [currencies[c] for c in np.random.choice(countries, n_transactions, p=[0.4, 0.2, 0.2, 0.2])],
    'Customer ID': customer_assignments,
    'Merchant Category': np.random.choice(merchant_categories, n_transactions),
    'Frequency': np.random.randint(1, 11, n_transactions),
    'Location': [f"{random.choice(cities[c])}, {random.uniform(-90, 90)}, {random.uniform(-180, 180)}" for c in np.random.choice(countries, n_transactions)],
    'Channel': np.random.choice(channels, n_transactions, p=[0.5, 0.25, 0.25])
}

df = pd.DataFrame(data)
df['Amount'] = df['Amount'].clip(upper=10000)

# Inject anomalies
anomaly_mask = np.random.random(n_transactions) < 0.015
df.loc[anomaly_mask, 'Amount'] = df.loc[anomaly_mask, 'Amount'] * np.random.uniform(5, 15)
df.loc[anomaly_mask, 'Location'] = [f"Anomaly_{i}, {random.uniform(-90, 90)}, {random.uniform(-180, 180)}" for i in range(anomaly_mask.sum())]
df.loc[anomaly_mask, 'Transaction Type'] = np.random.choice(['Fraud', 'Unusual'], size=anomaly_mask.sum())

df.to_csv('data/transaction_data.csv', index=False)
print(f"Generated {len(df)} transactions with {df['Customer ID'].nunique()} unique customers.")