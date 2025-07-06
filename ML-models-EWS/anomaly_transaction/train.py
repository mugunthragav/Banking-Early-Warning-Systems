import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import pickle
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load dataset
file_path = os.path.join(DATA_DIR, 'data.csv')
data = pd.read_csv(file_path, encoding='ISO-8859-1')
print(f"Data loaded from {file_path}. Shape: {data.shape}")

# Data Cleaning
data_cleaned = data.dropna(subset=['CustomerID']).copy()
data_cleaned['InvoiceDate'] = pd.to_datetime(data_cleaned['InvoiceDate'], format='%m/%d/%Y %H:%M', errors='coerce')
data_cleaned = data_cleaned.dropna(subset=['InvoiceDate']).drop_duplicates()

# Feature Engineering
data_cleaned['TotalSpent'] = data_cleaned['Quantity'] * data_cleaned['UnitPrice']
data_cleaned['DescriptionLength'] = data_cleaned['Description'].apply(lambda x: len(str(x)))
data_cleaned['TransactionHour'] = data_cleaned['InvoiceDate'].dt.hour
data_cleaned['TransactionDay'] = data_cleaned['InvoiceDate'].dt.dayofweek

# Customer Historical Stats
customer_stats = data_cleaned.groupby('CustomerID').agg({
    'TotalSpent': ['mean', 'std', 'count'],
    'InvoiceNo': 'nunique',
    'Quantity': ['mean', 'std']
}).reset_index()
customer_stats.columns = ['CustomerID', 'HistAvgSpent', 'HistStdSpent', 'HistCount', 'HistFreq', 'HistAvgQuantity', 'HistStdQuantity']
data_cleaned = data_cleaned.merge(customer_stats, on='CustomerID', how='left')

# Peer Group Stats (by Country)
peer_stats = data_cleaned.groupby('Country').agg({
    'TotalSpent': ['mean', 'std'],
    'InvoiceNo': 'nunique'
}).reset_index()
peer_stats.columns = ['Country', 'PeerAvgSpent', 'PeerStdSpent', 'PeerFreq']
data_cleaned = data_cleaned.merge(peer_stats, on='Country', how='left')

# Most Frequent Country per Customer (for UnusualDestination)
customer_countries = data_cleaned.groupby('CustomerID')['Country'].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
customer_countries.columns = ['CustomerID', 'MostFrequentCountry']
data_cleaned = data_cleaned.merge(customer_countries, on='CustomerID', how='left')
data_cleaned['UnusualDestination'] = (data_cleaned['Country'] != data_cleaned['MostFrequentCountry']).astype(int)

# New Criteria
data_cleaned['QuantitySpike'] = ((data_cleaned['Quantity'] - data_cleaned['HistAvgQuantity']) / data_cleaned['HistStdQuantity'].replace(0, 1e-10) > 3).astype(int)

# Handle NaN in stats
data_cleaned['HistStdSpent'] = data_cleaned['HistStdSpent'].fillna(0)
data_cleaned['PeerStdSpent'] = data_cleaned['PeerStdSpent'].fillna(0)
data_cleaned['HistStdQuantity'] = data_cleaned['HistStdQuantity'].fillna(0)

# Features
features = ['Quantity', 'UnitPrice', 'TotalSpent', 'DescriptionLength', 'TransactionHour', 'TransactionDay', 'HistAvgSpent', 'HistStdSpent', 'HistFreq', 'PeerAvgSpent', 'PeerStdSpent', 'PeerFreq']
data_cleaned = data_cleaned.dropna(subset=features)
print(f"Shape after NaN removal: {data_cleaned.shape}")

# Feature Selection and Scaling
selector = VarianceThreshold(threshold=0.1)
features_selected = selector.fit_transform(data_cleaned[features])
selected_features = [features[i] for i in range(len(features)) if selector.get_support()[i]]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(data_cleaned[features])

# Train Models on Full Dataset
iso_forest = IsolationForest(contamination=0.02, random_state=42)
iso_forest.fit(features_scaled)

print("Starting Autoencoder training...")
input_dim = features_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation='relu')(input_layer)
encoder = Dense(32, activation='relu')(encoder)
decoder = Dense(64, activation='relu')(encoder)
decoder = Dense(input_dim, activation='linear')(decoder)
autoencoder = Model(input_layer, decoder)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(features_scaled, features_scaled, epochs=5, batch_size=32, shuffle=True, validation_split=0.2, verbose=1)
reconstructions = autoencoder.predict(features_scaled)
mse = np.mean(np.power(features_scaled - reconstructions, 2), axis=1)
autoencoder_threshold = np.percentile(mse, 98)

# Predict Anomalies on Full Data
data_cleaned['IsoForest_Anomaly'] = iso_forest.predict(features_scaled)  # -1 (anomaly), 1 (normal)
data_cleaned['Autoencoder_Anomaly'] = (mse > autoencoder_threshold).astype(int)  # 1 (anomaly), 0 (normal)
data_cleaned['Autoencoder_Score'] = mse

# Anomaly Flags (3 Sigma or Score > 0.85)
data_cleaned['ZScore'] = (data_cleaned['TotalSpent'] - data_cleaned['HistAvgSpent']) / data_cleaned['HistStdSpent'].replace(0, 1e-10)
data_cleaned['RiskScore'] = np.where(
    (abs(data_cleaned['ZScore']) > 3) | (data_cleaned['Autoencoder_Score'] > 0.85),
    1.0,
    0.0
)

# Select Alert Data
alert_data = data_cleaned[['InvoiceDate', 'TotalSpent', 'Country', 'RiskScore', 'CustomerID', 'ZScore', 'Autoencoder_Score', 'IsoForest_Anomaly', 'Autoencoder_Anomaly', 'HistFreq', 'UnusualDestination', 'QuantitySpike']]
alert_data.to_csv(os.path.join(OUTPUT_DIR, 'alert_data.csv'), index=False)

# Common Anomalies
common_anomalies = data_cleaned[
    (data_cleaned['IsoForest_Anomaly'] == -1) &
    (data_cleaned['Autoencoder_Anomaly'] == 1)
]
data_cleaned.to_csv(os.path.join(OUTPUT_DIR, 'anomaly_results.csv'), index=False)
common_anomalies.to_csv(os.path.join(OUTPUT_DIR, 'common_anomalies.csv'), index=False)
print(f"Models and results saved to {OUTPUT_DIR}")