import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import os
import sys
import warnings
import time
import psutil

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

# Set BASE_DIR relative to the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

try:
    # Process data in chunks to minimize memory usage
    chunk_size = 5000
    chunks = []
    for chunk in pd.read_csv(os.path.join(DATA_DIR, 'transaction_data.csv'), chunksize=chunk_size):
        chunk['Date'] = pd.to_datetime(chunk['Date'], errors='coerce')
        chunk['Hour'] = chunk['Date'].dt.hour
        chunk['DayOfWeek'] = chunk['Date'].dt.dayofweek
        chunk['TotalSpent'] = chunk['Amount'] * chunk['Frequency']  # Calculated as Quantity * UnitPrice
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    print(f"Data loaded. Shape: {df.shape}")
    print(f"Sampled Customer IDs: {df['Customer ID'].unique()[:5]}")  # Show first 5 IDs for reference

    # Feature Engineering and Encoding
    features = ['Account Type', 'Country', 'Transaction Type', 'Amount', 'Currency',
                'Merchant Category', 'Frequency', 'Channel', 'Hour', 'DayOfWeek', 'TotalSpent']
    df_encoded = pd.get_dummies(df[features], drop_first=True,
                                columns=['Account Type', 'Country', 'Transaction Type', 'Currency', 'Channel',
                                         'Merchant Category'])
    scaler = StandardScaler()
    numerical_cols = ['Amount', 'Frequency', 'Hour', 'DayOfWeek', 'TotalSpent']
    df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
    print(f"Encoded feature set shape: {df_encoded.shape}")

    # Dimensionality Reduction with PCA
    pca = PCA(n_components=0.80)
    df_pca = pca.fit_transform(df_encoded)
    print(
        f"PCA reduced to {df_pca.shape} components explaining {pca.explained_variance_ratio_.sum() * 100:.2f}% variance")

    # Cluster and detect anomalies per customer
    results = []
    start_time = time.time()
    memory_before = psutil.virtual_memory().percent
    print(f"Memory usage before processing: {memory_before}%")
    for customer_id, group in df.groupby('Customer ID'):
        if len(group) > 1:  # Skip customers with single transaction
            group_encoded = pd.get_dummies(group[features], drop_first=True,
                                           columns=['Account Type', 'Country', 'Transaction Type', 'Currency',
                                                    'Channel', 'Merchant Category'])
            group_encoded = group_encoded.reindex(columns=df_encoded.columns, fill_value=0)
            group_encoded[numerical_cols] = scaler.transform(group_encoded[numerical_cols])
            group_pca = pca.transform(group_encoded)

            # Clustering with MiniBatchKMeans
            kmeans = MiniBatchKMeans(n_clusters=min(2, len(group)), batch_size=500, random_state=42)
            cluster_labels = kmeans.fit_predict(group_pca)

            # Anomaly detection with IsolationForest
            iso_forest = IsolationForest(contamination=0.02, random_state=42)  # Matches notebook's initial value
            iso_labels = iso_forest.fit_predict(group_pca)
            anomalies = iso_labels == -1

            distances = np.min(kmeans.transform(group_pca), axis=1)  # Distance to nearest cluster center
            results.append(pd.DataFrame({
                'Customer ID': customer_id,
                'Transaction ID': group['Transaction ID'],
                'Cluster': cluster_labels,
                'IsAnomaly': anomalies,
                'Distance': distances
            }))
    df_results = pd.concat(results, ignore_index=True)
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    memory_after = psutil.virtual_memory().percent
    print(f"Memory usage after processing: {memory_after}%")

    # Save results
    start_save = time.time()
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'customer_anomalies.csv'), index=False)
    print(
        f"Results saved in {time.time() - start_save:.2f} seconds to {os.path.join(OUTPUT_DIR, 'customer_anomalies.csv')}")


    # Function to query customer anomalies
    def get_customer_anomalies(customer_id):
        customer_data = df_results[df_results['Customer ID'] == customer_id]
        if not customer_data.empty:
            anomalies = customer_data[customer_data['IsAnomaly']]
            cluster = customer_data['Cluster'].iloc[0]
            print(f"Customer ID: {customer_id}")
            print(f"Cluster: {cluster}")
            if not anomalies.empty:
                print("Anomalies:")
                for _, row in anomalies.iterrows():
                    print(f"  Transaction ID: {row['Transaction ID']}, Distance: {row['Distance']:.2f}")
            else:
                print("No anomalies detected.")
        else:
            print(f"No data found for Customer ID: {customer_id}")


    # Example usage with a sampled ID
    sampled_ids = df['Customer ID'].unique()
    if len(sampled_ids) > 0:
        get_customer_anomalies(sampled_ids[0])  # Use first sampled ID
    else:
        print("No customers in dataset.")

except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Script execution completed.")