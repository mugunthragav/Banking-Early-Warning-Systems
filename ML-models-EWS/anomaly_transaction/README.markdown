Anomaly Detection in E-Commerce Transactions
This project implements an anomaly detection system for the Kaggle E-Commerce Dataset, identifying potential fraudulent transactions using unsupervised learning models (Isolation Forest and Autoencoder). It includes a training script (train.py), an interactive Dash dashboard (app.py), and a FastAPI endpoint (api.py) for real-time anomaly queries.

Project Overview
Objective
Detect fraudulent e-commerce transactions (e.g., unusual purchases, transaction spikes, or unexpected locations) in a dataset spanning 01/12/2010 to 09/12/2011. The dataset includes:

Columns: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country.
Size: 401,525 transactions after cleaning, 4,372 unique CustomerID values.

Key Features

Data Preprocessing: Cleans and prepares transaction data.
Feature Engineering: Adds behavioral metrics for fraud detection.
Model Training: Uses Isolation Forest and Autoencoder to identify anomalies.
Dashboard: Interactive UI to explore anomalies.
API: Programmatic access for fraud monitoring systems.
Scalability: Supports data updates and production deployment.


How It Works
1. Data Preprocessing

Input: Loads data.csv, removes rows with missing CustomerID.
Cleaning:
Converts InvoiceDate to datetime (%m/%d/%Y %H:%M).
Drops duplicates.


Output: 401,525 rows.

2. Feature Engineering

Derived Features:
TotalSpent = Quantity * UnitPrice.
TransactionHour, TransactionDay (from InvoiceDate).


Behavioral Metrics:
HistFreq: Unique invoices per customer.
UnusualDestination: Flags transactions from unusual countries.
QuantitySpike: Flags quantity outliers (>3 standard deviations).


Risk Scoring:
RiskScore: 1.0 if ZScore > 3 or Autoencoder_Score > 0.85, else 0.0.



3. Model Training

Models:
Isolation Forest: Labels anomalies as -1 (contamination = 0.02).
Autoencoder: Flags high reconstruction errors (MSE > 98th percentile) as 1.


Training:
Uses features like TotalSpent, HistFreq, etc.
Scales features with StandardScaler.


Output:
Models: iso_forest.pkl, autoencoder.pkl, scaler.pkl.
Results: anomaly_results.csv (all data), common_anomalies.csv (4,338 rows).



4. Dashboard (app.py)

Features:
Shows 4,338 high-confidence anomalies.
Dropdowns for CustomerID and metrics (TotalSpent, HistFreq, etc.).
Displays top 10 anomalies with a bar chart and Drill-Down Table.


Access: http://127.0.0.1:8050.

5. API (api.py)

Purpose: Real-time anomaly queries for integration.
Endpoints:
GET /customer_ids: Lists valid CustomerID values (4,372 total).{"total_customer_ids": 4372, "customer_ids": [17850.0, 15311.0, ...], ...}


POST /api/anomalies: Retrieves anomalies for a CustomerID.
Request:{"customer_id": "15311", "filter_metric": "TotalSpent"}


Response:{
  "anomaly_list": [{"InvoiceNo": "536381", "metric_value": 97.75, "RiskScore": 0.0}, ...],
  "chart_data": {"InvoiceNo": ["536381", ...], "TotalSpent": [97.75, ...], ...},
  "drill_down_data": [{"CustomerID": 15311.0, "TotalSpent": 97.75, ...}, ...]
}


Metrics: TotalSpent, HistFreq, UnusualDestination, QuantitySpike.






Project Structure
anomaly_detection/
├── data/
│   └── data.csv           # Kaggle dataset
├── output/
│   ├── iso_forest.pkl     # Isolation Forest model
│   ├── autoencoder.pkl    # Autoencoder model
│   ├── scaler.pkl         # StandardScaler
│   ├── anomaly_results.csv # All transactions
│   ├── common_anomalies.csv # High-confidence anomalies
│   ├── alert_data.csv     # Dashboard/API data
├── train.py               # Training script
├── app.py                 # Dash dashboard
├── api.py                 # FastAPI endpoint
├── README.md              # Documentation
├── requirements.txt       # Dependencies
├── .venv/                 # Virtual environment


Setup

Clone the Repository:
git clone <repository-url>
cd anomaly_detection


Set Up Virtual Environment:
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux


Install Dependencies:
pip install -r requirements.txt

Ensure requirements.txt includes:
pandas
numpy
scikit-learn
tensorflow
dash
fastapi
uvicorn


Download Dataset:

Get data.csv from Kaggle.
Place in data/.




Usage
1. Train Models
python train.py


Generates .pkl and .csv files in output/.
Runtime: ~15-20 minutes.

2. Run Dashboard
python app.py


Access: http://127.0.0.1:8050.
Select a CustomerID and metric to view anomalies.

3. Run API
python api.py


Access Swagger UI: http://localhost:8000/docs.
Test with curl:curl -X POST http://localhost:8000/api/anomalies -H "Content-Type: application/json" -d '{"customer_id": "15311", "filter_metric": "TotalSpent"}'




Data Updates
To add new transactions:

Append New Data:

Add to data.csv or use a new file (update train.py if needed).


Retrain Models:
python train.py


Updates .pkl and .csv files.


Relaunch API:
python api.py


Relaunch Dashboard (if used):
python app.py




Notes

Performance: ~1-second response time.
Deployment:
API: uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4.


Integration: Use API for real-time fraud monitoring systems.
Support: Check logs in api.py or app.py for errors.

