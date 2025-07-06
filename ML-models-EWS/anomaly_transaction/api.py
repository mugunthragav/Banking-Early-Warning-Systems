from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import logging
import sys
import tensorflow as tf
from typing import List, Dict, Any, Union

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Initialize FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting transaction anomalies using Isolation Forest and Autoencoder",
    version="1.0.0"
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Output directory: {OUTPUT_DIR}")

# Check if output directory exists
if not os.path.exists(OUTPUT_DIR):
    logger.error(f"Output directory {OUTPUT_DIR} does not exist")
    raise FileNotFoundError(f"Output directory {OUTPUT_DIR} does not exist")

# Load models and scaler
try:
    logger.info("Loading iso_forest.pkl")
    with open(os.path.join(OUTPUT_DIR, 'iso_forest.pkl'), 'rb') as f:
        iso_forest = pickle.load(f)
    logger.info("Loading autoencoder.pkl")
    with open(os.path.join(OUTPUT_DIR, 'autoencoder.pkl'), 'rb') as f:
        autoencoder = pickle.load(f)
    logger.info("Loading scaler.pkl")
    with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    logger.info("Successfully loaded models and scaler")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise

# Load data
try:
    logger.info("Loading anomaly_results.csv")
    anomaly_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'anomaly_results.csv'))
    logger.info("Loading alert_data.csv")
    alert_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'alert_data.csv'))
    logger.info("Loading common_anomalies.csv")
    common_anomalies = pd.read_csv(os.path.join(OUTPUT_DIR, 'common_anomalies.csv'))
    logger.info(f"Anomaly data shape: {anomaly_data.shape}")
    logger.info(f"Available CustomerIDs (first 10): {anomaly_data['CustomerID'].unique()[:10].tolist()}")
except FileNotFoundError as e:
    logger.error(f"File not found: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading data: {e}")
    raise

# Pydantic models
class AnomalyRequest(BaseModel):
    customer_id: Union[str, float, int]  # Accept string, float, or int
    filter_metric: str

class AnomalyItem(BaseModel):
    InvoiceNo: str
    metric_value: float
    RiskScore: float

class AnomalyResponse(BaseModel):
    anomaly_list: List[AnomalyItem]
    chart_data: Dict[str, List[Any]]
    drill_down_data: List[Dict[str, Any]]

# Root endpoint
@app.get("/")
async def home():
    return {
        "message": "Welcome to the Fraud Detection API",
        "endpoint": "/api/anomalies (POST)",
        "example": {
            "customer_id": str(anomaly_data['CustomerID'].iloc[0]),
            "filter_metric": "TotalSpent"
        },
        "valid_metrics": ["TotalSpent", "HistFreq", "UnusualDestination", "QuantitySpike"],
        "docs": "/docs",
        "customer_ids_endpoint": "/customer_ids (GET)"
    }

# Favicon endpoint
@app.get("/favicon.ico")
async def favicon():
    return PlainTextResponse("", status_code=204)

# CustomerIDs endpoint
@app.get("/customer_ids")
async def get_customer_ids():
    customer_ids = anomaly_data['CustomerID'].unique().tolist()
    return {
        "total_customer_ids": len(customer_ids),
        "customer_ids": customer_ids[:50],  # Limit to 50 for response size
        "note": "Full list available in output/anomaly_results.csv"
    }

# API endpoint
@app.post("/api/anomalies", response_model=AnomalyResponse)
async def get_anomalies(request: AnomalyRequest):
    try:
        # Convert customer_id to float, handling strings or numbers
        try:
            customer_id = float(str(request.customer_id).strip())
        except ValueError:
            logger.warning(f"Invalid customer_id format: {request.customer_id}")
            raise HTTPException(status_code=400, detail="Invalid customer_id format. Must be a number.")

        filter_metric = request.filter_metric
        logger.info(f"Received request: customer_id={customer_id}, filter_metric={filter_metric}")

        # Validate inputs
        if not customer_id or not filter_metric:
            logger.warning("Missing customer_id or filter_metric")
            raise HTTPException(status_code=400, detail="Missing customer_id or filter_metric")
        if customer_id not in anomaly_data['CustomerID'].values:
            valid_ids = anomaly_data['CustomerID'].unique()[:10].tolist()
            logger.warning(f"CustomerID {customer_id} not found")
            raise HTTPException(
                status_code=404,
                detail=f"CustomerID {customer_id} not found. Valid IDs (first 10): {valid_ids}, Total valid IDs: {len(anomaly_data['CustomerID'].unique())}. See /customer_ids for more."
            )
        valid_metrics = ['TotalSpent', 'HistFreq', 'UnusualDestination', 'QuantitySpike']
        if filter_metric not in valid_metrics:
            logger.warning(f"Invalid filter_metric: {filter_metric}")
            raise HTTPException(status_code=400, detail=f"Invalid filter_metric. Must be one of {valid_metrics}")

        # Filter data by customer
        filtered_data = anomaly_data[anomaly_data['CustomerID'] == customer_id]

        # Check if filtered data is empty
        if filtered_data.empty:
            logger.warning(f"No data found for CustomerID {customer_id} in anomaly_data")
            raise HTTPException(status_code=404, detail=f"No anomaly data found for CustomerID {customer_id}")

        # Anomaly list
        sorted_anomalies = filtered_data.sort_values(by=filter_metric, ascending=False).head(10)
        anomaly_list = [
            AnomalyItem(
                InvoiceNo=str(row['InvoiceNo']),  # Ensure string
                metric_value=round(float(row[filter_metric]), 2),  # Ensure float
                RiskScore=round(float(row['RiskScore']), 2)  # Ensure float
            )
            for _, row in sorted_anomalies.iterrows()
        ]

        # Chart data
        chart_data = {
            'InvoiceNo': sorted_anomalies['InvoiceNo'].astype(str).tolist(),
            filter_metric: sorted_anomalies[filter_metric].round(2).tolist(),
            'RiskScore': sorted_anomalies['RiskScore'].round(2).tolist()
        }

        # Drill-down table
        drill_down_data = alert_data[alert_data['CustomerID'] == customer_id].to_dict('records')

        # Response
        response = AnomalyResponse(
            anomaly_list=anomaly_list,
            chart_data=chart_data,
            drill_down_data=drill_down_data
        )
        logger.info(f"Processed request for CustomerID: {customer_id}, Metric: {filter_metric}")
        return response

    except HTTPException as e:
        raise e  # Re-raise HTTPException to ensure correct status code
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == '__main__':
    import uvicorn
    logger.info("Starting FastAPI server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)