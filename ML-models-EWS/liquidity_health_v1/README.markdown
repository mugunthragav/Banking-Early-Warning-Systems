Liquidity & Stability Dashboard and API
Welcome to the Liquidity & Stability project! This repository provides two implementations for analyzing financial metrics: an interactive web dashboard (via Dash) and a RESTful API (via FastAPI). Both implementations offer insights into liquidity and stability indicators such as LCR (Liquidity Coverage Ratio) and NSFR (Net Stable Funding Ratio), enabling users to monitor, forecast, and stress-test key metrics.

Dashboard: A user-friendly web application for visualizing historical data, forecasts, and stress simulations interactively.
API: A programmatic interface for accessing the same functionality via HTTP endpoints, ideal for integration into other systems.

Built with a focus on scalability, reliability, and ease of use, this project serves as a powerful tool for monitoring, risk assessment, and strategic planning.
Overview:
The project provides the following core features:

Access to historical LCR and NSFR data for the last year.
Dynamic forecasting of LCR and NSFR using ARIMA/LSTM models, with range-based outputs (min, max, avg).
Stress testing with adjustable parameters for HQLA, outflows, inflows, stable funding, and required funding.
Exportable alerts for low LCR scenarios in CSV format.

The Dashboard offers interactive plots and tables for visual analysis, while the API provides JSON responses for programmatic access.
Prerequisites:

Python 3.8 or higher
Required Python packages (install via requirements.txt)

Installation:

Clone the repository:git clone <repository-url>
cd liquidity_health_v1


Create a virtual environment and activate it:python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install the dependencies:
For the Dashboard, ensure Dash and its dependencies are installed:pip install -r requirements.txt


For the API, additionally install FastAPI and Uvicorn:pip install fastapi uvicorn





Data Preparation:

Place your transaction data in the data directory as transaction_data.csv. Ensure it includes columns like date, hqla_value, outflows, inflows, required_funding, stable_funding, account_id, and division with appropriate date formats (e.g., %Y-%m-%d %H:%M:%S).
The dataset should ideally span sufficient historical data (e.g., months or years) to support rolling windows (30 days for LCR, 365 days for NSFR) and forecasting models.

Training Models:
Since pre-trained model files (.pkl) are not provided, you need to train the models before running either the dashboard or the API:

Navigate to the src/core directory:cd src/core


Run the training script for LCR models:python lcr_models.py


This will generate best_lcr_model.pkl in the models directory.


Run the training script for NSFR models:python nsfr_models.py


This will generate best_nsfr_model.pkl in the models directory.


Note: Ensure your transaction_data.csv is populated with sufficient data for training (e.g., at least 365 days for NSFR calculations).

Running the Dashboard (app.py)
The Dash-based dashboard provides an interactive web interface for exploring liquidity and stability metrics.

Launch the Dashboard:

Return to the root directory:cd ../..


Start the Dash server:python src/dashboard/app.py


Open your browser and go to http://localhost:8050 to access the dashboard.


Usage:

Tabs Navigation: Switch between LCR and NSFR analyses using the dashboard tabs.
Forecasting:
Adjust the forecast period input and click "Update Forecast" to view dynamic projections.
Forecasts show a range (min to max) and average value over the specified period.


Stress Testing:
In the "Stress Simulation" section, enter adjustments for HQLA, outflows, inflows, stable funding, and required funding (e.g., (-5000, 200, 0, -5000, 5000)).
Click "Run Simulation" to see the impact on LCR (for 30, 90, 180 days) and NSFR (for 365 days), displayed as a range and average (e.g., "LCR/NSFR ≈ 134.70% to 193.74% (avg: 173.75%)").


Export Alerts: Use the "Export Alerts" button to download LCR alerts as a CSV file.
Exploration: Explore interactive plots and data tables to analyze historical trends, forecasts, and stress test results.



Running the API (api.py)
The FastAPI-based API provides programmatic access to the same functionality via HTTP endpoints.

Launch the API:

Return to the root directory:cd ../..


Start the FastAPI server:python src/dashboard/api.py


The API will be available at http://localhost:8000. Access the interactive API documentation at http://localhost:8000/docs.


API Endpoints:
LCR Analysis

GET /lcr/historical

Retrieve historical LCR data for the last year.
Response: JSON array of {date, lcr} entries.
Example: curl http://localhost:8000/lcr/historical


GET /lcr/forecast?periods=

Generate LCR forecast for a specified number of periods (1 period = 30 days).
Query Parameter: periods (default: 1)
Response: JSON with forecast data and range (min, max, avg).
Example: curl http://localhost:8000/lcr/forecast?periods=2


GET /lcr/low-liquidity-accounts

Get accounts with low LCR (<100%).
Response: JSON array of {account_id, low_lcr_count} entries.
Example: curl http://localhost:8000/lcr/low-liquidity-accounts


GET /lcr/low-liquidity-divisions

Get divisions with low LCR (<100%).
Response: JSON array of {division, low_lcr_count} entries.
Example: curl http://localhost:8000/lcr/low-liquidity-divisions


GET /lcr/alerts

Export LCR alerts as CSV for periods where LCR < 100%.
Response: CSV file with columns Date, Account ID, Division, LCR (%), Recommendation.
Example: curl http://localhost:8000/lcr/alerts



NSFR Analysis:

GET /nsfr/historical

Retrieve historical NSFR data for the last year.
Response: JSON array of {date, nsfr} entries.
Example: curl http://localhost:8000/nsfr/historical


GET /nsfr/forecast?days=

Generate NSFR forecast for a specified number of days.
Query Parameter: days (default: 365)
Response: JSON with forecast data and range (min, max, avg).
Example: curl http://localhost:8000/nsfr/forecast?days=730



Stress Simulation

POST /stress-simulation
Run a stress simulation with adjustments for HQLA, outflows, inflows, stable funding, and required funding.
Request Body (JSON):{
  "hqla_adjustment": -5000.0,
  "outflow_adjustment": 200.0,
  "inflow_adjustment": 0.0,
  "stable_funding_adjustment": -5000.0,
  "required_funding_adjustment": 5000.0
}


Response: JSON with simulation results for 30, 90, 180, and 365 days, including min, max, avg, and full series.
Example:curl -X POST http://localhost:8000/stress-simulation \
  -H "Content-Type: application/json" \
  -d '{"hqla_adjustment": -5000, "outflow_adjustment": 200, "inflow_adjustment": 0, "stable_funding_adjustment": -5000, "required_funding_adjustment": 5000}'







Usage Notes

Dashboard: Ideal for users who prefer a visual interface to explore data interactively. Requires Dash dependencies.
API: Suitable for programmatic access and integration into other systems. Requires FastAPI and Uvicorn.
The project handles large datasets (e.g., 18,264 rows spanning 1975-04-20 to 2025-04-20) and provides robust forecasting and stress testing capabilities.
All numerical values in API responses are rounded to 2 decimal places and converted to native Python types for seamless JSON serialization.

