Commodity Trading Model Project
This repository contains a sophisticated machine learning-based risk management system for commodity trading, leveraging historical and synthetic price data to predict Value at Risk (VaR), perform stress testing, and visualize results via a Dash dashboard. The model supports multiple commodities (e.g., @NG for natural gas, @CL#C for crude oil) and is extensible for real-time data integration.
Overview
The project implements a HistGradientBoostingClassifier to identify feature importance, employs Monte Carlo simulations for VaR estimation, and conducts stress testing under predefined market scenarios. Results are stored in a JSON file (models/all_commodities_model.json) and visualized interactively through a Dash dashboard and a FastAPI-based API.
Features

Feature importance analysis using permutation importance.
Monte Carlo and historical VaR calculations with configurable confidence levels (95%, 99%) and horizons (1, 10 days).
Stress testing for scenarios like 2008 Crash, 1987 Crash, COVID Drop, and Rate Hike.
Scalable batch processing for multiple commodities.
Dash-based web dashboard for real-time visualization.
FastAPI-based API for programmatic access to risk metrics, visualizations, and reports.

Installation
1. Clone the Repository
git clone https://github.com/yourusername/green-labs.git
cd market_health

2. Set Up Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt


Execution
1. Run the Model
python -m src.main --mode batch_strategy


Options:
--mode: strategy for a single symbol, batch_strategy for multiple symbols, schedule for scheduled runs.
--synthetic: Use synthetic data instead of historical data.


Output: Saves results to models/all_commodities_model.json.

2. Launch the Dashboard
python app.py


Access at http://127.0.0.1:8050/. Use the dropdown to select commodities and view interactive visualizations.

3. Launch the API
python api.py


Access at http://0.0.0.0:8000. Interactive API documentation is available at http://0.0.0.0:8000/docs.
The API serves risk metrics, visualizations, and reports generated from models/all_commodities_model.json.

API Integration and Implementation
The project includes a FastAPI-based API (api.py) that provides programmatic access to the market risk metrics and visualizations available in the Dash dashboard. The API is designed to serve data from the models/all_commodities_model.json file and generate PDF reports using the report.py module.
API Setup

Ensure Dependencies: Install  requirements.txt by pip install -r requirements


File Placement: Place api.py in the project root alongside app.py and report.py. Ensure models/all_commodities_model.json exists and is populated by running the model (python -m src.main --mode batch_strategy).
Run the API:python api.py


The API runs on http://0.0.0.0:8000 by default.
Use the interactive Swagger UI at http://0.0.0.0:8000/docs to test endpoints.



API Endpoints
The API provides the following endpoints, mirroring the Dash dashboard's functionality:

GET /symbols

Description: Returns a list of available symbols and their sector descriptions.
Response: JSON array of objects with symbol and description fields.
Example:[
    {"symbol": "@CL#C", "description": "Energy"},
    {"symbol": "@ES", "description": "Equity Index"}
]




GET /metrics/{symbol}

Description: Retrieves all risk metrics for a specified symbol, including portfolio VaR, feature importance, VaR metrics, stress test metrics, backtest metrics, and a formatted text output.
Response: JSON object containing symbol details, metrics, and text output.
Example:{
    "symbol": "@CL#C",
    "description": "Energy",
    "portfolio_var": 1234.56,
    "feature_importance": [{"feature": "Volatility", "importance": 0.35}, ...],
    "var_metrics": [{"method": "mc_var_0.95_1d", "var": 1000.0, ...}, ...],
    "stress_metrics": [{"scenario": "2008_Crash", "loss": 2000.0, ...}, ...],
    "backtest_metrics": [{"metric": "mc_var_0.95_1d", "exceedances": 5, ...}, ...],
    "text_output": "=== Feature Importance ===\nVolatility: 0.3500\n..."
}




GET /visualizations/{symbol}

Description: Returns Plotly JSON for feature importance and Monte Carlo VaR distribution visualizations, suitable for rendering in a frontend.
Response: JSON object with Plotly figure JSON for feature importance and VaR distribution.
Example:{
    "feature_importance": {"data": [...], "layout": {...}},
    "mc_var_distribution": {"data": [...], "layout": {...}}
}




GET /heatmap/{symbol}

Description: Returns heatmap data and Plotly JSON for the risk exposure heatmap, combining VaR and stress test losses.
Response: JSON object with symbol details, heatmap data, and Plotly JSON.
Example:{
    "symbol": "@CL#C",
    "description": "Energy",
    "heatmap_data": [{"scenario_metric": "mc_var_0.95_1d", "value": 1000.0}, ...],
    "plotly_json": {"data": [...], "layout": {...}}
}




GET /report/{symbol}

Description: Generates and downloads a PDF report for the specified symbol, using the generate_report function from report.py.
Response: File response with the generated PDF.
Example: Downloads a file named {symbol}_report.pdf.



API Integration

Frontend Integration: Use the /visualizations/{symbol} and /heatmap/{symbol} endpoints to render Plotly charts in a web application. The /metrics/{symbol} endpoint provides data for custom tables or displays.
Backend Integration: Call /metrics/{symbol} or /report/{symbol} from other services to retrieve risk metrics or generate reports programmatically.
Error Handling: The API includes robust error handling, logging issues to logs/market_risk.log. Check logs for debugging.
Security: For production, add authentication (e.g., OAuth2) and rate limiting. Modify api.py to include middleware as needed.
Scalability: The API serves static data from models/all_commodities_model.json. For large datasets, consider caching or migrating to a database.
