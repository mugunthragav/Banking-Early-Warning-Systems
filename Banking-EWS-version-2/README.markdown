# Combined Risk API

This repository contains a **Combined Risk API** built with FastAPI, integrating Credit, Liquidity, and Market risk analysis for financial institutions. The API provides endpoints for risk predictions, metrics, visualizations, and reports, serving as a backend for risk management applications.



## Prerequisites

- **Python 3.11+**: Download from [python.org](https://www.python.org/downloads/).
- **Virtualenv**: For isolating dependencies.
- **Git**: For cloning or managing the repository (optional).
- **Windows**: Tested on Windows at `D:\early_warning_loyola\combined_risk_api`.
- **Model Files**: Ensure all model files in `models\` are present.

## Setup Instructions

### 1. Navigate to the Project Directory
 If in a Git repository, clone it:
```bash
git clone <repository-url>
cd D:\early_warning_loyola\combined_risk_api
```

### 2. Set Up the Environment

#### a. Create and Activate a Virtual Environment
```bash
cd D:\early_warning_loyola\combined_risk_api
python -m venv venv
venv\Scripts\activate
```

#### b. Install Python Dependencies
Install required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install core dependencies:
```bash
pip install fastapi uvicorn pandas numpy xgboost joblib shap plotly pydantic
```

For report generation (if `report.py` uses PDF libraries):
```bash
pip install weasyprint reportlab
```

#### c. Verify Model Files
Ensure the following exist in `D:\early_warning_loyola\combined_risk_api\models\`:
- `final_ensembled_model.json`
- `final_ensembled_model_v2.pkl`
- `final_ensembled_model.pkl`
- `scaler_correct_new.pkl`
- `all_commodities_model.json`

See [Troubleshooting](#troubleshooting) if any are missing.

#### d. Configure Logging
The API logs to `logs\` and the console. Ensure `logs\` is writable. If console encoding issues occur (e.g., `UnicodeEncodeError`), see [Troubleshooting](#troubleshooting).

### 3. Run the API
From the project root with the virtual environment activated:
```bash
python main.py
```

The API starts on `http://0.0.0.0:8000`. Access the interactive API docs at `http://localhost:8000/docs`.

## API Endpoints

### Health Check
- **GET /health**: Check API status.
  ```bash
  curl http://localhost:8000/health
  ```
  Expected response:
  ```json
  {
    "status": "healthy",
    "components": {
      "credit": "operational (mocked)",
      "liquidity": "operational",
      "market": "operational"
    }
  }
  ```

### Liquidity Risk
- **GET /liquidity/health**: Check model loading status.
  ```bash
  curl http://localhost:8000/liquidity/health
  ```
  Expected response:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "scaler_loaded": true,
    "shap_available": true,
    "xgboost_version": "2.0.3"
  }
  ```
- **POST /liquidity/predict**: Predict liquidity risk.
  ```bash
  curl -X POST http://localhost:8000/liquidity/predict -H "Content-Type: application/json" -d @payload.json
  ```
  Example `payload.json`:
  ```json
  {
    "13_CASH": 1000000,
    "Treasury_bills": 500000,
    "Labels_Liquid": 200000,
    "Curr_Deposit": 1500000,
    "Fixed_Deposit": 1000000,
    "General_Savings": 800000,
    "Borrowing_Borrow": 300000,
    "Balance_Interbank": 400000,
    "Warehouse_Flag": 100000,
    "earnings_Capital": 1000000,
    "earnings_Gross_Loans": 2000000,
    "Commercial_Flag": 50000,
    "Liabilities_Income": 5000000,
    "Deposit_Trend": 0.02,
    "Funding_Rate": 0.01,
    "Institution_Models": "Bank1,Bank2",
    "earnings_amounts": "100000,150000"
  }
  ```

### Market Risk
- **GET /market/symbols**: List available symbols.
  ```bash
  curl http://localhost:8000/market/symbols
  ```
- **GET /market/metrics/{symbol}**: Get risk metrics (e.g., `@CL#C`).
  ```bash
  curl http://localhost:8000/market/metrics/@CL#C
  ```
- **GET /market/visualizations/{symbol}**: Get visualizations.
  ```bash
  curl http://localhost:8000/market/visualizations/@CL#C
  ```
- **GET /market/heatmap/{symbol}**: Get risk exposure heatmap.
  ```bash
  curl http://localhost:8000/market/heatmap/@CL#C
  ```
- **GET /market/report/{symbol}**: Download a PDF report.
  ```bash
  curl http://localhost:8000/market/report/@CL#C -o cl_report.pdf
  ```

### Credit Risk (Mocked)
- **POST /credit/predict/batch_json**: Predict credit risk.
  ```bash
  curl -X POST http://localhost:8000/credit/predict/batch_json -H "Content-Type: application/json" -d '{"applications":[{"loan_id":"123","amount":10000},{"loan_id":"456","amount":20000}]}'
  ```
  Expected response:
  ```json
  {
    "predictions": [
      {"loan_id": "123", "risk_score": 0.75, "approval_status": "Approved"},
      {"loan_id": "456", "risk_score": 0.85, "approval_status": "Denied"}
    ]
  }
  ```

## Troubleshooting

### API Issues
- **UnicodeEncodeError in Logs**:
  - If logs show `UnicodeEncodeError` for characters like `✅`, ensure `liquidity_router.py` uses ASCII-friendly messages (e.g., `[SUCCESS]`). Update log handlers:
    ```python
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    try:
        handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    ```
    Replace in `D:\early_warning_loyola\combined_risk_api\routers\liquidity_router.py`.
- **Model Loading Errors**:
  - Verify model paths in `liquidity_router.py` and `market_router.py`. Create mock files if needed:
    - For `all_commodities_model.json`:
      ```json
      {
        "@CL#C": {
          "mdi_imp": {"feature1": 0.5, "feature2": 0.3},
          "monte_carlo_var": {"mc_var_0.95_1d": {"var": 1000, "revenue_impact": 500}},
          "historical_var": {},
          "stress_results": {"2008_Crash": {"loss": 2000}},
          "backtest_results": {},
          "portfolio_var": 1500
        }
      }
      ```
      Save as `D:\early_warning_loyola\combined_risk_api\models\all_commodities_model.json`.
    - For `final_ensembled_model.json`:
      ```bash
      python -c "import joblib, xgboost; model = joblib.load('D:\\early_warning_loyola\\combined_risk_api\\models\\final_ensembled_model.pkl'); model.save_model('D:\\early_warning_loyola\\combined_risk_api\\models\\final_ensembled_model.json')"
      ```
- **SHAP Initialization Fails**:
  - If `liquidity_router.py` logs errors (e.g., `'XGBModel' has no attribute 'feature_weights'`), ensure compatible versions:
    ```bash
    pip install shap==0.45.1 xgboost==2.0.3
    ```
  - Check model attributes:
    ```bash
    python -c "import joblib; model = joblib.load('D:\\early_warning_loyola\\combined_risk_api\\models\\final_ensembled_model.pkl'); print(type(model), dir(model))"
    ```
- **Report Generation Fails**:
  - If `/market/report/{symbol}` fails, check `D:\early_warning_loyola\combined_risk_api\app\report.py`. Use a mock:
    ```python
    import os
    def generate_report(data, config):
        output_dir = config['report']['output_dir']
        symbol = list(data.keys())[0]
        report_path = os.path.join(output_dir, f"{symbol}_report.pdf")
        with open(report_path, 'wb') as f:
            f.write(b"%PDF-1.4\n% Mock PDF\n")
    ```
    Save as `D:\early_warning_loyola\combined_risk_api\app\report.py`.

### Credit Integration
- The Credit router is mocked. To integrate real functionality, locate `initial_preprocessor.py` in `D:\early_warning_loyola\creditrisk_loyola`, copy it to `D:\early_warning_loyola\combined_risk_api\app\processing\`, and update `credit_router.py`.

## Contributing

- Report issues or submit changes via the repository.
- Maintain consistent logging and error handling in new endpoints.

## License

This project is proprietary and intended for internal use at Loyola. Contact the project maintainer for licensing details.

## Contact

For support, contact the project maintainer or check the repository’s issue tracker.