# Early Warning System (EWS) Project

The Early Warning System (EWS) is a financial risk monitoring tool that provides a Dash-based UI and Flask APIs to calculate and predict 28 financial risk components, aligned with regulatory standards such as Basel III, IFRS 9, CCAR, FRTB, FATF, CRR, Dodd-Frank, TLAC, and FCA. The application includes dashboards for visualizing risk metrics and an AI Analytics tab for testing machine learning (ML) model predictions. It uses synthetic datasets and pre-trained ML models, with interactive forms for real-time predictions.

## Project Structure
```
D:\EWS-app\
├── data/
│   ├── credit_risk_dataset.csv        # Synthetic Credit Risk Dataset
│   ├── liquidity_risk_dataset.csv     # Synthetic Liquidity Risk Dataset
│   ├── market_risk_dataset.csv        # Synthetic Market Risk Dataset
│   ├── capital_compliance_dataset.csv # Synthetic Capital and Compliance Dataset
├── models/
│   ├── ml_models.pkl                 # Pre-trained ML models
├── src/
│   ├── generate_datasets.py          # Script to generate synthetic datasets
│   ├── train_ml_models.py           # Script to train ML models
│   ├── ews_app.py                   # Dash app with Flask APIs
├── requirements.txt                  # Python dependencies
├── README.md                        # This file
```

## Prerequisites
- **Operating System**: Windows (project directory at `D:\EWS-app`)
- **Python**: Version 3.9 or higher
- **Dependencies**: Listed in `requirements.txt`:


## Setup Instructions

1. **Create Project Directory**:
	- Extract the files and place it in above structure

2. **Install Dependencies**:
   - Open a command prompt and navigate to the project directory:
     ```bash
     cd EWS-app
     ```
   - Install the required Python packages:
     ```bash
     pip install -r requirements.txt
     ```
   - **Note**: If `QuantLib` installation fails, install a C++ compiler (e.g., Microsoft Visual C++ Build Tools) or use a pre-built wheel from a trusted source.

3. **Generate Synthetic Datasets**:
   - Run the dataset generation script to create the required CSV files:
     ```bash
     python src/generate_datasets.py
     ```
   - This generates four CSV files in `data/`:
     - `credit_risk_dataset.csv`
     - `liquidity_risk_dataset.csv`
     - `market_risk_dataset.csv`
     - `capital_compliance_dataset.csv`
   - Each file contains 20,000 rows of synthetic data for credit risk, liquidity risk, market risk, and capital/compliance metrics, respectively .

4. **Train Machine Learning Models**:
   - Run the ML model training script to generate the pre-trained models:
     ```bash
     python src/train_ml_models.py
     ```
   - This creates `ml_models.pkl` in `models/`, containing 21 trained ML models for predicting risk metrics (e.g., NPL Ratio, PCR, ECL, etc.).

5. **Run the EWS Application**:
   - Start the Dash app with Flask APIs:
     ```bash
     python src/ews_app.py
     ```
   - The application will run on `http://0.0.0.0:8000`. Open this URL in a web browser to access the UI.
   - **Note**: If port 8000 is in use, modify the port in `ews_app.py` (e.g., change `flask_app.run(host='0.0.0.0', port=8000)` to `port=8050`).

## Using the Application

### Dash UI
The UI has two tabs:

1. **Dashboards Tab**:
   - Displays 28 financial risk components organized into four categories: Credit Risk, Liquidity Risk, Market Risk, and Capital & Compliance.
   - Each component shows:
     - A gauge chart (using Plotly) with the calculated value.
     - The metric value (e.g., "NPL Ratio: 6.50%").
     - A status (e.g., "High Risk" for NPL Ratio > 5%, color-coded green for compliant, red for non-compliant).
     - An interpretation (e.g., "NPL Ratio > 5% indicates high credit risk (Basel III).").
   - Components are styled with Tailwind CSS for a modern, responsive look.

2. **AI Analytics Tab**:
   - Lists 21 ML models for predicting risk metrics (e.g., NPL Ratio Prediction, PCR Prediction).
   - Each model has an interactive form with:
     - A textarea for entering JSON input data.
     - A "Predict" button to trigger the prediction.
     - Output fields showing the predicted value and status (e.g., "Predicted NPL Ratio: 6.50%, Status: High Risk").
   - See the "Testing AI Analytics" section below for JSON input examples.

### Flask APIs
The application provides two types of APIs:
- **Calculation APIs**: Compute risk metrics using formulas (e.g., `/api/npl_ratio`).
- **Prediction APIs**: Use ML models to predict risk metrics (e.g., `/api/predict/npl_ratio`).
- Test APIs using `curl` or Postman. Example for NPL Ratio prediction:
  ```bash
  curl -X POST "http://0.0.0.0:8000/api/predict/npl_ratio" -H "Content-Type: application/json" -d '{"data": [{"Principal_Amount": 100000, "Days_Past_Due": 100, "Credit_Score": 500}]}'
  ```
  Expected response: `{"prediction": <value>, "status": "<status>", "interpretation": "Predicted NPL Ratio > 5% indicates high credit risk."}`

## Testing AI Analytics

To test the 21 ML models in the AI Analytics tab, enter the following JSON data into the respective textarea fields and click "Predict". The JSON inputs are tailored to the expected schema for each model, ensuring compatibility with the ML models in `ml_models.pkl`.

1. **NPL Ratio Prediction**:
   ```json
   [{"Principal_Amount": 100000, "Days_Past_Due": 100, "Credit_Score": 500}]
   ```

2. **PCR Prediction**:
   ```json
   [{"Provision_Amount": 10000, "Principal_Amount": 100000, "Days_Past_Due": 100}]
   ```

3. **ECL Prediction**:
   ```json
   [{"Credit_Score": 500, "Collateral_Value": 50000, "Principal_Amount": 100000, "Commitment_Amount": 20000}]
   ```

4. **PD Prediction**:
   ```json
   [{"Credit_Score": 500, "Days_Past_Due": 100}]
   ```

5. **LGD Prediction**:
   ```json
   [{"Collateral_Value": 50000, "Principal_Amount": 100000}]
   ```

6. **EAD Prediction**:
   ```json
   [{"Principal_Amount": 100000, "Commitment_Amount": 20000}]
   ```

7. **IFRS 9 Staging**:
   ```json
   [{"Credit_Score": 500, "Days_Past_Due": 100}]
   ```

8. **LCR Forecast**:
   ```json
   [{"Asset_Value": 500000, "Outflow_Amount": 400000, "ASF_Weight": 0.8}]
   ```

9. **NSFR Forecast**:
   ```json
   [{"Funding_Amount": 600000, "Asset_Value": 500000, "ASF_Weight": 0.8, "RSF_Weight": 0.5}]
   ```

10. **VaR Estimation**:
    ```json
    [{"Quantity": 1000, "Closing_Price": 50, "Return_Volatility": 0.02}]
    ```

11. **ES Estimation**:
    ```json
    [{"Quantity": 1000, "Closing_Price": 50, "Return_Volatility": 0.02}]
    ```

12. **Tier 1 Ratio Prediction**:
    ```json
    [{"Capital_Amount": 1000000, "Deduction_Amount": 50000, "Exposure_Amount": 5000000, "Risk_Weight": 0.8}]
    ```

13. **CET1 Ratio Prediction**:
    ```json
    [{"Capital_Amount": 800000, "Deduction_Amount": 40000, "Exposure_Amount": 5000000, "Risk_Weight": 0.8}]
    ```

14. **AML Detection**:
    ```json
    [{"Transaction_Amount": 200000, "Risk_Score": 0.9}]
    ```

15. **LIBOR Exposure**:
    ```json
    [{"Transaction_Amount": 200000, "Rate_Type": "LIBOR"}]
    ```

16. **SCB Prediction**:
    ```json
    [{"Capital_Amount": 1000000, "Loss_Amount": 500000, "Stress_Scenario_ID": "S1"}]
    ```

17. **CCAR Readiness**:
    ```json
    [{"Capital_Amount": 1000000, "Loss_Amount": 200000, "Risk_Score": 0.7}]
    ```

18. **Basel III Readiness**:
    ```json
    [{"Capital_Amount": 1000000, "Loss_Amount": 200000, "Risk_Score": 0.7}]
    ```

19. **Compliance Score**:
    ```json
    [{"Capital_Amount": 1000000, "Loss_Amount": 200000, "Risk_Score": 0.7}]
    ```

20. **Operational RWA**:
    ```json
    [{"Revenue_Amount": 1000000, "Loss_Amount": 50000}]
    ```

21. **Composite Risk Index**:
    ```json
    [{"Capital_Amount": 1000000, "Loss_Amount": 200000, "Risk_Score": 0.7}]
    ```

### Testing Instructions
- **In the UI**:
  1. Open `http://0.0.0.0:8000` in a browser.
  2. Navigate to the **AI Analytics** tab.
  3. For each model (e.g., "NPL Ratio Prediction"), copy the corresponding JSON data into the textarea.
  4. Click the "Predict" button to view the predicted value and status (e.g., "Predicted NPL Ratio: 6.50%, Status: High Risk").
  5. If an error appears (e.g., "Error: Invalid JSON"), verify the JSON syntax using a tool like `jsonlint.com`.

- **Via API**:
  1. Use `curl` or Postman to send a POST request to the prediction endpoint. Example:
     ```bash
     curl -X POST "http://0.0.0.0:8000/api/predict/npl_ratio" -H "Content-Type: application/json" -d '{"data": [{"Principal_Amount": 100000, "Days_Past_Due": 100, "Credit_Score": 500}]}'
     ```
  2. Check the response for the predicted value, status, and interpretation.

