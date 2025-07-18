# Credit Health Project

## Overview
The `credit_health` project is a credit risk analysis tool designed to predict key risk metrics for loans, including Probability of Default (PD), Loss Given Default (LGD), and Exposure at Default (EAD). It uses machine learning models trained on the LendingClub dataset (`loan_data_2007_2014.csv`) to assess individual loan risk and generate portfolio-level insights, including a Basel III report for regulatory compliance. The project provides a user interface (via Streamlit), an API (via FastAPI), and visualization scripts to explore the data and model performance.

### Features
- **Model Training**: Train PD (XGBoost), LGD (Decision Tree), and EAD (Gradient Boosting) models using modular scripts in `src/`, orchestrated by `main.py`.
- **Scaler Creation**: Generate a scaler (`scaler.pkl`) for feature scaling, handled by `main.py`.
- **Individual Loan Predictions**: Predict PD, LGD, EAD, and Expected Loss for a single loan via a UI (`app.py`) or API (`api.py`).
- **Portfolio Insights**: Generate a Basel III report with portfolio-level metrics like Total Risk-Weighted Assets (RWA), Capital Requirement, and Expected Loss using `generate_basel_report.py`.
- **Visualization**: Generate plots for exploratory data analysis (EDA), feature relationships, model evaluation, and portfolio metrics using `plots.py`.
- **User Interface**: A Streamlit app (`app.py`) for interactive loan risk prediction and portfolio analysis.
- **API**: A FastAPI application (`api.py`) for programmatic access to predictions and portfolio metrics.

## Project Structure
```
credit_health/
│
├── data/                                    # Directory for datasets (created during execution, not tracked in Git)
│   ├── loan_data_2007_2014.csv              # Raw dataset (required, download from Google Drive, not tracked in Git)
│   ├── loan_data_2007_2014_preprocessed.csv # Preprocessed dataset (generated by main.py, not tracked in Git)
│   └── loan_data_with_predictions.csv       # Dataset with predictions (generated by generate_basel_report.py, not tracked in Git)
│
├── models/
│   ├── pd_model_xgboost.pkl                 # Trained PD model (generated by main.py)
│   ├── lgd_model_dt.pkl                     # Trained LGD model (generated by main.py)
│   ├── ead_model_gbm.pkl                    # Trained EAD model (generated by main.py)
│   ├── scaler.pkl                           # Scaler for feature scaling (generated by main.py)
│   ├── pd_feature_importance.pkl            # Feature importance for PD model (generated by main.py)
│   ├── lgd_feature_importance.pkl           # Feature importance for LGD model (generated by main.py)
│   └── ead_feature_importance.pkl           # Feature importance for EAD model (generated by main.py)
│
├── plots/
│   ├── funded_amnt_distribution.png         # Distribution of funded_amnt (generated by plots.py)
│   ├── int_rate_distribution.png            # Distribution of int_rate (generated by plots.py)
│   ├── int_rate_vs_loan_status.png          # Interest rate vs. loan status (generated by plots.py)
│   ├── roc_curve_pd.png                     # ROC curve for PD model (generated by plots.py)
│   ├── pd_feature_importance.png            # Feature importance for PD model (generated by plots.py)
│   ├── lgd_feature_importance.png           # Feature importance for LGD model (generated by plots.py)
│   ├── ead_feature_importance.png           # Feature importance for EAD model (generated by plots.py)
│   └── portfolio_metrics.png                # Portfolio-level metrics (generated by plots.py)
│
├── results/
│   └── loan_risk_predictions.csv            # Saved predictions from app.py and api.py (generated)
│
├── src/
│   ├── data_preprocessing.py                # Preprocessing script
│   ├── pd_modeling.py                       # PD model training script
│   ├── lgd_modeling.py                      # LGD model training script
│   ├── ead_modeling.py                      # EAD model training script
│   └── expected_loss.py                     # Expected Loss calculation script
│
├── .gitignore                               # Git ignore file to exclude large datasets
├── app.py                                   # Streamlit UI for predictions and portfolio insights
├── api.py                                   # FastAPI application for predictions and portfolio metrics
├── config.yaml                              # Configuration file for feature definitions (required)
├── generate_basel_report.py                 # Script to generate Basel III report
├── main.py                                  # Main script to orchestrate preprocessing, scaler creation, and model training
├── plots.py                                 # Standalone script to generate plots
├── utils.py                                 # Utility functions for evaluation and saving results
├── basel_report.txt                         # Basel III report output (generated by generate_basel_report.py)
└── README.md                                # Project documentation
```

## Prerequisites
- **Python**: Version 3.8 or higher
- **Virtual Environment**: Recommended to isolate dependencies
- **Dataset**: The raw dataset `loan_data_2007_2014_preprocessed.csv` (download from Google Drive, see below)
- **Dependencies**: Install the required packages using the `requirements.txt` file (see below)

## Setup Instructions
Follow these steps to set up the project on a fresh system.

### 1. Clone the Repository
Clone the repository from GitLab:
```bash
git clone https://gitlab.com/rasaai/green-labs.git
cd Banking/credit_health

```

### 2. Set Up a Virtual Environment
Create and activate a virtual environment to isolate dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
Create a `requirements.txt` file in the project root directory with the following content:
```
pandas
numpy
scikit-learn
xgboost
imblearn
matplotlib
seaborn
pyyaml
joblib
scipy
fastapi
uvicorn
pandas
requests
streamlit
tabulate
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

### 4. Prepare the Dataset
- **Download the Raw Dataset**:
  - The raw dataset `loan_data_2007_2014_preprocessed.csv` is not included in the Git repository due to its large size (>100 MB), which exceeds GitLab's file size limit.
  - Download it from the following Google Drive link:
    [Download loan_data_2007_2014.csv from Google Drive](https://drive.google.com/drive/folders/1FlZSJ80tQ9yB7D58AkLUV9G7-mzyLX-i?usp=sharing)
- **Place the Dataset**:
  - Create the `credit_health/data/` directory if it doesn’t exist:
    ```bash
    mkdir -p credit_health/data
    ```
  - Move the downloaded `loan_data_2007_2014_preprocessed.csv` file to the `credit_health/data/` directory:
    ```
    credit_health/data/loan_data_2007_2014_preprocessed.csv
    ```



## Running the Project
Follow these steps in order to preprocess the data, train models, create the scaler, generate the Basel III report, create plots, and run the UI and API.

### 1. Preprocess Data, Create Scaler, and Train Models
Run `main.py` to orchestrate the preprocessing, scaler creation, and training of the PD, LGD, and EAD models. This script will:
- Load and preprocess the raw dataset (`credit_health/data/loan_data_2007_2014.csv`) using `src/data_preprocessing.py`.
- Create a binary target for PD based on `loan_status` (1 for default, 0 for non-default).
- Engineer features (e.g., `term_int`, `emp_length_int`, `mths_since_issue_d`).
- Create dummy variables for categorical features.
- Create and save a scaler (`scaler.pkl`) for feature scaling.
- Train the PD (XGBoost), LGD (Decision Tree), and EAD (Gradient Boosting) models by calling functions from `src/pd_modeling.py`, `src/lgd_modeling.py`, and `src/ead_modeling.py`.
- Save the models and their feature importance.

**Command**:
```bash
cd credit_health
python main.py
```

**Outputs**:
- `credit_health/data/loan_data_2007_2014_preprocessed.csv`: Preprocessed dataset (not tracked in Git).
- `credit_health/models/scaler.pkl`: Scaler for feature scaling.
- `credit_health/models/pd_model_xgboost.pkl`: Trained PD model.
- `credit_health/models/lgd_model_dt.pkl`: Trained LGD model.
- `credit_health/models/ead_model_gbm.pkl`: Trained EAD model.
- `credit_health/models/pd_feature_importance.pkl`: Feature importance for PD model.
- `credit_health/models/lgd_feature_importance.pkl`: Feature importance for LGD model.
- `credit_health/models/ead_feature_importance.pkl`: Feature importance for EAD model.

**Notes**:
- Ensure `credit_health/data/loan_data_2007_2014.csv` and `credit_health/config.yaml` exist before running this script.
- If the script fails, check the logs for errors (e.g., missing columns, incorrect data types).

### 2. Generate the Basel III Report
Run `generate_basel_report.py` to apply the trained models to the entire dataset and generate a Basel III report. This script will:
- Load the preprocessed dataset and trained models.
- Predict PD, LGD, and EAD for all loans.
- Calculate Expected Loss using `src/expected_loss.py`.
- Calculate Risk-Weighted Assets (RWA) using a simplified Basel III formula.
- Generate portfolio-level metrics (e.g., Total RWA, Capital Requirement).

**Command**:
```bash
python generate_basel_report.py
```

**Outputs**:
- `credit_health/basel_report.txt`: Basel III report with portfolio-level metrics.
- `credit_health/data/loan_data_with_predictions.csv`: Dataset with predictions (PD, LGD, EAD, Expected Loss, RWA, not tracked in Git).

**Notes**:
- This script requires the preprocessed dataset and trained models from Step 1.
- Ensure `credit_health/models/scaler.pkl` and the model `.pkl` files exist.
- If you see a `DtypeWarning` (e.g., "Columns (20) have mixed types"), see the Troubleshooting section for how to handle it.

### 3. Generate Plots
Run `plots.py` to generate visualizations for EDA, feature relationships, model evaluation, and portfolio metrics. This script will:
- Create EDA plots (e.g., distribution of `funded_amnt`, `int_rate`).
- Create relationship plots (e.g., `int_rate` vs. `loan_status`).
- Create model evaluation plots (e.g., ROC curve for PD, feature importance for PD, LGD, EAD).
- Create portfolio metrics plots (e.g., average PD, LGD, EAD, total Expected Loss, total RWA).

**Command**:
```bash
python plots.py
```

**Outputs** (in `credit_health/plots/` directory):
- `funded_amnt_distribution.png`: Distribution of `funded_amnt`.
- `int_rate_distribution.png`: Distribution of `int_rate`.
- `int_rate_vs_loan_status.png`: Interest rate vs. loan status.
- `roc_curve_pd.png`: ROC curve for PD model.
- `pd_feature_importance.png`: Feature importance for PD model.
- `lgd_feature_importance.png`: Feature importance for LGD model.
- `ead_feature_importance.png`: Feature importance for EAD model.
- `portfolio_metrics.png`: Portfolio-level metrics.



### 4. Run the Streamlit UI (`app.py`)
The Streamlit app provides an interactive interface to predict risk metrics for individual loans and view portfolio insights.

**Command**:
```bash
streamlit run app.py
```

**Usage**:
- Open the app in your browser (typically at `http://localhost:8501`).
- **Individual Loan Prediction Tab**:
  - Enter loan details (e.g., Funded Amount, Interest Rate, Grade).
  - Click "Predict" to see PD, LGD, EAD, and Expected Loss.
  - Predictions are saved to `credit_health/results/loan_risk_predictions.csv`.
- **Portfolio Insights Tab**:
  - View the latest Basel III report (`credit_health/basel_report.txt`).
  - Generate a new Basel III report by clicking "Generate Basel III Report".
  - See portfolio-level metrics from `credit_health/data/loan_data_with_predictions.csv`.

**Notes**:
- Ensure all model files and the preprocessed dataset are present.
- The app requires `streamlit` to be installed.
- If the app fails to load, ensure the feature engineering in `app.py` matches the training pipeline (see Troubleshooting).

### 5. Run the FastAPI Application (`api.py`)
The FastAPI application provides programmatic access to predictions and portfolio metrics.

**Command**:
```bash
uvicorn api:app --reload
```

**Usage**:
- Access the API at `http://127.0.0.1:8000`.
- **Endpoints**:
  - **POST `/predict`**:
    - Predict risk metrics for a single loan.
    - Example request:
      ```json
      {
          "loan_id": 9999999,
          "funded_amnt": 15000.0,
          "int_rate": 25.0,
          "grade": "A",
          "dti": 35.0,
          "home_ownership": "RENT",
          "purpose": "small_business",
          "initial_list_status": "f",
          "term": " 36 months",
          "annual_inc": 20000.0,
          "emp_length": "< 1 year",
          "delinq_2yrs": 2,
          "inq_last_6mths": 4,
          "open_acc": 5,
          "pub_rec": 1,
          "total_acc": 10,
          "acc_now_delinq": 0,
          "total_rev_hi_lim": 30000,
          "installment": 500.0,
          "mths_since_last_delinq": 12,
          "mths_since_last_record": 24,
          "revol_bal": 10000,
          "revol_util": 90.0,
          "issue_d": "Jan-19",
          "earliest_cr_line": "Jan-10",
          "verification_status": "Not Verified"
      }
      ```
    - Response includes PD, LGD, EAD, Expected Loss, and decision factors.
    - Predictions are saved to `credit_health/results/loan_risk_predictions.csv`.
  - **GET `/basel-report`**:
    - Retrieve the Basel III report metrics from `credit_health/basel_report.txt`.
  - **GET `/portfolio-metrics`**:
    - Retrieve portfolio-level metrics from `credit_health/data/loan_data_with_predictions.csv`.
  - **POST `/generate-basel-report`**:
    - Generate a new Basel III report.



## Outputs
- **Preprocessed Data** (not tracked in Git):
  - `credit_health/data/loan_data_2007_2014_preprocessed.csv`
- **Dataset with Predictions** (not tracked in Git):
  - `credit_health/data/loan_data_with_predictions.csv`
- **Model Files** (in `credit_health/models/`):
  - `pd_model_xgboost.pkl`
  - `lgd_model_dt.pkl`
  - `ead_model_gbm.pkl`
  - `scaler.pkl`
  - `pd_feature_importance.pkl`
  - `lgd_feature_importance.pkl`
  - `ead_feature_importance.pkl`
- **Basel III Report**:
  - `credit_health/basel_report.txt`
- **Plots** (in `credit_health/plots/`):
  - `funded_amnt_distribution.png`
  - `int_rate_distribution.png`
  - `int_rate_vs_loan_status.png`
  - `roc_curve_pd.png`
  - `pd_feature_importance.png`
  - `lgd_feature_importance.png`
  - `ead_feature_importance.png`
  - `portfolio_metrics.png`
- **Prediction Logs**:
  - `credit_health/results/loan_risk_predictions.csv`

