import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Suppress warnings for cleaner console output (use sparingly in production)
warnings.filterwarnings('ignore')

# --- Configuration ---
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(',')
MODEL_JSON_PATH = os.getenv("MODEL_JSON_PATH", "final_ensembled_model.json")
MODEL_PKL_V2_PATH = os.getenv("MODEL_PKL_V2_PATH", "final_ensembled_model_v2.pkl")
MODEL_PKL_PATH = os.getenv("MODEL_PKL_PATH", "final_ensembled_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler_correct_new.pkl")
FAILED_BANKS_SET = {"Pulaski Savings Bank", "UCO Bank", "ICICI Bank", "SBI"}

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Enable CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and explainer
model_predict = None
model_scaler = None
explainer = None

# --- Helper Functions for Model Loading ---

def load_xgboost_model(json_path: str, pkl_v2_path: str, pkl_path: str):
    """
    Attempts to load an XGBoost model from various specified paths,
    prioritizing JSON, then v2 PKL, then original PKL.
    """
    if os.path.exists(json_path):
        logger.info(f"Attempting to load model from JSON format: {json_path}")
        model = xgb.XGBClassifier()
        model.load_model(json_path)
        return model
    elif os.path.exists(pkl_v2_path):
        logger.info(f"Attempting to load model from v2 PKL format: {pkl_v2_path}")
        return joblib.load(pkl_v2_path)
    elif os.path.exists(pkl_path):
        logger.info(f"Attempting to load model from original PKL format: {pkl_path}")
        return joblib.load(pkl_path)
    else:
        raise FileNotFoundError(
            f"No XGBoost model found at: {json_path}, {pkl_v2_path}, or {pkl_path}"
        )

def load_scaler(scaler_path: str):
    """Loads the scaler model from the specified path."""
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from: {scaler_path}")
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

# --- Startup Event: Load Models and Explainer ---
@app.on_event("startup")
async def load_models_and_explainer():
    """
    Loads machine learning models (XGBoost and scaler) and initializes the SHAP explainer
    when the FastAPI application starts up.
    """
    global model_predict, model_scaler, explainer

    try:
        model_predict = load_xgboost_model(MODEL_JSON_PATH, MODEL_PKL_V2_PATH, MODEL_PKL_PATH)
        model_scaler = load_scaler(SCALER_PATH)
        logger.info("✅ Models loaded successfully.")

        try:
            if hasattr(model_predict, 'get_booster') or 'XGB' in str(type(model_predict)):
                logger.info("Initializing SHAP TreeExplainer...")
                explainer = shap.TreeExplainer(model_predict)
            else:
                logger.info("Initializing SHAP general Explainer...")
                explainer = shap.Explainer(model_predict)
            logger.info("✅ SHAP explainer initialized successfully.")

        except Exception as shap_init_error:
            logger.warning(f"SHAP TreeExplainer/Explainer failed, attempting PermutationExplainer: {shap_init_error}")
            try:
                dummy_scaled_background = np.zeros((10, 9))
                explainer = shap.explainers.Permutation(model_predict.predict, dummy_scaled_background)
                logger.info("✅ SHAP PermutationExplainer initialized.")
            except Exception as e:
                logger.error(f"❌ All SHAP explainer methods failed: {e}", exc_info=True)
                explainer = None

    except (FileNotFoundError, RuntimeError) as e:
        logger.critical(f"❌ Critical error during model loading: {e}")
        raise RuntimeError(f"Application startup failed due to model loading error: {e}")
    except Exception as e:
        logger.critical(f"❌ An unexpected error occurred during startup: {e}", exc_info=True)
        raise RuntimeError(f"Application startup failed: {e}")

# --- Pydantic Input Model for Angular Frontend ---
class AngularFlatInput(BaseModel):
    cash_13: float = Field(..., alias='13_CASH', description="Cash and balances with central banks (13)")
    treasury_bills_22: float = Field(..., alias='22_TREASURY_BILLS', description="Treasury bills (22)")
    other_gov_securities_23: float = Field(..., alias='23_OTHER_GOV_SECURITIES', description="Other government securities (23)")
    curr_acc_01: float = Field(..., alias='01_CURR_ACC', description="Current accounts (01)")
    time_deposit_02: float = Field(..., alias='02_TIME_DEPOSIT', description="Time deposits (02)")
    savings_03: float = Field(..., alias='03_SAVINGS', description="Savings deposits (03)")
    borrowing_from_public_06: float = Field(..., alias='06_BORROWING_FROM_PUBLIC', description="Borrowing from public (06)")
    interbanks_loan_payable_07: float = Field(..., alias='07_INTERBANKS_LOAN_PAYABLE', description="Interbank loans payable (07)")
    off_balsheet_commitments_11: float = Field(..., alias='11_OFF_BALSHEET_COMMITMENTS', description="Off-balance sheet commitments (11)")
    ewaq_capital: float = Field(..., alias='EWAQ_Capital', description="Equity Weighted Average Quality Capital")
    ewaq_gross_loans: float = Field(..., alias='EWAQ_GrossLoans', description="Equity Weighted Average Quality Gross Loans")
    commercial_bills_25: float = Field(..., alias='25_COMMERCIAL_BILLS', description="Commercial bills (25)")
    f125_liab_total: float = Field(..., alias='F125_LIAB_TOTAL', description="Total liabilities (F125)")

    frontend_deposit_growth_rate: float = Field(..., alias='Deposit_Growth_Rate', description="Deposit Growth Rate from frontend")
    frontend_funding_cost_proxy: float = Field(..., alias='Funding_Cost_Change_Proxy', description="Funding Cost Change Proxy from frontend")
    frontend_exposed_banks: str = Field(..., alias='exposed_banks', description="Comma-separated string of exposed bank names")
    frontend_exposure_amounts: str = Field(..., alias='exposure_amounts', description="Comma-separated string of exposure amounts")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
                    "13_CASH": 1000000.0,
                    "22_TREASURY_BILLS": 500000.0,
                    "23_OTHER_GOV_SECURITIES": 200000.0,
                    "01_CURR_ACC": 3000000.0,
                    "02_TIME_DEPOSIT": 4000000.0,
                    "03_SAVINGS": 5000000.0,
                    "06_BORROWING_FROM_PUBLIC": 100000.0,
                    "07_INTERBANKS_LOAN_PAYABLE": 50000.0,
                    "11_OFF_BALSHEET_COMMITMENTS": 20000.0,
                    "EWAQ_Capital": 1500000.0,
                    "EWAQ_GrossLoans": 8000000.0,
                    "25_COMMERCIAL_BILLS": 100000.0,
                    "F125_LIAB_TOTAL": 10000000.0,
                    "Deposit_Growth_Rate": 0.05,
                    "Funding_Cost_Change_Proxy": -0.01,
                    "exposed_banks": "Safe Bank A, SBI, Safe Bank B",
                    "exposure_amounts": "100000, 50000, 200000"
                }
            ]
        }
    }

# --- Feature Engineering Helper ---
def calculate_derived_features(data: AngularFlatInput) -> dict:
    base_vars = {
        '13_CASH': data.cash_13, '22_TREASURY_BILLS': data.treasury_bills_22,
        '23_OTHER_GOV_SECURITIES': data.other_gov_securities_23,
        '01_CURR_ACC': data.curr_acc_01, '02_TIME_DEPOSIT': data.time_deposit_02,
        '03_SAVINGS': data.savings_03, '06_BORROWING_FROM_PUBLIC': data.borrowing_from_public_06,
        '07_INTERBANKS_LOAN_PAYABLE': data.interbanks_loan_payable_07,
        '11_OFF_BALSHEET_COMMITMENTS': data.off_balsheet_commitments_11,
        'EWAQ_Capital': data.ewaq_capital, 'EWAQ_GrossLoans': data.ewaq_gross_loans,
        '25_COMMERCIAL_BILLS': data.commercial_bills_25, 'F125_LIAB_TOTAL': data.f125_liab_total
    }

    OUTFLOW_RATE_CURR_ACC = 0.03; OUTFLOW_RATE_SAVINGS = 0.10; OUTFLOW_RATE_TIME_DEPOSIT = 0.05
    OUTFLOW_RATE_INTERBANKS_LOAN_PAYABLE = 0.25; OUTFLOW_RATE_BORROWING_FROM_PUBLIC = 0.20
    OUTFLOW_RATE_OFF_BALSHEET_COMMITMENTS = 0.20
    ASF_RATE_TIME_DEPOSIT = 0.95; ASF_RATE_SAVINGS = 0.90; ASF_RATE_CURR_ACC = 0.50
    RSF_RATE_GROSS_LOANS = 0.85; RSF_RATE_COMMERCIAL_BILLS = 0.50
    RSF_RATE_OFF_BALSHEET_COMMITMENTS = 0.05; RSF_RATE_INTERBANKS_LOAN_PAYABLE = 0.20

    HQLA = (base_vars.get('13_CASH', 0) + base_vars.get('22_TREASURY_BILLS', 0) +
            base_vars.get('23_OTHER_GOV_SECURITIES', 0))
    Total_Outflows = (OUTFLOW_RATE_CURR_ACC * base_vars.get('01_CURR_ACC', 0) +
                      OUTFLOW_RATE_SAVINGS * base_vars.get('03_SAVINGS', 0) +
                      OUTFLOW_RATE_TIME_DEPOSIT * base_vars.get('02_TIME_DEPOSIT', 0) +
                      OUTFLOW_RATE_INTERBANKS_LOAN_PAYABLE * base_vars.get('07_INTERBANKS_LOAN_PAYABLE', 0) +
                      OUTFLOW_RATE_BORROWING_FROM_PUBLIC * base_vars.get('06_BORROWING_FROM_PUBLIC', 0) +
                      OUTFLOW_RATE_OFF_BALSHEET_COMMITMENTS * base_vars.get('11_OFF_BALSHEET_COMMITMENTS', 0))
    LCR = HQLA / Total_Outflows if Total_Outflows else 0

    ASF = (base_vars.get('EWAQ_Capital', 0) + ASF_RATE_TIME_DEPOSIT * base_vars.get('02_TIME_DEPOSIT', 0) +
           ASF_RATE_SAVINGS * base_vars.get('03_SAVINGS', 0) + ASF_RATE_CURR_ACC * base_vars.get('01_CURR_ACC', 0))
    RSF = (RSF_RATE_GROSS_LOANS * base_vars.get('EWAQ_GrossLoans', 0) +
           RSF_RATE_COMMERCIAL_BILLS * base_vars.get('25_COMMERCIAL_BILLS', 0) +
           RSF_RATE_OFF_BALSHEET_COMMITMENTS * base_vars.get('11_OFF_BALSHEET_COMMITMENTS', 0) +
           RSF_RATE_INTERBANKS_LOAN_PAYABLE * base_vars.get('07_INTERBANKS_LOAN_PAYABLE', 0))
    NSFR = ASF / RSF if RSF else 0

    Loan_to_Capital = base_vars.get('EWAQ_GrossLoans', 0) / (base_vars.get('EWAQ_Capital', 0) + 1e-9)
    LCR_Shortfall = max(1 - LCR, 0)
    Stable_Funding_Gap = ASF - RSF
    Capital_Adequacy = base_vars.get('EWAQ_Capital', 0) / (base_vars.get('F125_LIAB_TOTAL', 0) + 1e-9)

    exposure_to_failed = 0
    try:
        banks = [b.strip() for b in data.frontend_exposed_banks.split(',') if b.strip()]
        amounts = [float(a.strip()) for a in data.frontend_exposure_amounts.split(',') if a.strip()]
        if len(banks) != len(amounts):
            logger.warning("Mismatch between number of exposed banks and exposure amounts.")
        
        for b, a in zip(banks, amounts):
            if b in FAILED_BANKS_SET:
                exposure_to_failed += a
    except ValueError as e:
        logger.error(f"Error parsing exposed banks/amounts: {e}. Defaulting exposure_to_failed to 0.")
        exposure_to_failed = 0

    model_input = {
        'LCR': LCR, 'NSFR': NSFR, 'Deposit_Growth_Rate': data.frontend_deposit_growth_rate,
        'Funding_Cost_Change_Proxy': data.frontend_funding_cost_proxy,
        'Loan_to_Capital': Loan_to_Capital, 'LCR_Shortfall': LCR_Shortfall,
        'Stable_Funding_Gap': Stable_Funding_Gap, 'Capital_Adequacy': Capital_Adequacy,
        'Exposure_To_Failed': exposure_to_failed
    }
    return model_input

# --- SHAP Explanation Helper ---
def get_shap_explanations_and_advice(
    model_input_df: pd.DataFrame,
    explainer: shap.Explainer,
    predicted_risk: int
) -> tuple[dict, list]:
    explanations = {}
    advice_list = []

    if explainer is None:
        advice_list.append({
            "feature": "N/A", "impact_value": 0, "impact_on_risk": "N/A",
            "advice": "SHAP explainer not available, cannot generate specific advice.",
            "explanation_text": "SHAP explainer not available. Please check backend logs for model loading issues."
        })
        return explanations, advice_list

    try:
        shap_values_raw = explainer(model_input_df)
        shap_vals_for_positive_class = None

        target_class_index = 1

        if isinstance(shap_values_raw, list):
            if len(shap_values_raw) == 2 and isinstance(shap_values_raw[target_class_index], np.ndarray):
                shap_vals_for_positive_class = shap_values_raw[target_class_index]
                if shap_vals_for_positive_class.ndim > 1:
                    shap_vals_for_positive_class = shap_vals_for_positive_class.flatten()
        elif isinstance(shap_values_raw, np.ndarray):
            if shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2:
                shap_vals_for_positive_class = shap_values_raw[0, :, target_class_index]
            elif shap_values_raw.ndim == 2 and shap_values_raw.shape[0] == 1:
                shap_vals_for_positive_class = shap_values_raw[0]
            elif shap_values_raw.ndim == 1:
                shap_vals_for_positive_class = shap_values_raw
        elif hasattr(shap_values_raw, 'values') and isinstance(shap_values_raw.values, np.ndarray):
            if shap_values_raw.values.ndim == 3 and shap_values_raw.values.shape[2] == 2:
                shap_vals_for_positive_class = shap_values_raw.values[0, :, target_class_index]
            elif shap_values_raw.values.ndim == 2 and shap_values_raw.values.shape[0] == 1:
                shap_vals_for_positive_class = shap_values_raw.values[0]
            elif shap_values_raw.values.ndim == 1:
                shap_vals_for_positive_class = shap_values_raw.values
        
        logger.info(f"SHAP raw result type: {type(shap_values_raw)}")
        if isinstance(shap_values_raw, np.ndarray):
            logger.info(f"SHAP raw result shape: {shap_values_raw.shape}")
        elif hasattr(shap_values_raw, 'values') and isinstance(shap_values_raw.values, np.ndarray):
             logger.info(f"SHAP Explanation object values shape: {shap_values_raw.values.shape}")
        logger.info(f"SHAP positive class values (after extraction) type: {type(shap_vals_for_positive_class)}")
        if isinstance(shap_vals_for_positive_class, np.ndarray):
            logger.info(f"SHAP positive class values (after extraction) shape: {shap_vals_for_positive_class.shape}")


        if shap_vals_for_positive_class is not None and shap_vals_for_positive_class.ndim == 1:
            shap_dict = dict(zip(model_input_df.columns, shap_vals_for_positive_class))
            sorted_explanations = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

            top_features = sorted_explanations[:3]
            
            for feature, impact in top_features:
                logger.debug(f"Generating explanation for feature: {feature}, impact: {impact}")

                if abs(impact) > 0.001:
                    advice_mapping = {
                        "LCR": "Consider increasing your High-Quality Liquid Assets or reducing short-term outflows.",
                        "NSFR": "Improve long-term funding or reduce long-term illiquid assets.",
                        "Loan_to_Capital": "Reduce loan exposure or increase capital to strengthen balance sheet.",
                        "LCR_Shortfall": "Work on improving LCR to minimize liquidity stress.",
                        "Stable_Funding_Gap": "Reduce funding mismatch by aligning assets and liabilities better.",
                        "Capital_Adequacy": "Increase core capital to ensure better coverage against liabilities.",
                        "Exposure_To_Failed": "Minimize exposures to historically failing institutions.",
                        "Deposit_Growth_Rate": "Improve deposit mobilization strategies.",
                        "Funding_Cost_Change_Proxy": "Try to stabilize and reduce cost of funding sources."
                    }
                    advice_text = advice_mapping.get(feature, "No specific advice for this feature.")
                    
                    impact_direction = "increases risk" if impact > 0 else "decreases risk"
                    
                    explanation_text = (
                        f"This feature {impact_direction} by {abs(impact):.4f}."
                    )
                    
                    advice_list.append({
                        "feature": feature, "impact_value": round(float(impact), 4),
                        "impact_on_risk": impact_direction, "advice": advice_text,
                        "explanation_text": explanation_text
                    })
                else:
                    advice_list.append({
                        "feature": feature, "impact_value": round(float(impact), 4),
                        "impact_on_risk": "negligible impact",
                        "advice": "This feature had a negligible impact on the risk prediction.",
                        "explanation_text": f"This feature had a negligible impact ({abs(impact):.4f})."
                    })
        else:
            logger.warning(f"Could not determine SHAP values for the positive class in a usable 1D format. Final SHAP values type: {type(shap_vals_for_positive_class)}, ndim: {getattr(shap_vals_for_positive_class, 'ndim', 'N/A')}")
            advice_list.append({
                "feature": "N/A", "impact_value": 0, "impact_on_risk": "N/A",
                "advice": "No specific advice could be generated due to SHAP explanation issues.",
                "explanation_text": "SHAP values format not recognized or positive class values not found. Check backend logs."
            })

    except Exception as e:
        logger.error(f"Error during SHAP explanation or advice generation: {e}", exc_info=True)
        advice_list.append({
            "feature": "N/A", "impact_value": 0, "impact_on_risk": "N/A",
            "advice": "An error occurred during advice generation.",
            "explanation_text": f"SHAP explanation failed: {str(e)}. Check backend logs."
        })

    return explanations, advice_list

# --- API Endpoints ---

@app.get("/health")
def health_check():
    return {
        "status": "healthy", "model_loaded": model_predict is not None,
        "scaler_loaded": model_scaler is not None, "shap_available": explainer is not None,
        "xgboost_version": xgb.__version__
    }

@app.post("/predict/")
async def predict_risk(data: AngularFlatInput):
    if model_predict is None or model_scaler is None:
        logger.error("Prediction requested but models are not loaded.")
        raise HTTPException(status_code=500, detail="Models not loaded. Please check server logs.")

    try:
        model_input = calculate_derived_features(data)
        input_df = pd.DataFrame([model_input])
        input_scaled = model_scaler.transform(input_df)

        predicted_class = model_predict.predict(input_scaled)[0]
        risk_label = "RISKY" if predicted_class == 1 else "NOT RISKY"

        logger.info(f"Predicted Risk Label: {risk_label}")

        shap_explanations_dict, advice_list = get_shap_explanations_and_advice(
            pd.DataFrame(input_scaled, columns=input_df.columns),
            explainer,
            predicted_class
        )

        return {
            "risk_label": risk_label,
            # Removed risk_probability and risk_category from the response
            "shap_explanation": shap_explanations_dict,
            "advice": advice_list,
            "model_input_features": model_input
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Prediction endpoint failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    logging.getLogger().setLevel(logging.DEBUG)
    logger.info(f"Starting FastAPI application on http://0.0.0.0:8000 with CORS for origins: {ALLOWED_ORIGINS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
