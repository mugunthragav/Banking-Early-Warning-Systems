import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import xgboost as xgb
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Dynamic project root
project_root = Path(__file__).resolve().parent.parent

# Configuration
MODEL_JSON_PATH = os.getenv("MODEL_JSON_PATH", str(project_root / "models" / "final_ensembled_model.json"))
MODEL_PKL_V2_PATH = os.getenv("MODEL_PKL_V2_PATH", str(project_root / "models" / "final_ensembled_model_v2.pkl"))
MODEL_PKL_PATH = os.getenv("MODEL_PKL_PATH", str(project_root / "models" / "final_ensembled_model.pkl"))
SCALER_PATH = os.getenv("SCALER_PATH", str(project_root / "models" / "scaler_correct_new.pkl"))
FAILED_BANKS_SET = {"Pulaski Savings Bank", "UCO Bank", "ICICI", "SBI"}

# Logger setup with UTF-8 encoding
logger = logging.getLogger(__name__)
if not logger.handlers:  # Prevent duplicate handlers
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    try:
        handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass  # StreamHandler may not support reconfigure in some Python versions
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

router = APIRouter(
    prefix="",
    tags=["Liquidity Risk"]
)

# Global variables
model_predict = None
model_scaler = None
explainer = None
_initialized = False  # Flag to prevent duplicate initialization

# Helper Functions
def load_xgboost_model(json_path: str, pkl_v2_path: str, pkl_path: str):
    if os.path.exists(json_path):
        logger.info(f"Loading model from JSON: {json_path}")
        model = xgb.XGBClassifier()
        model.load_model(json_path)
        return model
    elif os.path.exists(pkl_v2_path):
        logger.info(f"Loading model from v2 PKL: {pkl_v2_path}")
        return joblib.load(pkl_v2_path)
    elif os.path.exists(pkl_path):
        logger.info(f"Loading model from PKL: {pkl_path}")
        return joblib.load(pkl_path)
    else:
        raise FileNotFoundError(
            f"No XGBoost model found at: {json_path}, {pkl_v2_path}, or {pkl_path}"
        )

def load_scaler(scaler_path: str):
    if os.path.exists(scaler_path):
        logger.info(f"Loading scaler from: {scaler_path}")
        return joblib.load(scaler_path)
    else:
        raise FileNotFoundError(f"Scaler not found at: {scaler_path}")

# Startup event
@router.on_event("startup")
async def load_models_and_explainer():
    global model_predict, model_scaler, explainer, _initialized
    if _initialized:
        logger.info("Models already initialized, skipping startup.")
        return

    try:
        model_predict = load_xgboost_model(MODEL_JSON_PATH, MODEL_PKL_V2_PATH, MODEL_PKL_PATH)
        model_scaler = load_scaler(SCALER_PATH)
        logger.info("[SUCCESS] Liquidity models loaded successfully.")

        try:
            logger.info("Initializing SHAP TreeExplainer...")
            explainer = shap.TreeExplainer(model_predict)
            logger.info("[SUCCESS] SHAP TreeExplainer initialized successfully.")
        except Exception as shap_init_error:
            logger.warning(f"SHAP TreeExplainer failed: {shap_init_error}. Attempting PermutationExplainer.")
            try:
                dummy_scaled_background = np.zeros((10, 9))
                explainer = shap.explainers.Permutation(model_predict.predict, dummy_scaled_background)
                logger.info("[SUCCESS] SHAP PermutationExplainer initialized.")
            except Exception as e:
                logger.error(f"All SHAP explainer methods failed: {e}")
                explainer = None
        _initialized = True
    except (FileNotFoundError, RuntimeError) as e:
        logger.critical(f"Critical error during model loading: {e}")
        raise RuntimeError(f"Liquidity router startup failed: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error during startup: {str(e)}")
        raise RuntimeError(f"Liquidity router startup failed: {str(e)}")

# Pydantic Input Model
class AngularFlatInput(BaseModel):
    cash_13: float = Field(..., alias='13_CASH')
    treasury_bills_35: float = Field(..., alias='Treasury_bills')
    labels_liquid: float = Field(..., alias='Labels_Liquid')
    curr_deposit_01: float = Field(..., alias='Curr_Deposit')
    fixed_deposit_02: float = Field(..., alias='Fixed_Deposit')
    savings_03: float = Field(..., alias='General_Savings')
    borrowing_borrow_06: float = Field(..., alias='Borrowing_Borrow')
    interbanks_borrow_07: float = Field(..., alias='Balance_Interbank')
    warehouse_commitment_11: float = Field(..., alias='Warehouse_Flag')
    capital_warehouse: float = Field(..., alias='earnings_Capital')
    gross_loans_warehouse: float = Field(..., alias='earnings_Gross_Loans')
    commercial_bills_22: float = Field(..., alias='Commercial_Flag')
    total_liab_123: float = Field(..., alias='Liabilities_Income')
    deposit_increase: float = Field(..., alias='Deposit_Growth_Rate')  # Updated alias
    funding_rate: float = Field(..., alias='Funding_Cost_Change_Proxy')  # Updated alias
    exposed_banks: str = Field(..., alias='Institution_Models')
    exposure_values: str = Field(..., alias='earnings_amounts')

    class Config:
        allow_population_by_field_name = True

def calculate_derived_features(data: AngularFlatInput) -> dict:
    base_vars = {
        '13_CASH': data.cash_13,
        'Treasury_bills': data.treasury_bills_35,
        'Labels_Liquid': data.labels_liquid,
        'Curr_Deposit': data.curr_deposit_01,
        'Fixed_Deposit': data.fixed_deposit_02,
        'General_Savings': data.savings_03,
        'Borrowing_Borrow': data.borrowing_borrow_06,
        'Balance_Interbank': data.interbanks_borrow_07,
        'Warehouse_Flag': data.warehouse_commitment_11,
        'Commercial_Flag': data.commercial_bills_22,
        'earnings_Capital': data.capital_warehouse,
        'earnings_Gross_Loans': data.gross_loans_warehouse,
        'Liabilities_Income': data.total_liab_123
    }

    OUTFLOW_RATE_CURR_ACT = 0.05
    OUTFLOW_RATE_SAVINGS = 0.10
    OUTFLOW_RATE_SAVED = 0.05
    OUTFLOW_RATE_INTERBALANCE = 0.25
    OUTFLOW_BORROW = 0.20
    OUTFLOW_OUTCOME = 0.02
    ASF_RATE_SAVED = 0.95
    ASF_RATE_SAVINGS = 0.90
    ASF_RATE_ACT = 0.50
    ACE_RATE_GROSS_LOANS = 0.85
    ACE_RATE_COMMERCIAL = 0.50
    ACE_RATE_OUTCOME = 0.05
    ACE_RATE_INTERBALANCE = 0.20

    HQLA = (base_vars.get('13_CASH', 0) + base_vars.get('Treasury_bills', 0) +
            base_vars.get('Labels_Liquid', 0))
    Total_Outflows = (
        OUTFLOW_RATE_CURR_ACT * base_vars.get('Curr_Deposit', 0) +
        OUTFLOW_RATE_SAVINGS * base_vars.get('General_Savings', 0) +
        OUTFLOW_RATE_SAVED * base_vars.get('Fixed_Deposit', 0) +
        OUTFLOW_RATE_INTERBALANCE * base_vars.get('Balance_Interbank', 0) +
        OUTFLOW_BORROW * base_vars.get('Borrowing_Borrow', 0) +
        OUTFLOW_OUTCOME * base_vars.get('Warehouse_Flag', 0)
    )
    LCR = HQLA / Total_Outflows if Total_Outflows else 0

    ASF = (
        base_vars.get('earnings_Capital', 0) +
        ASF_RATE_SAVED * base_vars.get('Fixed_Deposit', 0) +
        ASF_RATE_SAVINGS * base_vars.get('General_Savings', 0) +
        ASF_RATE_ACT * base_vars.get('Curr_Deposit', 0)
    )
    ACE = (
        ACE_RATE_GROSS_LOANS * base_vars.get('earnings_Gross_Loans', 0) +
        ACE_RATE_COMMERCIAL * base_vars.get('Commercial_Flag', 0) +
        ACE_RATE_OUTCOME * base_vars.get('Warehouse_Flag', 0) +
        ACE_RATE_INTERBALANCE * base_vars.get('Balance_Interbank', 0)
    )
    NSFR = ASF / ACE if ACE else 0

    Loan_to_Capital = base_vars.get('earnings_Gross_Loans', 0) / (base_vars.get('earnings_Capital', 0) + 1e-9)
    LCR_Shortfall = max(1 - LCR, 0)
    Stable_Funding_Gap = ASF - ACE
    Capital_Adequacy = base_vars.get('earnings_Capital', 0) / (base_vars.get('Liabilities_Income', 0) + 1e-9)

    exposure_to_failed = 0
    try:
        banks = [b.strip() for b in data.exposed_banks.split(',') if b.strip()]
        amounts = [float(a.strip()) for a in data.exposure_values.split(',') if a.strip()]
        if len(banks) != len(amounts):
            logger.warning("Mismatch between number of exposed banks and exposure amounts.")
        
        for b, a in zip(banks, amounts):
            if b in FAILED_BANKS_SET:
                exposure_to_failed += a
    except ValueError as e:
        logger.error(f"Error parsing exposed banks/amounts: {e}. Defaulting exposure_to_failed to 0.")
        exposure_to_failed = 0

    model_input = {
        'LCR': LCR,
        'NSFR': NSFR,
        'Deposit_Growth_Rate': data.deposit_increase,  # Updated feature name
        'Funding_Cost_Change_Proxy': data.funding_rate,  # Updated feature name
        'Loan_to_Capital': Loan_to_Capital,
        'LCR_Shortfall': LCR_Shortfall,
        'Stable_Funding_Gap': Stable_Funding_Gap,
        'Capital_Adequacy': Capital_Adequacy,
        'Exposure_To_Failed': exposure_to_failed
    }
    return model_input

def get_shap_explanations_and_advice(model_input_df: pd.DataFrame, explainer: shap.Explainer, predicted_risk: int) -> tuple[dict, list]:
    explanations = {}
    advice_list = []

    if explainer is None:
        advice_list.append({
            "feature": "N/A",
            "impact_value": 0,
            "impact_on_risk": "N/A",
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
                    shap_vals_for_positive_class = shap_values_raw[target_class_index].flatten()
            elif len(shap_values_raw) == 1:
                shap_vals_for_positive_class = shap_values_raw[0]
                if shap_vals_for_positive_class.ndim > 1:
                    shap_vals_for_positive_class = shap_values_raw[0].flatten()
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

        if shap_vals_for_positive_class is not None and shap_vals_for_positive_class.ndim == 1:
            shap_dict = dict(zip(model_input_df.columns, shap_vals_for_positive_class))
            sorted_explanations = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_explanations[:3]
            
            for feature, impact in top_features:
                advice_mapping = {
                    "LCR": "Consider increasing your High-Quality Liquid Assets or reducing short-term outflows.",
                    "NSFR": "Improve long-term funding or reduce long-term illiquid assets.",
                    "Loan_to_Capital": "Reduce loan exposure or increase capital to strengthen balance sheet.",
                    "LCR_Shortfall": "Work on improving LCR to minimize liquidity stress.",
                    "Stable_Funding_Gap": "Reduce funding mismatch by aligning assets and liabilities better.",
                    "Capital_Adequacy": "Increase core capital to ensure better coverage against liabilities.",
                    "Exposure_To_Failed": "Minimize exposures to historically failing institutions.",
                    "Deposit_Growth_Rate": "Improve deposit mobilization strategies.",  # Updated
                    "Funding_Cost_Change_Proxy": "Try to stabilize and reduce cost of funding sources."  # Updated
                }
                advice_text = advice_mapping.get(feature, "No specific advice for this feature.")
                impact_direction = "increases risk" if impact > 0 else "decreases risk"
                explanation_text = f"This feature {impact_direction} by {abs(impact):.4f}."
                advice_list.append({
                    "feature": feature,
                    "impact_value": round(float(impact), 4),
                    "impact_on_risk": impact_direction,
                    "advice": advice_text,
                    "explanation_text": explanation_text
                })
        else:
            logger.warning(f"Could not determine SHAP values for positive class. Type: {type(shap_vals_for_positive_class)}")
            advice_list.append({
                "feature": "N/A",
                "impact_value": 0,
                "impact_on_risk": "N/A",
                "advice": "No specific advice due to SHAP issues.",
                "explanation_text": "SHAP values format not recognized."
            })
    except Exception as e:
        logger.error(f"Error during SHAP explanation: {e}")
        advice_list.append({
            "feature": "N/A",
            "impact_value": 0,
            "impact_on_risk": "N/A",
            "advice": "An error occurred during advice generation.",
            "explanation_text": f"SHAP explanation failed: {str(e)}."
        })

    return explanations, advice_list

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_predict is not None,
        "scaler_loaded": model_scaler is not None,
        "shap_available": explainer is not None,
        "xgboost_version": xgb.__version__
    }

@router.post("/predict")
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
            "shap_explanation": shap_explanations_dict,
            "advice": advice_list,
            "model_input_features": model_input
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")