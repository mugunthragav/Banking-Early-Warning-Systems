# backend/app/schemas/loan_application_schemas.py
from pydantic import BaseModel, Field, ConfigDict # Import ConfigDict for Pydantic V2
from typing import List, Optional, Dict, Any
# from datetime import datetime # Not currently used in these schemas, can be removed if not needed

# --- Input Schemas ---
class LoanApplicationInputItem(BaseModel):
    # Your 35 mandatory input fields
    # Using Optional for now as in your provided code. If a field is truly mandatory
    # in the incoming request (cannot be omitted), remove Optional and default None.
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: Optional[float] = None
    term: str # e.g., " 36 months"
    int_rate: Optional[float] = None
    installment: Optional[float] = None
    grade: str
    emp_length: Optional[float] = None # Received as numeric
    home_ownership: str
    annual_inc: Optional[float] = None
    verification_status: str
    dti: Optional[float] = None
    delinq_2yrs: Optional[float] = None # Changed to float to match other numerics for flexibility
    inq_last_6mths: Optional[float] = None
    mths_since_last_delinq: Optional[float] = None
    open_acc: Optional[float] = None
    pub_rec: Optional[float] = None
    revol_bal: Optional[float] = None # Changed to float
    revol_util: Optional[float] = None
    total_acc: Optional[float] = None
    initial_list_status: str
    out_prncp: Optional[float] = None
    total_pymnt: Optional[float] = None
    total_rec_prncp: Optional[float] = None
    total_rec_int: Optional[float] = None
    total_rec_late_fee: Optional[float] = None
    recoveries: Optional[float] = None
    collection_recovery_fee: Optional[float] = None
    last_pymnt_amnt: Optional[float] = None
    tot_coll_amt: Optional[float] = None # Changed to float
    tot_cur_bal: Optional[float] = None # Changed to float
    total_rev_hi_lim: Optional[float] = None # Changed to float
    mths_since_earliest_cr_line: Optional[float] = None
    purpose: str

    # Pydantic V2 configuration to allow creating this model from ORM objects (SQLAlchemy models)
    model_config = ConfigDict(from_attributes=True)


class LoanApplicationRequestBatch(BaseModel):
    applications: List[LoanApplicationInputItem]
    model_config = ConfigDict(from_attributes=True)


# --- Output Schemas ---

class PredictionResultItem(BaseModel):
    application_db_id: Optional[int] = None
    status_message: str
    original_input_snippet: Optional[Dict[str, Any]] = None

    # ML Predicted Outputs
    pd_ml_probability: Optional[float] = None
    pd_ml_prediction: Optional[int] = None
    probability_of_repayment: Optional[float] = None
    lgd_ml_ann: Optional[float] = None
    recovery_rate_ml: Optional[float] = None
    ead_ml_meta: Optional[float] = None
    expected_loss_ml: Optional[float] = None

    # AI Interpretation
    ai_interpretation_text: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResultItem]
    batch_id: Optional[str] = None
    processing_summary: Optional[Dict[str, Any]] = None # e.g., total processed, errors
    model_config = ConfigDict(from_attributes=True)

class LeanLoanPredictionResult(BaseModel):
    """
    Represents an individual loan prediction result with a reduced set of fields.
    Omits: status_message, original_input_snippet, ai_interpretation_text
    """
    application_db_id: Optional[int] = None

    # ML Predicted Outputs
    pd_ml_probability: Optional[float] = None
    pd_ml_prediction: Optional[int] = None  # Crucial for defaulters_percentage
    probability_of_repayment: Optional[float] = None
    lgd_ml_ann: Optional[float] = None
    recovery_rate_ml: Optional[float] = None
    ead_ml_meta: Optional[float] = None
    expected_loss_ml: Optional[float] = None # Crucial for cumulative_expected_loss and credit_risk_percentage

    model_config = ConfigDict(from_attributes=True)


class AggregatedPredictionResponse(BaseModel):
    """
    Represents the batch prediction response with aggregated metrics at the top
    and a list of lean individual results.
    """
    # New Aggregate Metrics
    cumulative_expected_loss: float = Field(description="Sum of predicted ML expected loss for all applications in the batch.")
    credit_risk_percentage: float = Field(description="Overall credit risk: Sum of expected loss / sum of loan amounts for the batch.")
    defaulters_percentage: float = Field(description="Percentage of applications predicted to default (pd_ml_prediction = 1) in the batch.")

    # AI Summary for Aggregate Metrics
    aggregate_metrics_ai_summary: str = Field(description="AI-generated interpretation of the aggregate risk metrics.")

    # List of Lean Individual Results
    results: List[LeanLoanPredictionResult] = Field(description="List of individual prediction results with reduced fields.")

    # Optional: You might want to retain some batch-level info if needed
    #batch_id: Optional[str] = None
    #processing_summary: Optional[Dict[str, Any]] = None # e.g., total processed, errors

    model_config = ConfigDict(from_attributes=True)