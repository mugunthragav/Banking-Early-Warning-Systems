# backend/app/database/models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Index, BigInteger
from sqlalchemy.sql import func  # For server-side default timestamps
from .db_session import Base  # Import Base from your db_session.py

class ApplicationLog(Base):
    __tablename__ = "application_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    # --- Timestamping and Status ---
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    #processed_at = Column(DateTime(timezone=True), onupdate=func.now())  # Or set explicitly when processing finishes
    status_message = Column(String(255), nullable=True)  # e.g., "Successfully processed", "Contact Mukunth", "Row dropped..."

    # --- Basic Loan Info ---
    loan_amnt = Column(BigInteger, nullable=True)

    # --- ML Predicted Outputs ---
    pd_ml_probability = Column(Float, nullable=True)
    pd_ml_prediction = Column(Integer, nullable=True)  # 0 or 1
    probability_of_repayment = Column(Float, nullable=True)  # 1 - pd_ml_probability
    lgd_ml_ann = Column(Float, nullable=True)  # LGD value
    recovery_rate_ml = Column(Float, nullable=True)  # 1 - lgd_ml_ann
    ead_ml_meta = Column(Float, nullable=True)
    expected_loss_ml = Column(Float, nullable=True)

    # --- AI Interpretation ---
    #ai_interpretation_text = Column(Text, nullable=True)

    # --- Optional: Add an index for querying by creation date or status ---
    #Index("ix_application_logs_created_at", created_at)

# --- ADD ALL THE MISSING INPUT COLUMNS HERE ---
    loan_amnt = Column(BigInteger, nullable=True)
    funded_amnt = Column(Float, nullable=True)
    funded_amnt_inv = Column(Float, nullable=True)
    term = Column(String(255), nullable=True)
    int_rate = Column(Float, nullable=True)
    installment = Column(Float, nullable=True)
    grade = Column(String(255), nullable=True)
    emp_length = Column(Float, nullable=True)
    home_ownership = Column(String(255), nullable=True)
    annual_inc = Column(Float, nullable=True)
    verification_status = Column(String(255), nullable=True)
    dti = Column(Float, nullable=True)
    delinq_2yrs = Column(Float, nullable=True)
    inq_last_6mths = Column(Float, nullable=True)
    mths_since_last_delinq = Column(Float, nullable=True)
    open_acc = Column(Float, nullable=True)
    pub_rec = Column(Float, nullable=True)
    revol_bal = Column(Float, nullable=True)
    revol_util = Column(Float, nullable=True)
    total_acc = Column(Float, nullable=True)
    initial_list_status = Column(String(255), nullable=True)
    out_prncp = Column(Float, nullable=True)
    total_pymnt = Column(Float, nullable=True)
    total_rec_prncp = Column(Float, nullable=True)
    total_rec_int = Column(Float, nullable=True)
    total_rec_late_fee = Column(Float, nullable=True)
    recoveries = Column(Float, nullable=True)
    collection_recovery_fee = Column(Float, nullable=True)
    last_pymnt_amnt = Column(Float, nullable=True)
    tot_coll_amt = Column(Float, nullable=True)
    tot_cur_bal = Column(Float, nullable=True)
    total_rev_hi_lim = Column(Float, nullable=True)
    mths_since_earliest_cr_line = Column(Float, nullable=True)
    purpose = Column(String(255), nullable=True)
