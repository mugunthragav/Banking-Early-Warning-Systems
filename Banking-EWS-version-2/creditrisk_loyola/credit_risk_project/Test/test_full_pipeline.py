# backend/test_full_pipeline.py
import pandas as pd
import numpy as np
import sys
import os
import re
from typing import Dict, List, Any, Optional, TypeVar, Sequence
from sqlalchemy.orm import Session
from app.database.model import ApplicationLog
from app.schemas.loan_application_schemas import AggregatedPredictionResponse

# When running 'python test_full_pipeline.py' from the 'backend/' directory,
# 'backend/' is automatically added to sys.path, so 'app' is directly importable.
try:
    from app.services.prediction_orchestration_service import orchestrate_predictions_for_batch
    from app.database.db_session import SessionLocal, engine, Base
    from app.core import config
except ImportError as e:
    print(f"CRITICAL Error importing application modules: {e}")
    print("Please ensure that:")
    print("1. This script ('test_full_pipeline.py') is located directly in your 'backend/' directory.")
    print("2. You are running this script from the 'backend/' directory (e.g., 'python test_full_pipeline.py').")
    print("3. The 'app/' module is a direct subdirectory of 'backend/'.")
    print("4. All necessary __init__.py files exist in 'app/' and its subdirectories.")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

def format_float(value: Optional[float]) -> str:
    """Format float values nicely for display"""
    if value is None:
        return "None"
    return f"{value:.6f}"

def test_prediction_pipeline():
    db = SessionLocal()
    try:
        # Create a sample dataframe with all required fields
        data = {
            'loan_amnt': [22500, 10000, 5000],
            'funded_amnt': [22500, 10000, 5000],
            'funded_amnt_inv': [22500, 10000, 5000],
            'term': ['60 months', '36 months', '36 months'],
            'int_rate': [15.27, 13.49, 12.69],
            'installment': [548.23, 339.31, 167.82],
            'grade': ['D', 'C', 'B'],
            'emp_length': [10, 0, 2],
            'home_ownership': ['RENT', 'RENT', 'RENT'],
            'annual_inc': [50000, 45000, 60000],
            'verification_status': ['Not Verified', 'Source Verified', 'Verified'],
            'dti': [20.5, 15.2, 18.7],
            'delinq_2yrs': [0, 0, 0],
            'inq_last_6mths': [2, 1, 0],
            'mths_since_last_delinq': [None, None, None],
            'open_acc': [8, 7, 6],
            'pub_rec': [0, 0, 0],
            'revol_bal': [15000, 12000, 8000],
            'revol_util': [65.2, 58.7, 45.3],
            'total_acc': [12, 10, 8],
            'initial_list_status': ['f', 'w', 'w'],
            'out_prncp': [0, 0, 0],
            'total_pymnt': [0, 0, 0],
            'total_rec_prncp': [0, 0, 0],
            'total_rec_int': [0, 0, 0],
            'total_rec_late_fee': [0, 0, 0],
            'recoveries': [0, 0, 0],
            'collection_recovery_fee': [0, 0, 0],
            'last_pymnt_amnt': [0, 0, 0],
            'tot_coll_amt': [0, 0, 0],
            'tot_cur_bal': [35000, 28000, 20000],
            'total_rev_hi_lim': [40000, 32000, 25000],
            'mths_since_earliest_cr_line': [120, 85, 60],
            'purpose': ['debt_consolidation', 'home_improvement', 'credit_card']
        }
        df = pd.DataFrame(data)
        
        print("\n=== Running Prediction Pipeline Test ===")
        print(f"Input data shape: {df.shape}")
        
        # Get predictions
        results = orchestrate_predictions_for_batch(df, db)
        
        print("\n=== Results Summary ===")
        print(f"Cumulative Expected Loss: {format_float(results.cumulative_expected_loss)}")
        print(f"Credit Risk Percentage: {format_float(results.credit_risk_percentage)}%")
        print(f"Defaulters Percentage: {format_float(results.defaulters_percentage)}%")
        
        print("\n=== Individual Results ===")
        for result in results.results:
            print(f"\nApplication ID: {result.application_db_id}")
            print(f"Status: {result.status_message}")
            print(f"PD Probability: {format_float(result.pd_ml_probability)}")
            print(f"Expected Loss: {format_float(result.expected_loss_ml)}")
            
    except Exception as e:
        print(f"\nERROR during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    test_prediction_pipeline()
