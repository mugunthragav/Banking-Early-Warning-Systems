# backend/app/risk_models_ml/ead_ml_predictor.py
import pandas as pd
import numpy as np
# The call to perform_shared_initial_processing is REMOVED from inside this function.
# It will be done by the orchestrator service BEFORE this function is called.
from ..core.artifact_loader import feature_ead_order, ead_model # Corrected import

# This internal clean_column_names might do further EAD-specific adjustments
# or ensure a very specific format if the global one wasn't enough.
def clean_column_names_ead_specific(df_to_clean): # Renamed for clarity
    df_to_clean.columns = (
        df_to_clean.columns
        .str.strip() # May be redundant
        .str.lower() # May be redundant
        .str.replace(' ', '_') # May be redundant
        .str.replace(r'[^\w]', '_', regex=True) # May be redundant
    )
    return df_to_clean

def ead(df_after_initial_processing: pd.DataFrame): # Argument name changed for clarity
    """
    Predicts EAD.
    Args:
        df_after_initial_processing (pd.DataFrame): DataFrame that has ALREADY been processed by
                                                    perform_shared_initial_processing.
                                                    It's assumed to contain 'funded_amnt' if not dropped by NaNs.
    """
    # Make a copy to avoid modifying the DataFrame passed from the orchestrator
    df = df_after_initial_processing.copy()

    # --- EAD-Specific Preprocessing Starts Here ---
    # The line 'df = perform_shared_initial_processing(df)' is REMOVED from here.
    df = clean_column_names_ead_specific(df) # Apply EAD-specific column cleaning if necessary

    processed_df = df.copy()
    
    if 'funded_amnt' not in processed_df.columns:
      raise ValueError("Missing 'funded_amnt' column needed for EAD calculation.")
    
    funded_amnt = processed_df['funded_amnt'].astype(float)

    
    if 'term' in processed_df.columns:
        processed_df['term'] = processed_df['term'].str.replace('months', '').str.strip().astype(int)
        processed_df['term_60'] = (processed_df['term'] > 36).astype(int)
    
    purpose_values = ['credit_card', 'debt_consolidation', 'wedding', 'vacation',
                      'educational', 'home_improvement', 'house', 'major_purchase', 
                      'moving', 'medical', 'other', 'renewable_energy', 'small_business']
    
    if 'purpose' in processed_df.columns:
        processed_df['purpose'] = processed_df['purpose'].str.lower().str.replace(' ', '_')
        for value in purpose_values:
            processed_df[f'purpose_{value}'] = (processed_df['purpose'] == value).astype(int)
    
    grade_values = ['b', 'c', 'd', 'e', 'f', 'g']
    if 'grade' in processed_df.columns:
        processed_df['grade'] = processed_df['grade'].str.lower()
        for value in grade_values:
            processed_df[f'grade_{value}'] = (processed_df['grade'] == value).astype(int)
    
    home_ownership_values = ['rent', 'other', 'own']
    if 'home_ownership' in processed_df.columns:
        processed_df['home_ownership'] = processed_df['home_ownership'].str.lower()
        for value in home_ownership_values:
            processed_df[f'home_ownership_{value}'] = (processed_df['home_ownership'] == value).astype(int)

    verification_values = ['source_verified', 'verified']
    if 'verification_status' in processed_df.columns:
        processed_df['verification_status'] = processed_df['verification_status'].str.lower().str.replace(' ', '_')
        for value in verification_values:
            processed_df[f'verification_status_{value}'] = (processed_df['verification_status'] == value).astype(int)

    if 'initial_list_status' in processed_df.columns:
        processed_df['initial_list_status_w'] = (processed_df['initial_list_status'] == 'w').astype(int)

    feature_cols = [
        'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
        'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',
        'total_rec_late_fee', 'open_acc', 'tot_coll_amt', 'total_pymnt', 'last_pymnt_amnt', 'collection_recovery_fee',
        'mths_since_earliest_cr_line', 'pub_rec', 'revol_util', 'total_acc',
        'tot_cur_bal', 'total_rev_hi_lim', 'grade_b', 'grade_c', 'grade_d',
        'grade_e', 'grade_f', 'grade_g', 'term_60', 'home_ownership_other',
        'home_ownership_own', 'home_ownership_rent',
        'verification_status_source_verified', 'verification_status_verified',
        'initial_list_status_w', 'purpose_credit_card',
        'purpose_debt_consolidation', 'purpose_educational',
        'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase',
        'purpose_medical', 'purpose_moving', 'purpose_other',
        'purpose_renewable_energy', 'purpose_small_business',
        'purpose_vacation', 'purpose_wedding'
    ]

    # Check for missing columns
    missing_cols = [col for col in feature_cols if col not in processed_df.columns]
    if missing_cols:
      print(f"[EAD MODEL] Missing columns: {missing_cols}")
      raise ValueError("Input data is not in expected format. See logs for details.")

    final_df = pd.DataFrame()
    
    final_df = processed_df[feature_cols].astype(float)

    final_df = final_df[feature_ead_order]

    final_df['CCF'] = ead_model.predict(final_df)
    final_df['CCF'] = final_df['CCF'].clip(lower=0, upper=1)  # Ensure CCF is between 0 and 1
    final_df['EAD'] = final_df['CCF']*funded_amnt

    return final_df['EAD']