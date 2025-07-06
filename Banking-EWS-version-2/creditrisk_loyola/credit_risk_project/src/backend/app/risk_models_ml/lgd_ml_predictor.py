# backend/app/risk_models_ml/lgd_ml_predictor.py
import pandas as pd
import numpy as np
# No direct import of perform_shared_initial_processing needed here
from ..core.artifact_loader import lgd_model, lgd_scaler, feature_lgd_order # Corrected import

# This internal clean_column_names might do further LGD-specific adjustments
# or ensure a very specific format if the global one wasn't enough.
def clean_column_names_lgd_specific(df_to_clean): # Renamed for clarity
    df_to_clean.columns = (
        df_to_clean.columns
        .str.strip() # May be redundant
        .str.lower() # May be redundant
        .str.replace(' ', '_') # May be redundant
        .str.replace(r'[^\w]', '_', regex=True) # May be redundant if global did advanced cleaning
    )
    return df_to_clean

def lgd(df_after_initial_processing: pd.DataFrame):
    """
    Predicts LGD.
    Args:
        df_after_initial_processing (pd.DataFrame): DataFrame that has ALREADY been processed by
                                                    perform_shared_initial_processing.
    """
    # Make a copy to avoid modifying the DataFrame passed from the orchestrator
    df = df_after_initial_processing.copy()

    # --- LGD-Specific Preprocessing Starts Here ---
    df = clean_column_names_lgd_specific(df) # Apply LGD-specific column cleaning if necessary

    # The line '#processed_df = perform_shared_initial_processing(df)' is REMOVED
    # because df_after_initial_processing is already the result of that.
    processed_df = df.copy() 

    # Convert term to numeric and create dummy
    if 'term' in processed_df.columns:
        processed_df['term'] = processed_df['term'].str.replace('months', '').str.strip().astype(int)
        processed_df['term_60'] = (processed_df['term'] > 36).astype(int)

    # Purpose one-hot
    purpose_values = ['credit_card', 'debt_consolidation', 'wedding', 'vacation',
                      'educational', 'home_improvement', 'house', 'major_purchase', 
                      'moving', 'medical', 'other', 'renewable_energy', 'small_business']
    if 'purpose' in processed_df.columns:
        processed_df['purpose'] = processed_df['purpose'].str.lower().str.replace(' ', '_')
        processed_df['purpose'] = processed_df['purpose'].apply(lambda x: x if x in purpose_values else 'other')
        for value in purpose_values:
            processed_df[f'purpose_{value}'] = (processed_df['purpose'] == value).astype(int)

    # Grade one-hot (ref=A)
    grade_values = ['b', 'c', 'd', 'e', 'f', 'g']
    if 'grade' in processed_df.columns:
        processed_df['grade'] = processed_df['grade'].str.lower()
        for value in grade_values:
            processed_df[f'grade_{value}'] = (processed_df['grade'] == value).astype(int)

    # Home ownership one-hot (ref=MORTGAGE)
    home_ownership_values = ['rent', 'other', 'own']
    if 'home_ownership' in processed_df.columns:
        processed_df['home_ownership'] = processed_df['home_ownership'].str.lower()
        for value in home_ownership_values:
            processed_df[f'home_ownership_{value}'] = (processed_df['home_ownership'] == value).astype(int)

    # Verification one-hot (ref=Not Verified)
    verification_values = ['source_verified', 'verified']
    if 'verification_status' in processed_df.columns:
        processed_df['verification_status'] = processed_df['verification_status'].str.lower().str.replace(' ', '_')
        for value in verification_values:
            processed_df[f'verification_status_{value}'] = (processed_df['verification_status'] == value).astype(int)

    # Initial list status
    if 'initial_list_status' in processed_df.columns:
        processed_df['initial_list_status_w'] = (processed_df['initial_list_status'].str.lower() == 'w').astype(int)

    # Final required columns
    feature_cols = [
        'emp_length', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs',
        'inq_last_6mths', 'mths_since_last_delinq', 'revol_bal',
        'total_rec_prncp', 'collection_recovery_fee',
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
      print(f"[LGD MODEL] Missing columns: {missing_cols}")
      raise ValueError("Input data is not in expected format. See logs for details.")

    # Select and scale
    final_df = processed_df[feature_cols].astype(float)

    final_df = final_df[feature_lgd_order]

    final_df = pd.DataFrame(lgd_scaler.transform(final_df), columns=final_df.columns, index=final_df.index)
    
    final_df['recovery_rate'] = lgd_model.predict(final_df)
    final_df['recovery_rate'] = final_df['recovery_rate'].clip(lower=0, upper=1)
    final_df['lgd'] = 1 - final_df['recovery_rate']

    return final_df['lgd']