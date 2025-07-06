#risk_models/pd_ml_predictor.py
import pandas as pd
import numpy as np
from ..processing.initial_preprocessor import perform_shared_initial_processing
from ..core.artifact_loader import woe_dict, pd_model


def predict_credit_risk(df_after_initial_processing: pd.DataFrame) -> pd.DataFrame:

    # Make a copy to avoid modifying the DataFrame passed from the orchestrator
    df = df_after_initial_processing.copy()

    def clean_column_names(df):
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        return df

    def process_emp_length(df: pd.DataFrame) -> pd.DataFrame:
        df['emp_length'] = df['emp_length'].apply(str)
        return df

    def process_status_map(df: pd.DataFrame) -> pd.DataFrame:
        status_map = {
            'Current': 0, 'In Grace Period': 0, 'Fully Paid': 0,
            'Charged Off': 1, 'Default': 1,
            'Late (16-30 days)': 1, 'Late (31-120 days)': 1,
            'Does not meet the credit policy. Status:Fully Paid': 0,
            'Does not meet the credit policy. Status:Charged Off': 1
        }
        if 'loan_status' in df.columns:
            df['loan_status'] = df['loan_status'].map(status_map)
        return df

    def bin_column(df, col, min_val, max_val, num_bins, right=True, label_sep='-', overflow_label=True):
        bin_width = (max_val - min_val) / num_bins
        bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
        bin_edges.append(float('inf'))
        if overflow_label:
            labels = [f'{int(bin_edges[i])}{label_sep}{int(bin_edges[i+1])}' for i in range(num_bins)] + [f'>{int(bin_edges[num_bins])}']
        else:
            labels = [f'{int(bin_edges[i])}{label_sep}{int(bin_edges[i+1])}' for i in range(num_bins)]
        df[f'{col}_binned'] = pd.cut(
            df[col],
            bins=bin_edges,
            labels=labels,
            right=right,
            include_lowest=True
        )
        return df

    def apply_woe_transformation(df, woe_dict):
        for var in woe_dict:
            if var in df.columns:
                df[f'{var}_woe'] = df[var].astype(str).map(woe_dict[var])
        return df
    # Start processing
    df = clean_column_names(df)
    df = process_emp_length(df)
    df = process_status_map(df)

    # Binning specs
    binning_specs = [
        ('funded_amnt', 500, 35000, 10, False, '–'),
        ('funded_amnt_inv', 0, 35000, 10, False, '–'),
        ('installment', 15, 1409, 10, False, '–'),
        ('annual_inc', 1896, 751706, 5, True, '-'),
        ('dti', 0, 39, 10, False, '–'),
        ('delinq_2yrs', 0, 2, 2, True, '-'),
        ('inq_last_6mths', 0, 3, 3, True, '-'),
        ('mths_since_last_delinq', 0, 75, 5, True, '-'),
        ('revol_bal', 0, 51379, 5, True, '-'),
        ('out_prncp', 0, 22512, 5, True, '-'),
        ('total_pymnt', 0, 40444, 5, True, '-'),
        ('total_rec_prncp', 0, 35000, 10, False, '–'),
        ('total_rec_int', 0, 9682, 5, True, '-'),
        ('total_rec_late_fee', 0, 7, 1, True, '-'),
        ('last_pymnt_amnt', 0, 18117, 5, True, '-'),
        ('mths_since_earliest_cr_line', 73, 587, 10, False, '–'),
        ('open_acc', 0, 25, 4, True, '-'),
        ('pub_rec', 0, 1, 1, True, '-'),
        ('revol_util', 0, 100, 4, True, '-'),
        ('total_acc', 1, 60, 5, True, '-'),
        ('tot_coll_amt', 0, 847, 1, True, '-'),
        ('tot_cur_bal', 0, 480003, 4, True, '-'),
        ('total_rev_hi_lim', 0, 112499, 4, True, '-')
    ]
    for col, min_val, max_val, num_bins, right, sep in binning_specs:
        if col in df.columns:
            bin_column(df, col, min_val, max_val, num_bins, right=right, label_sep=sep)

    # Apply WoE transformations
    df = apply_woe_transformation(df, woe_dict)

    # Final selected features for model
    selected_features = [
        'total_rec_late_fee_binned_woe', 'total_pymnt_binned_woe', 'inq_last_6mths_binned_woe',
        'verification_status_woe', 'total_rec_int_binned_woe', 'last_pymnt_amnt_binned_woe',
        'funded_amnt_inv_binned_woe', 'grade_woe', 'out_prncp_binned_woe',
        'mths_since_last_delinq_binned_woe', 'installment_binned_woe', 'total_rec_prncp_binned_woe',
        'funded_amnt_binned_woe', 'delinq_2yrs_binned_woe', 'initial_list_status_woe',
        'mths_since_earliest_cr_line_binned_woe', 'tot_cur_bal_binned_woe', 'dti_binned_woe',
        'term_woe', 'revol_util_binned_woe', 'tot_coll_amt_binned_woe', 'open_acc_binned_woe',
        'pub_rec_binned_woe', 'emp_length_woe', 'revol_bal_binned_woe', 'purpose_woe',
        'annual_inc_binned_woe', 'total_acc_binned_woe', 'total_rev_hi_lim_binned_woe'
    ]

    X_pred = df.reindex(columns=selected_features)

    probability = pd_model.predict_proba(X_pred)[:, 1]
    predictions = pd_model.predict(X_pred)

    result = pd.DataFrame({'probability': probability, 'prediction': predictions}, index=df.index)
    return result