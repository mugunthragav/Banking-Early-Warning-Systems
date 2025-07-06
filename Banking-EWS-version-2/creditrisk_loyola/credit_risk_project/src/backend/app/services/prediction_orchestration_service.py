import pandas as pd
from typing import List, Dict, Any, Tuple
from sqlalchemy.orm import Session
import re
import numpy as np

# Internal application imports
from ..processing.initial_preprocessor import perform_shared_initial_processing
from ..risk_models_ml import pd_ml_predictor, lgd_ml_predictor, ead_ml_predictor
from ..calculations.expected_loss import calculate_el
from ..ai_interpreter.result_interpreter_service import generate_aggregate_metrics_summary
from ..schemas.loan_application_schemas import LeanLoanPredictionResult, AggregatedPredictionResponse
from ..core import config
from ..database import crud, model

# --- Configuration ---
MANDATORY_INPUT_COLUMNS_CLEANED = sorted([
    'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
    'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc',
    'verification_status', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'initial_list_status',
    'out_prncp', 'total_pymnt', 'total_rec_prncp', 'total_rec_int',
    'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
    'last_pymnt_amnt', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim',
    'mths_since_earliest_cr_line', 'purpose'
])

CONTACT_MUKUNTH_MESSAGE = "Contact Mukunth"


def _clean_column_names_globally(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()
    cleaned_columns = []
    for col in df_copy.columns:
        new_col = str(col).strip().lower()
        new_col = new_col.replace(' ', '_')
        new_col = re.sub(r'[^\w_]', '', new_col)
        cleaned_columns.append(new_col)
    df_copy.columns = cleaned_columns
    return df_copy


def _validate_input_columns(df_cleaned_cols: pd.DataFrame) -> pd.DataFrame:
    current_mandatory_list = MANDATORY_INPUT_COLUMNS_CLEANED
    missing_mandatory = [col for col in current_mandatory_list if col not in df_cleaned_cols.columns]
    if missing_mandatory:
        raise ValueError(f"Missing mandatory columns in input: {', '.join(missing_mandatory)}")
    return df_cleaned_cols[current_mandatory_list]


def create_base_log_data(raw_row_series, loan_amnt=None, status_message=None):
    """Create a base log entry with only valid database fields."""
    log_entry_data = {}
    
    # Copy only the fields that match our database model
    for field in MANDATORY_INPUT_COLUMNS_CLEANED:
        if field in raw_row_series:
            log_entry_data[field] = raw_row_series[field]
    
    # Override with provided values
    if loan_amnt is not None:
        log_entry_data['loan_amnt'] = loan_amnt
    if status_message is not None:
        log_entry_data['status_message'] = status_message
    
    # Initialize ML fields as None
    ml_fields = ['pd_ml_probability', 'pd_ml_prediction', 'lgd_ml_ann', 
                 'recovery_rate_ml', 'ead_ml_meta', 'expected_loss_ml', 
                 'probability_of_repayment']
    for field in ml_fields:
        log_entry_data[field] = None
        
    return log_entry_data

def orchestrate_predictions_for_batch(
    raw_applications_df: pd.DataFrame,
    db: Session
) -> AggregatedPredictionResponse:
    """
    Orchestrates the prediction pipeline for a batch of loan applications.
    Returns an AggregatedPredictionResponse object containing ML predictions, 
    aggregate metrics and their AI summary.
    """
    individual_results_list = []
    
    try:
        globally_cleaned_df = _clean_column_names_globally(raw_applications_df)
        validated_input_df = _validate_input_columns(globally_cleaned_df)
    except ValueError as e:
        print(f"[Orchestrator] Input column validation failed: {e}")
        return AggregatedPredictionResponse(
            cumulative_expected_loss=0.0,
            credit_risk_percentage=0.0,
            defaulters_percentage=0.0,
            aggregate_metrics_ai_summary=f"Batch processing error: {str(e)}",
            results=[],
            batch_id=None,
            processing_summary={'error': str(e), 'total_processed': 0, 'successfully_processed': 0}
        )

    initially_processed_df = perform_shared_initial_processing(validated_input_df.copy())

    if initially_processed_df.empty and not raw_applications_df.empty:
        print("[Orchestrator] DataFrame is empty after initial processing (all rows may have had NaNs or input was empty).")
        temp_lean_results = []
        for index, raw_row_series in raw_applications_df.iterrows():
            log_entry_data = create_base_log_data(
                raw_row_series,
                loan_amnt=raw_row_series.get('loan_amnt'),
                status_message="Input data contained missing values in critical fields; row dropped."
            )
            
            try:
                print(f"[Orchestrator] Saving dropped row data to database:")
                print(f"[Orchestrator] Loan amount: {log_entry_data.get('loan_amnt')}")
                print(f"[Orchestrator] Status: {log_entry_data.get('status_message')}")
                
                db_log_entry = crud.create_application_log_entry(db=db, application_data=log_entry_data)
                print(f"[Orchestrator] Successfully saved dropped row with ID: {db_log_entry.id}")
                temp_lean_results.append(LeanLoanPredictionResult(
                    application_db_id=db_log_entry.id,
                    status_message=log_entry_data['status_message']
                ))
            except Exception as e:
                print(f"[Orchestrator] Error saving dropped row to database: {e}")
                temp_lean_results.append(LeanLoanPredictionResult(
                    application_db_id=None,
                    status_message=f"Error logging dropped row to database: {str(e)}"
                ))
                
        return AggregatedPredictionResponse(
            cumulative_expected_loss=0.0,
            credit_risk_percentage=0.0,
            defaulters_percentage=0.0,
            aggregate_metrics_ai_summary="All applications in the batch were dropped during initial processing.",
            results=temp_lean_results,
            batch_id=None,
            processing_summary={'error': "All rows dropped", 'total_processed': len(raw_applications_df), 'successfully_processed': 0}
        )
    elif raw_applications_df.empty:
        return AggregatedPredictionResponse(
            cumulative_expected_loss=0.0, credit_risk_percentage=0.0, defaulters_percentage=0.0,
            aggregate_metrics_ai_summary="Input batch was empty.", results=[], batch_id=None,
            processing_summary={'message': "Empty input batch", 'total_processed': 0, 'successfully_processed': 0}
        )

    for original_index, raw_row_series in raw_applications_df.iterrows():
        current_app_result = raw_row_series.to_dict()
        current_app_result['application_id_temp'] = original_index 
        current_app_result['status_message'] = "Processing"

        if original_index in validated_input_df.index:
            current_app_result['loan_amnt'] = validated_input_df.loc[original_index, 'loan_amnt']
        else:
            current_app_result['loan_amnt'] = raw_row_series.get('loan_amnt')

        if original_index not in initially_processed_df.index:
            current_app_result['status_message'] = "Row dropped during initial processing (contained NaNs)."
            for score_field in ['pd_ml_probability', 'pd_ml_prediction', 'lgd_ml_ann', 
                              'recovery_rate_ml', 'ead_ml_meta', 'expected_loss_ml', 
                              'probability_of_repayment']:
                current_app_result[score_field] = None
            
            if not isinstance(current_app_result, dict):
                raise TypeError(f"Type error: current_app_result became {type(current_app_result)} for dropped row.")
            individual_results_list.append(current_app_result)
            continue

        single_app_df_for_models = initially_processed_df.loc[[original_index]].copy()

        try:
            pd_results = pd_ml_predictor.predict_credit_risk(single_app_df_for_models)
            current_app_result['pd_ml_probability'] = pd_results['probability'].iloc[0]
            current_app_result['pd_ml_prediction'] = int(pd_results['prediction'].iloc[0])
            current_app_result['probability_of_repayment'] = 1.0 - current_app_result['pd_ml_probability']
        except Exception as e:
            print(f"[Orchestrator] Error in PD ML for index {original_index}: {e}")
            current_app_result.update({'pd_ml_probability': None, 'pd_ml_prediction': None, 'probability_of_repayment': None})
            current_app_result['status_message'] = config.CONTACT_MUKUNTH_MESSAGE

        try:
            lgd_val = lgd_ml_predictor.lgd(single_app_df_for_models).iloc[0]
            current_app_result['lgd_ml_ann'] = lgd_val
            current_app_result['recovery_rate_ml'] = 1.0 - lgd_val if lgd_val is not None else None
        except Exception as e:
            print(f"[Orchestrator] Error in LGD ML for index {original_index}: {e}")
            current_app_result.update({'lgd_ml_ann': None, 'recovery_rate_ml': None})
            current_app_result.setdefault('status_message', config.CONTACT_MUKUNTH_MESSAGE)

        try:
            ead_val = ead_ml_predictor.ead(single_app_df_for_models).iloc[0]
            current_app_result['ead_ml_meta'] = ead_val
        except Exception as e:
            print(f"[Orchestrator] Error in EAD ML for index {original_index}: {e}")
            current_app_result['ead_ml_meta'] = None
            current_app_result.setdefault('status_message', config.CONTACT_MUKUNTH_MESSAGE)

        if all(current_app_result.get(k) is not None for k in ['pd_ml_probability', 'lgd_ml_ann', 'ead_ml_meta']):
            current_app_result['expected_loss_ml'] = calculate_el(
                current_app_result['pd_ml_probability'], 
                current_app_result['lgd_ml_ann'], 
                current_app_result['ead_ml_meta']
            )
        else:
            current_app_result['expected_loss_ml'] = None

        if current_app_result['status_message'] == "Processing":
            current_app_result['status_message'] = "Successfully processed."
        
        if not isinstance(current_app_result, dict):
            error_message = (
                f"Critical type error before appending to individual_results_list: "
                f"current_app_result is not a dict for index {original_index}! "
                f"Type: {type(current_app_result)}, Value: {str(current_app_result)[:200]}"
            )
            print(error_message)
            individual_results_list.append({
                'application_id_temp': original_index,
                'status_message': 'Internal error: result item was not a dictionary.',
                'loan_amnt': current_app_result.get('loan_amnt') if isinstance(current_app_result, dict) else None
            })
        else:
            individual_results_list.append(current_app_result)

    final_logged_results_for_lean_output = []
    for app_res_dict in individual_results_list:
        if not isinstance(app_res_dict, dict):
            print(f"Skipping non-dict item in individual_results_list: {app_res_dict}")
            final_logged_results_for_lean_output.append({
                'application_db_id': None,
                'status_message': 'Internal error: item in results list was not a dictionary.'
            })
            continue

        # Clean the data for database insertion, removing any fields not in our model
        db_log_data = clean_data_for_db(app_res_dict)
        
        try:
            print(f"[Orchestrator] Preparing to save data to database:")
            print(f"[Orchestrator] Loan amount: {db_log_data.get('loan_amnt')}")
            print(f"[Orchestrator] PD: {db_log_data.get('pd_ml_probability')}")
            print(f"[Orchestrator] LGD: {db_log_data.get('lgd_ml_ann')}")
            print(f"[Orchestrator] EAD: {db_log_data.get('ead_ml_meta')}")
            print(f"[Orchestrator] Expected Loss: {db_log_data.get('expected_loss_ml')}")
            
            db_log_entry = crud.create_application_log_entry(db=db, application_data=db_log_data)
            print(f"[Orchestrator] Successfully saved entry with ID: {db_log_entry.id}")
            app_res_dict['application_db_id'] = db_log_entry.id
            final_logged_results_for_lean_output.append(app_res_dict)
        except Exception as e:
            print(f"[Orchestrator] Error logging application to DB: {e}")
            print(f"[Orchestrator] Full data that failed to save: {db_log_data}")
            app_res_dict.setdefault('status_message', f"Error logging results to database: {str(e)}")
            app_res_dict['application_db_id'] = None
            
            # Initialize all model fields to None for failed entries
            for field in LeanLoanPredictionResult.model_fields:
                app_res_dict.setdefault(field, None)
            final_logged_results_for_lean_output.append(app_res_dict)

    sum_expected_loss_ml = 0.0
    sum_loan_amnt = 0.0
    default_count = 0
    successfully_processed_count = 0

    for res in final_logged_results_for_lean_output:
        if not isinstance(res, dict):
            print(f"Skipping non-dict item in final_logged_results_for_lean_output for aggregation: {type(res)}")
            continue
        
        current_loan_amnt = res.get('loan_amnt')
        if res.get('expected_loss_ml') is not None and res.get('pd_ml_prediction') is not None and current_loan_amnt is not None:
            successfully_processed_count += 1
            sum_expected_loss_ml += res['expected_loss_ml']
            try:
                sum_loan_amnt += float(current_loan_amnt)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert loan_amnt '{current_loan_amnt}' to float for app_db_id {res.get('application_db_id')}.")

            if res['pd_ml_prediction'] == 1:
                default_count += 1
        elif current_loan_amnt is None and res.get('status_message') == "Successfully processed.":
            print(f"Warning: loan_amnt is missing for successfully processed app_db_id {res.get('application_db_id')}. It will be excluded from Credit Risk %.")


    cumulative_el = sum_expected_loss_ml
    credit_risk_pct = (sum_expected_loss_ml / sum_loan_amnt) if sum_loan_amnt > 0 else 0.0
    defaulters_pct = (default_count / successfully_processed_count) if successfully_processed_count > 0 else 0.0
    
    aggregate_ai_summary_text = "AI summary for aggregate metrics is unavailable."
    if successfully_processed_count > 0:
        try:
            llm_config_dict = {'api_key': config.LLM_API_KEY, 'model_name': config.LLM_MODEL_NAME}
            if not (llm_config_dict.get('api_key') and isinstance(llm_config_dict.get('api_key'), str) and llm_config_dict.get('api_key').startswith("sk-") and llm_config_dict.get('model_name')):
                print("[Orchestrator] OpenAI API key or model name is not configured or invalid. Skipping aggregate summary.")
                aggregate_ai_summary_text = "AI summary generation skipped due to missing or invalid OpenAI configuration."
            else:
                aggregate_ai_summary_text = generate_aggregate_metrics_summary(
                    cumulative_expected_loss=cumulative_el,
                    credit_risk_percentage=credit_risk_pct,
                    defaulters_percentage=defaulters_pct,
                    llm_config=llm_config_dict
                )
        except Exception as e:
            print(f"[Orchestrator] Error generating aggregate AI summary: {e}")
            aggregate_ai_summary_text = f"Error during generation of aggregate AI summary: {str(e)[:100]}"

    lean_results_for_response: List[LeanLoanPredictionResult] = []
    for res_dict in final_logged_results_for_lean_output:
        if not isinstance(res_dict, dict):
            print(f"Skipping non-dict item for LeanLoanPredictionResult creation: {type(res_dict)}")
            lean_results_for_response.append(LeanLoanPredictionResult(
                application_db_id=None,
                status_message="Internal error: result item was not a dictionary for lean output."
            ))
            continue
        lean_data = {field: res_dict.get(field) for field in LeanLoanPredictionResult.model_fields}
        lean_results_for_response.append(LeanLoanPredictionResult(**lean_data))

    total_applications_in_batch = len(raw_applications_df)
    applications_processed_through_loop = len(individual_results_list)
    
    processing_summary_dict = {
        "total_applications_in_batch": total_applications_in_batch,
        "applications_attempted_processing": applications_processed_through_loop,
        "successfully_scored_applications": successfully_processed_count,
        "applications_dropped_preprocessing": total_applications_in_batch - applications_processed_through_loop,
    }

    return AggregatedPredictionResponse(
        cumulative_expected_loss=cumulative_el,
        credit_risk_percentage=credit_risk_pct,
        defaulters_percentage=defaulters_pct,
        aggregate_metrics_ai_summary=aggregate_ai_summary_text,
        results=lean_results_for_response,
        batch_id=None,
        processing_summary=processing_summary_dict
    )

def clean_data_for_db(app_res_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Clean application data to ensure only valid fields are sent to the database."""
    # Only keep fields that match our ApplicationLog model
    valid_fields = [
        'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate',
        'installment', 'grade', 'emp_length', 'home_ownership', 'annual_inc',
        'verification_status', 'dti', 'delinq_2yrs', 'inq_last_6mths',
        'mths_since_last_delinq', 'open_acc', 'pub_rec', 'revol_bal',
        'revol_util', 'total_acc', 'initial_list_status', 'out_prncp',
        'total_pymnt', 'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee',
        'recoveries', 'collection_recovery_fee', 'last_pymnt_amnt', 'tot_coll_amt',
        'tot_cur_bal', 'total_rev_hi_lim', 'mths_since_earliest_cr_line',
        'purpose', 'status_message', 'pd_ml_probability', 'pd_ml_prediction',
        'probability_of_repayment', 'lgd_ml_ann', 'recovery_rate_ml',
        'ead_ml_meta', 'expected_loss_ml', 'ai_interpretation_text'
    ]
    
    cleaned_data = {}
    for field in valid_fields:
        if field in app_res_dict:
            value = app_res_dict[field]
            # Convert numpy data types to Python built-in types
            if isinstance(value, np.number):
                if np.issubdtype(type(value), np.integer):
                    value = int(value)
                elif np.issubdtype(type(value), np.floating):
                    value = float(value)
            cleaned_data[field] = value
    
    return cleaned_data