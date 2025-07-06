# backend/app/processing/initial_preprocessor.py
import pandas as pd
import re # For more advanced column cleaning if needed

def perform_shared_initial_processing(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Performs shared initial preprocessing on the raw input DataFrame.
    This function should encapsulate the logic of your original 'preprocess_input'.
    """
    df = df_raw.copy()

    # Step 1: Global Column Name Cleaning (if this was part of your original preprocess_input or desired now)
    # This makes column names consistent before any further operations.
    cleaned_columns = []
    for col in df.columns:
        new_col = str(col).strip().lower()
        new_col = re.sub(r'\s+', '_', new_col)          # Replace spaces with _
        new_col = re.sub(r'[^0-9a-zA-Z_]+', '', new_col) # Remove other non-alphanumeric
        cleaned_columns.append(new_col)
    df.columns = cleaned_columns

    # Step 2: Add any other logic that was in your original 'preprocess_input' function.
    # For example, if it handled specific data type conversions or initial feature engineering
    # that was common to EAD and LGD before their own detailed preprocessing.
    # (This part depends on what your original 'preprocess_input' did)
    # --- Placeholder for your original preprocess_input logic ---
    # Example: df['some_column'] = df['some_column'].as_type(float)
    # Example: df = some_other_initial_transformation(df)
    # --- End of placeholder ---

    # Step 3: Handle Missing Values by Dropping Rows (as per your latest requirement)
    # This step ensures that data fed to the individual model predictors has no missing values at the row level.
    df.dropna(how='any', inplace=True) # Drops rows with any NaN in any column

    if df.empty:
        print("Warning: DataFrame is empty after dropping rows with missing values. Check raw input and NaN handling.")
        # Depending on requirements, you might raise an error or return the empty DataFrame.

    return df