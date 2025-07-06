# backend/app/calculations/expected_loss.py

import pandas as pd
import numpy as np

def calculate_el(
    pd_series: pd.Series | float,
    lgd_series: pd.Series | float,
    ead_series: pd.Series | float
) -> pd.Series | float:
    """
    Calculates Expected Loss (EL) from Probability of Default (PD),
    Loss Given Default (LGD), and Exposure at Default (EAD).

    EL = PD * LGD * EAD

    Args:
        pd_series (pd.Series | float): A Pandas Series (or single float) containing PD values (between 0 and 1).
        lgd_series (pd.Series | float): A Pandas Series (or single float) containing LGD values (typically between 0 and 1).
        ead_series (pd.Series | float): A Pandas Series (or single float) containing EAD values (monetary amount).

    Returns:
        pd.Series | float: A Pandas Series (or single float) containing the calculated Expected Loss values.
                           Returns NaN where any of the inputs are NaN or if inputs are not aligned.
    """
    try:
        # Ensure inputs can be multiplied (e.g., if they are Series, their indices should align)
        # If inputs are single float values, direct multiplication will work.
        # If inputs are Series, pandas handles element-wise multiplication based on index.
        
        el = pd_series * lgd_series * ead_series
        
        # Optional: You might want to add specific error handling or logging here
        # if pd_series, lgd_series, or ead_series have unexpected values (e.g., outside typical ranges, though clipping might be done earlier)

        return el

    except TypeError as te:
        print(f"Error calculating Expected Loss due to type mismatch: {te}")
        # In case of mixed types that pandas can't handle or other type errors,
        # return NaN or raise the error depending on desired behavior.
        # If inputs were expected to be Series, create a Series of NaNs with the same length.
        if isinstance(pd_series, pd.Series):
            return pd.Series([np.nan] * len(pd_series), index=pd_series.index)
        elif isinstance(lgd_series, pd.Series):
            return pd.Series([np.nan] * len(lgd_series), index=lgd_series.index)
        elif isinstance(ead_series, pd.Series):
            return pd.Series([np.nan] * len(ead_series), index=ead_series.index)
        else:
            return np.nan # For single float inputs
    except Exception as e:
        print(f"An unexpected error occurred during Expected Loss calculation: {e}")
        # Fallback for other errors
        if isinstance(pd_series, pd.Series):
            return pd.Series([np.nan] * len(pd_series), index=pd_series.index)
        else:
            return np.nan
'''
# --- Example Usage (optional, for testing this module directly) ---
if __name__ == '__main__':
    # Example with Pandas Series
    pd_values = pd.Series([0.1, 0.2, 0.05, np.nan])
    lgd_values = pd.Series([0.45, 0.50, 0.40, 0.55])
    ead_values = pd.Series([10000, 20000, 5000, 15000])

    expected_loss_values = calculate_el(pd_values, lgd_values, ead_values)
    print("--- Expected Loss (Series) ---")
    print(expected_loss_values)
    # Expected:
    # 0     450.0
    # 1    2000.0
    # 2     100.0
    # 3       NaN
    # dtype: float64

    # Example with single float values
    pd_single = 0.15
    lgd_single = 0.60
    ead_single = 25000

    expected_loss_single = calculate_el(pd_single, lgd_single, ead_single)
    print("\n--- Expected Loss (Single Float) ---")
    print(expected_loss_single)
    # Expected: 2250.0

    # Example with type mismatch (if not handled by pandas element-wise ops)
    # This specific case might work with pandas due to broadcasting, but robust error handling is good.
    # pd_mixed = pd.Series([0.1, 0.2])
    # lgd_mixed = 0.5
    # ead_mixed = "not_a_number" # This would cause a TypeError
    # try:
    #     expected_loss_mixed = calculate_el(pd_mixed, lgd_mixed, ead_mixed)
    #     print("\n--- Expected Loss (Mixed Types - should error or be NaN) ---")
    #     print(expected_loss_mixed)
    # except Exception as e:
    #      print(f"\nError with mixed types as expected: {e}")
    '''