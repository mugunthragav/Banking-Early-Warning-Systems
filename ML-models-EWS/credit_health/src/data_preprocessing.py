import pandas as pd
import numpy as np
import logging
from datetime import datetime
import yaml # Import yaml here as it's used in load_config

# Set up logging (basic config might be handled in main, but good to have here too)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Use a logger instance instead of root logger for better practice
logger = logging.getLogger(__name__)
if not logger.handlers: # Prevent adding handlers multiple times if main also configures
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__) # Re-get after config


def load_config(config_path='config.yaml'):
    """Loads configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def load_data(data_path):
    """Loads data from a CSV file."""
    try:
        logger.info(f"Loading dataset from {data_path}")
        # Use low_memory=False to avoid DtypeWarning with mixed types
        data = pd.read_csv(data_path, low_memory=False)
        logger.info(f"Loaded dataset with shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise


def preprocess_data(data, config, is_training=True):
    """
    Preprocesses the loan data by engineering features, handling missing values,
    creating dummy variables, and defining target variables.
    """
    try:
        # Log the first few rows to inspect data format
        logger.info(f"Raw input data head:\n{data.head().to_markdown()}")

        # Engineer features
        # Extract integer term from string (e.g., ' 36 months' -> 36)
        if 'term' in data.columns:
            data['term_int'] = data['term'].str.extract(r'(\d+)').astype(float).fillna(0).astype(int) # Handle potential NaNs after extract
            logger.info("Engineered feature: term_int")
        else:
             logger.warning("'term' column not found for term_int engineering.")
             data['term_int'] = 0 # Add column with default value


        # Convert employment length to integer months
        if 'emp_length' in data.columns:
            # Using a dictionary replacement and handling NaN
            data['emp_length_int'] = data['emp_length'].replace({
                '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
                '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
                '10+ years': 10, np.nan: 0 # Treat NaN as 0 years
            }).astype(int)
            logger.info("Engineered feature: emp_length_int")
        else:
             logger.warning("'emp_length' column not found for emp_length_int engineering.")
             data['emp_length_int'] = 0 # Add column with default value


        # Convert date columns to datetime objects with the CORRECT format
        # The log shows 'YYYY-MM-DD' format, not 'Mon-YY'
        if 'issue_d' in data.columns:
            # Use errors='coerce' to turn unparseable dates into NaT (Not a Time)
            data['issue_d'] = pd.to_datetime(data['issue_d'], format='%Y-%m-%d', errors='coerce')
            logger.info("Converted 'issue_d' to datetime.")
        else:
            logger.warning("'issue_d' column not found for datetime conversion.")
            data['issue_d'] = pd.NaT # Add column with NaT

        if 'earliest_cr_line' in data.columns:
            # Use errors='coerce' to turn unparseable dates into NaT
            data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'], format='%Y-%m-%d', errors='coerce')
            logger.info("Converted 'earliest_cr_line' to datetime.")
        else:
            logger.warning("'earliest_cr_line' column not found for datetime conversion.")
            data['earliest_cr_line'] = pd.NaT # Add column with NaT


        # Calculate months since issue date and earliest credit line
        # Note: Using datetime.now() makes results non-reproducible.
        # Consider using a fixed reference date or difference between dates.
        if 'issue_d' in data.columns and not data['issue_d'].isnull().all():
             # Calculate difference in days, then convert to months
             data['mths_since_issue_d'] = (datetime.now() - data['issue_d']).dt.days // 30
             # Fill NaNs resulting from NaT issue_d
             data['mths_since_issue_d'] = data['mths_since_issue_d'].fillna(data['mths_since_issue_d'].median() if not data['mths_since_issue_d'].dropna().empty else 0).astype(int)
             logger.info("Engineered feature: mths_since_issue_d")
        else:
             logger.warning("'issue_d' data missing or all NaT for mths_since_issue_d engineering.")
             data['mths_since_issue_d'] = 0 # Add column with default value

        if 'earliest_cr_line' in data.columns and not data['earliest_cr_line'].isnull().all():
             # Calculate difference in days, then convert to months
             data['mths_since_earliest_cr_line'] = (datetime.now() - data['earliest_cr_line']).dt.days // 30
             # Fill NaNs resulting from NaT earliest_cr_line
             data['mths_since_earliest_cr_line'] = data['mths_since_earliest_cr_line'].fillna(data['mths_since_earliest_cr_line'].median() if not data['mths_since_earliest_cr_line'].dropna().empty else 0).astype(int)
             logger.info("Engineered feature: mths_since_earliest_cr_line")
        else:
             logger.warning("'earliest_cr_line' data missing or all NaT for mths_since_earliest_cr_line engineering.")
             data['mths_since_earliest_cr_line'] = 0 # Add column with default value


        # Impute NaNs for specified numerical features
        # Ensure columns exist before attempting imputation
        numerical_features_to_impute = [
            'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim',
            'installment', 'mths_since_last_delinq', 'mths_since_last_record',
            'revol_bal', 'revol_util', 'term_int', 'emp_length_int',
            'mths_since_issue_d', 'mths_since_earliest_cr_line' # Include engineered features
        ]
        imputed_cols = []
        for col in numerical_features_to_impute:
            if col in data.columns:
                # Impute with median, except for annual_inc where you specified 0
                imputation_value = 0 if col == 'annual_inc' else (data[col].median() if not data[col].dropna().empty else 0)
                data[col] = data[col].fillna(imputation_value)
                imputed_cols.append(col)
            else:
                logger.warning(f"Numerical imputation column '{col}' not found in data.")
                # If a numerical feature is expected but missing, add it with default 0s
                if col in config.get('features', {}).get('all', []):
                     data[col] = 0 # Add missing expected numerical feature
                     logger.warning(f"Added missing expected numerical feature '{col}' with 0s.")


        logger.info(f"Imputed NaN for numerical columns: {imputed_cols}")


        # Create dummy variables
        dummy_cols = ['grade', 'home_ownership', 'purpose', 'initial_list_status', 'verification_status']
        created_dummy_cols = []
        for col in dummy_cols:
            if col in data.columns:
                # Ensure the column is of object/category type before creating dummies
                if data[col].dtype == 'object' or isinstance(data[col].dtype, pd.CategoricalDtype):
                     dummies = pd.get_dummies(data[col], prefix=col, dummy_na=False)
                     # Avoid adding duplicate columns if they already exist (e.g., from preprocessed data)
                     new_dummy_cols = [c for c in dummies.columns if c not in data.columns]
                     data = pd.concat([data, dummies[new_dummy_cols]], axis=1)
                     created_dummy_cols.extend(new_dummy_cols)
                     # Log only the new dummy columns created
                     # logger.info(f"Created dummy variables for {col}: {new_dummy_cols}")
                else:
                     logger.warning(f"Column '{col}' is not of object/category type ({data[col].dtype}). Skipping dummy creation.")
            else:
                logger.warning(f"Dummy column source '{col}' not found in data.")
                # If a source column for dummies is missing, ensure the expected dummy columns are added with 0s later


        logger.info(f"Created dummy variables: {created_dummy_cols}")
        # logger.info(f"All columns after dummy creation: {list(data.columns)}") # Can be very long

        # Ensure all expected features (including dummies) are present
        # This step is crucial if the input CSV is already partially preprocessed
        expected_features = config['features']['all']
        missing_expected_features = [feature for feature in expected_features if feature not in data.columns]

        if missing_expected_features:
            logger.warning(f"Adding missing expected features with 0s: {missing_expected_features}")
            for feature in missing_expected_features:
                data[feature] = 0 # Add missing columns with default value 0

        # Ensure the order of columns matches the expected features list
        # This is important for consistent model input
        # Select only the expected features, dropping any extra columns
        data_processed = data[expected_features].copy() # Create a copy to avoid SettingWithCopyWarning
        logger.info(f"Selected final features in specified order.")
        logger.info(f"Final features shape: {data_processed.shape}")
        # logger.info(f"Final features head:\n{data_processed.head().to_markdown()}")


        # Create binary target for PD (1 for default, 0 for non-default)
        if is_training and 'loan_status' in data.columns:
            default_statuses = ['Charged Off', 'Default', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
            # Use .loc for assignment to avoid SettingWithCopyWarning if data was a slice
            data['target_pd'] = data['loan_status'].apply(lambda x: 1 if x in default_statuses else 0)
            logger.info(f"Created target_pd with values: {data['target_pd'].value_counts().to_dict()}")
        elif is_training:
             logger.warning("'loan_status' column not found for target_pd creation.")
             data['target_pd'] = 0 # Add column with default 0 if training but no loan_status
        else:
            # If not training, target_pd is not created based on loan_status
            # It might be provided separately or not needed depending on inference logic
            # For now, return None as originally intended if not training
            data['target_pd'] = None


        # Filter defaulted loans for LGD and EAD
        defaulted_loans = None
        defaulted_features = None
        if is_training and 'loan_status' in data.columns and 'recoveries' in data.columns and 'funded_amnt' in data.columns and 'total_rec_prncp' in data.columns:
            defaulted_statuses_lgd_ead = ['Charged Off', 'Default'] # Typically only fully defaulted for LGD/EAD training
            defaulted_loans = data[data['loan_status'].isin(defaulted_statuses_lgd_ead)].copy() # Use .copy()

            if not defaulted_loans.empty:
                # Calculate recovery rate (handle funded_amnt = 0 to avoid division by zero)
                defaulted_loans['recovery_rate'] = defaulted_loans.apply(
                    lambda row: row['recoveries'] / row['funded_amnt'] if row['funded_amnt'] > 0 else 0, axis=1
                )
                defaulted_loans['recovery_rate'] = defaulted_loans['recovery_rate'].clip(0, 1) # Clip between 0 and 1

                # Calculate CCF (handle funded_amnt = 0 to avoid division by zero)
                defaulted_loans['CCF'] = defaulted_loans.apply(
                     lambda row: (row['funded_amnt'] - row['total_rec_prncp']) / row['funded_amnt'] if row['funded_amnt'] > 0 else 0, axis=1
                )
                defaulted_loans['CCF'] = defaulted_loans['CCF'].clip(0, 1) # Clip between 0 and 1

                logger.info(f"Filtered and processed defaulted loans: {defaulted_loans.shape}")
                # Select features for defaulted loans, ensuring columns exist and are in order
                defaulted_features = defaulted_loans[expected_features].copy() # Use .copy()

            else:
                 logger.warning("No defaulted loans found for LGD/EAD training.")
                 defaulted_loans = pd.DataFrame(columns=list(data.columns) + ['recovery_rate', 'CCF']) # Return empty df with expected cols
                 defaulted_features = pd.DataFrame(columns=expected_features) # Return empty df with expected cols


        elif is_training:
             logger.warning("Missing columns ('loan_status', 'recoveries', 'funded_amnt', or 'total_rec_prncp') for LGD/EAD training. Skipping LGD/EAD data preparation.")
             defaulted_loans = pd.DataFrame(columns=list(data.columns) + ['recovery_rate', 'CCF'])
             defaulted_features = pd.DataFrame(columns=expected_features)
        # If not training, defaulted_loans and defaulted_features remain None as initialized


        # Return the original data DataFrame (potentially with new columns like target_pd),
        # the filtered defaulted_loans DataFrame, the features DataFrame for PD,
        # and the features DataFrame for defaulted loans (for LGD/EAD).
        # Ensure the returned features DataFrames only contain the expected columns.
        return data, defaulted_loans, data_processed, defaulted_features # Return data_processed for PD features


    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise

