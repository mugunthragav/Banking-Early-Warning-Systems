# backend/app/core/config.py

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional

# Load environment variables from .env file at the backend's root
# The .env file should be in the 'backend/' directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent # This should point to 'backend/'
load_dotenv(dotenv_path=BASE_DIR / ".env")

# --- General Application Settings ---
APP_NAME: str = "Credit Risk Suite API"
API_V1_STR: str = "/api/v1"

# --- Artifact Configuration ---
# MODEL_DIR from .env is the primary source for the base path to artifacts
# ARTIFACTS_PATH will be constructed using MODEL_DIR
# Ensure MODEL_DIR in your .env points to './artifacts' relative to the backend/ directory.
MODEL_DIR_ENV: Optional[str] = os.getenv("MODEL_DIR")
if MODEL_DIR_ENV is None:
    # Default path if MODEL_DIR is not set in .env, assuming 'artifacts' is in 'backend/'
    ARTIFACTS_PATH: Path = BASE_DIR / "artifacts"
    print(f"Warning: MODEL_DIR environment variable not set. Defaulting ARTIFACTS_PATH to: {ARTIFACTS_PATH}")
else:
    # If MODEL_DIR is relative (e.g., "./artifacts"), make it absolute from BASE_DIR
    if not os.path.isabs(MODEL_DIR_ENV):
        ARTIFACTS_PATH: Path = (BASE_DIR / MODEL_DIR_ENV).resolve()
    else:
        ARTIFACTS_PATH: Path = Path(MODEL_DIR_ENV)

# Specific artifact filenames (these should match keys in your .env for artifact_loader.py)
# artifact_loader.py will use os.getenv for these, but we can define them here for clarity
# or as fallbacks if needed. The names here should match your actual file names.
PD_MODEL_FILENAME: str = os.getenv("PD_MODEL_FILE", "pd_model.pkl")
WOE_DICT_FILENAME: str = os.getenv("WOE_DICT_FILE", "woe_dict.pkl")
PD_FEATURES_ORDER_FILENAME: str = os.getenv("PD_FEATURES_ORDER_FILE", "pd_features_order.pkl") # If you have one

LGD_MODEL_FILENAME: str = os.getenv("LGD_MODEL_FILE", "lgd_mlp_model.pkl")
LGD_SCALER_FILENAME: str = os.getenv("LGD_SCALER_FILE", "min_max_scaler.pkl")
LGD_FEATURE_ORDER_FILENAME: str = os.getenv("FEATURE_ORDER_LGD_FILE", "feature_order_lgd_mlp.pkl")

EAD_MODEL_FILENAME: str = os.getenv("EAD_MODEL_FILE", "ead_meta_model.pkl")
EAD_FEATURE_ORDER_FILENAME: str = os.getenv("FEATURE_ORDER_EAD_FILE", "feature_order_ead_meta_model.pkl")

# --- Database Configuration ---
DB_USER: str = " your_db_user"  # Replace with your actual username or set in .env
DB_PASSWORD: str = "your_db_password"  # Replace with your actual password or set in .env
DB_HOST: str = "localhost"  # Replace with your actual host or set in .env
DB_PORT: str = "3306"  # Default MySQL port
DB_NAME: str = "credit_risk_db"  # Replace with your actual database name or set in .env

# Example for MySQL: "mysql+mysqlconnector://user:password@host:port/db_name"
# Example for PostgreSQL: "postgresql+psycopg2://user:password@host:port/db_name"
# Example for SQLite: "sqlite:///./credit_risk_app.db" (file will be in backend/ directory)
SQLALCHEMY_DATABASE_URL: str = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# For SQLite, you might want a simpler default if other DB vars are not set:
# if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]) and "sqlite" not in SQLALCHEMY_DATABASE_URL.lower():
# SQLALCHEMY_DATABASE_URL = f"sqlite:///{BASE_DIR / 'credit_risk_app.db'}"


# --- Lookup Scoring Configuration ---
DEFAULT_LOOKUP_PD: float = 0.50
DEFAULT_LOOKUP_LGD: float = 0.50
DEFAULT_LOOKUP_EAD: float = 1000.0

# Funded Amount Binning for Lookup Preprocessor
# These MUST align with how 'funded_amnt_binned' was created in your historical CSV for lookup
# Example bins, replace with your actual configuration
LOOKUP_FUNDED_AMNT_BINS: List[float | int] = [0, 500, 3950, 7400, 10850, 14300, 17750, 21200, 24650, 28100, 31550, 35000, float('inf')]
LOOKUP_FUNDED_AMNT_LABELS: List[str] = [
    "<500", "500-3950", "3950-7400", "7400-10850", "10850-14300",
    "14300-17750", "17750-21200", "21200-24650", "24650-28100",
    "28100-31550", "31550-35000", ">35000"
]
# Ensure len(LOOKUP_FUNDED_AMNT_LABELS) == len(LOOKUP_FUNDED_AMNT_BINS) - 1




# --- Orchestrator Configuration (constants already in orchestrator, but good to have them centrally) ---
# These are copied from your prediction_orchestration_service.py, good to centralize them here.
MANDATORY_INPUT_COLUMNS_CLEANED: List[str] = sorted([
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


# In config.py
LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY")  # Load from environment variable
LLM_MODEL_NAME: Optional[str] = os.getenv("LLM_MODEL_NAME", "gpt-3.5-turbo")

KEY_FEATURES_FOR_AI_INTERPRETER: List[str] = ['loan_amnt', 'purpose', 'annual_inc', 'dti', 'grade']
CONTACT_MUKUNTH_MESSAGE: str = "Contact Mukunth"

# Default values for ML predictions if a model fails (used by orchestrator)
DEFAULT_PD_ML_PROBABILITY: float = 0.99 # High risk default
DEFAULT_PD_ML_CLASS: int = 1
DEFAULT_LGD_ML_VALUE: float = 0.90 # High risk default
DEFAULT_EAD_ML_VALUE: float = 25000 # Example, adjust as needed


# --- CORS Configuration (for FastAPI main.py) ---
CORS_ALLOWED_ORIGINS: List[str] = [
    os.getenv("FRONTEND_URL", "http://localhost:4200"), # Default Angular dev port
    # Add other origins if needed, e.g., your deployed frontend URL
]

# You can add more configurations here as your application grows.
# For example, logging configurations, external service URLs, etc.

print(f"INFO: Artifacts path configured to: {ARTIFACTS_PATH}")
print(f"INFO: Database URL configured to: {SQLALCHEMY_DATABASE_URL if 'your_db_password' not in SQLALCHEMY_DATABASE_URL else 'mysql+mysqlconnector://USER:****@HOST:PORT/DB_NAME'}")
if LLM_API_KEY == "YOUR_LLM_API_KEY_HERE" or not LLM_API_KEY:
    print("Warning: LLM_API_KEY is not set or using placeholder. AI Interpreter may not function.")

# Load .env file (assuming .env is in backend/ and config.py is in backend/app/core/)
BASE_DIR_FOR_ENV = Path(__file__).resolve().parent.parent.parent # Should point to backend/
load_dotenv(dotenv_path=BASE_DIR_FOR_ENV / ".env")