import pickle
import os
from dotenv import load_dotenv

# Load .env
load_dotenv(override=True)

MODEL_DIR = os.getenv("MODEL_DIR")
LGD_SCALER_FILE = os.getenv("LGD_SCALER_FILE")
LGD_MODEL_FILE = os.getenv("LGD_MODEL_FILE")
FEATURE_ORDER_LGD_FILE = os.getenv("FEATURE_ORDER_LGD_FILE")
EAD_MODEL_FILE=os.getenv("EAD_MODEL_FILE")
FEATURE_ORDER_EAD_FILE=os.getenv("FEATURE_ORDER_EAD_FILE")
WOE_DICT_FILE=os.getenv("WOE_DICT_FILE")
PD_MODEL_FILE=os.getenv("PD_MODEL_FILE")

# Full paths
lgd_scaler_path = os.path.join(MODEL_DIR, LGD_SCALER_FILE)
lgd_model_path = os.path.join(MODEL_DIR, LGD_MODEL_FILE)
lgd_feature_order_path = os.path.join(MODEL_DIR, FEATURE_ORDER_LGD_FILE)
ead_model_path = os.path.join(MODEL_DIR, EAD_MODEL_FILE)
ead_feature_order_path = os.path.join(MODEL_DIR, FEATURE_ORDER_EAD_FILE)
woe_dict_path = os.path.join(MODEL_DIR, WOE_DICT_FILE)
pd_model_path = os.path.join(MODEL_DIR, PD_MODEL_FILE)

# Load models
with open(lgd_scaler_path, 'rb') as f:
    lgd_scaler = pickle.load(f)
    
with open(lgd_model_path, 'rb') as f:
    lgd_model = pickle.load(f)

with open(lgd_feature_order_path, 'rb') as f:
    feature_lgd_order = pickle.load(f)

with open(ead_feature_order_path,'rb') as f:
    feature_ead_order = pickle.load(f)

with open(ead_model_path, 'rb') as f:
    ead_model = pickle.load(f)

with open(woe_dict_path, 'rb') as f:
    woe_dict = pickle.load(f)

with open(pd_model_path, 'rb') as f:
    pd_model = pickle.load(f)

