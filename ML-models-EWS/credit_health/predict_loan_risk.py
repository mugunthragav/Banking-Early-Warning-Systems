import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_scaler():
    """Load models and scaler."""
    try:
        pd_model = pickle.load(open('models/pd_model_logistic_regression.pkl', 'rb'))
        lgd_model_dt = pickle.load(open('models/lgd_model_decision_tree.pkl', 'rb'))
        lgd_model_svr = pickle.load(open('models/lgd_model_svr.pkl', 'rb'))
        ead_model = pickle.load(open('models/ead_model_xgboost.pkl', 'rb'))
        try:
            scaler = pickle.load(open('models/scaler.pkl', 'rb'))
            logging.info("Loaded scaler")
        except FileNotFoundError:
            logging.warning("Scaler file not found. Creating a default scaler.")
            scaler = StandardScaler()  # Fallback, but should be trained
        logging.info("Loaded models")
        return pd_model, lgd_model_dt, lgd_model_svr, ead_model, scaler
    except Exception as e:
        logging.error(f"Error loading models or scaler: {e}")
        raise

def predict_loan_risk(data, config, loan_id=None):
    """Predict loan risk metrics."""
    try:
        from src.data_preprocessing import preprocess_data
        pd_model, lgd_model_dt, lgd_model_svr, ead_model, scaler = load_model_and_scaler()

        # Preprocess data
        data_processed, _, df_features, _ = preprocess_data(data, config)
        logging.info(f"Input features: {df_features.columns.tolist()}")
        logging.info(f"Input feature values: {df_features.iloc[0].to_dict()}")

        # Scale features
        X_scaled = scaler.transform(df_features)
        logging.info(f"Scaled feature values: {X_scaled[0]}")

        # Predict PD
        pd_prob = pd_model.predict_proba(X_scaled)[:, 1]
        pd_pred = (pd_prob > 0.5).astype(int)
        logging.info(f"PD probabilities: {pd_prob}")

        # Predict LGD
        lgd_pred_dt = lgd_model_dt.predict(X_scaled)
        lgd_pred_svr = lgd_model_svr.predict(X_scaled)
        lgd_pred = 0.6 * lgd_pred_dt + 0.4 * lgd_pred_svr
        lgd_pred = np.clip(lgd_pred, 0, 1)
        logging.info(f"LGD predictions: DT={lgd_pred_dt}, SVR={lgd_pred_svr}, Ensemble={lgd_pred}")

        # Predict EAD
        ccf_pred = ead_model.predict(X_scaled)
        ccf_pred = np.clip(ccf_pred, 0, 1)
        ead_pred = ccf_pred * data['funded_amnt'].values
        logging.info(f"EAD predictions: CCF={ccf_pred}, EAD={ead_pred}")

        # Calculate Expected Loss
        expected_loss = pd_prob * lgd_pred * ead_pred

        # Prepare results
        results = pd.DataFrame({
            'Loan_ID': [loan_id] if loan_id else data.index,
            'Will_Default': ['Yes' if p == 1 else 'No' for p in pd_pred],
            'PD_Probability': pd_prob,
            'LGD_Percentage': lgd_pred,
            'Loss_Amount': lgd_pred * data['funded_amnt'].values,
            'EAD_Amount': ead_pred,
            'Expected_Loss': expected_loss
        })

        # Save results
        output_path = config['data']['output_path']
        results.to_csv(output_path, mode='a', header=not pd.io.common.file_exists(output_path), index=False)
        logging.info(f"Results saved to {output_path}")

        return results

    except Exception as e:
        logging.error(f"Error predicting loan risk: {e}")
        raise