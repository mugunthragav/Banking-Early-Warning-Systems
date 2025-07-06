import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from datetime import datetime
import logging
from src.data_preprocessing import load_config
from src.expected_loss import calculate_expected_loss

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_basel_report():
    """Generate Basel III report with portfolio-level metrics."""
    try:
        # Load preprocessed data and models
        logging.info("Loading preprocessed data and models...")
        config = load_config('config.yaml')
        data = pd.read_csv('data/loan_data_2007_2014_preprocessed.csv')

        # Load the scaler and models
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/pd_model_xgboost.pkl', 'rb') as f:
            pd_model = pickle.load(f)
        with open('models/lgd_model_dt.pkl', 'rb') as f:
            lgd_model = pickle.load(f)
        with open('models/ead_model_gbm.pkl', 'rb') as f:
            ead_model = pickle.load(f)

        # Select features for prediction (ensure the same order as during training)
        feature_columns = config['features']['all']
        logging.info(f"Expected features: {feature_columns}")
        X = data[feature_columns]

        # Verify feature names match the scaler's expectations
        logging.info(f"Features in X: {list(X.columns)}")
        if list(X.columns) != feature_columns:
            raise ValueError("Feature names or their order in X do not match the expected features from config.yaml")

        # Scale features
        X_scaled = scaler.transform(X)

        # Predict PD, LGD, and EAD
        logging.info("Predicting PD, LGD, and EAD...")
        pd_pred = pd_model.predict_proba(X_scaled)[:, 1]
        lgd_pred = lgd_model.predict(X_scaled)
        ead_pred = ead_model.predict(X_scaled)

        # Clip predictions to valid ranges
        pd_pred = np.clip(pd_pred, 0, 1)
        lgd_pred = np.clip(lgd_pred, 0, 1)
        ead_pred = np.clip(ead_pred, 0, 1)

        # Calculate Expected Loss (using src/expected_loss.py)
        logging.info("Calculating Expected Loss...")
        # Since we only have one LGD model (Decision Tree), use lgd_pred for both lgd_dt and lgd_svr
        result, summary = calculate_expected_loss(data, pd_pred, lgd_pred, lgd_pred, ead_pred)

        # Add predictions to the dataset
        data['PD_Probability'] = pd_pred
        data['LGD_Prediction'] = lgd_pred
        data['EAD_Prediction'] = ead_pred
        data['Expected_Loss'] = result['EL_DT']  # Use EL_DT (since lgd_dt and lgd_svr are the same)

        # Calculate Risk-Weighted Assets (RWA) using Basel III formula
        logging.info("Calculating Risk-Weighted Assets (RWA)...")
        # Simplified Basel III RWA calculation: RWA = EAD * Risk Weight
        # Risk Weight approximated based on PD (can be refined with more detailed Basel III rules)
        data['Risk_Weight'] = data['PD_Probability'].apply(lambda pd: 0.5 if pd < 0.03 else (0.75 if pd < 0.1 else 1.5))
        data['RWA'] = data['EAD_Prediction'] * data['funded_amnt'] * data['Risk_Weight']

        # Portfolio-level metrics
        total_rwa = data['RWA'].sum()
        total_expected_loss = data['Expected_Loss'].sum()
        capital_requirement = total_rwa * 0.08  # Basel III: 8% capital requirement
        portfolio_funded = data['funded_amnt'].sum()

        # Generate Basel III report
        logging.info("Generating Basel III report...")
        report = f"""
Basel III Report
=================
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Funded Amount: ${portfolio_funded:,.2f}
Total Expected Loss: ${total_expected_loss:,.2f}
Total Risk-Weighted Assets (RWA): ${total_rwa:,.2f}
Capital Requirement (8% of RWA): ${capital_requirement:,.2f}
EL/Funded Amount Ratio: {total_expected_loss / portfolio_funded:.4f}
=================
Summary Statistics:
{summary.to_string()}
        """

        # Save the report
        with open('basel_report.txt', 'w') as f:
            f.write(report)
        logging.info("Saved Basel III report to basel_report.txt")

        # Save the dataset with predictions
        data.to_csv('data/loan_data_with_predictions.csv', index=False)
        logging.info("Saved dataset with predictions to data/loan_data_with_predictions.csv")

    except Exception as e:
        logging.error(f"Error generating Basel III report: {e}")
        raise

if __name__ == "__main__":
    generate_basel_report()