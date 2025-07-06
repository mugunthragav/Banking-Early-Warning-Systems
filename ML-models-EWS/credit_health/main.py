import os
import pickle
import logging
from sklearn.preprocessing import StandardScaler
from src.data_preprocessing import load_data, load_config, preprocess_data
from src.pd_modeling import train_pd_model
from src.lgd_modeling import train_lgd_model
from src.ead_modeling import train_ead_model

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

def main():
    # Step 1: Load configuration and raw data
    logging.info("Loading configuration and data...")
    config = load_config('config.yaml')
    raw_data_path = 'data/loan_data_2007_2014_preprocessed.csv'
    data = load_data(raw_data_path)

    # Step 2: Preprocess the data
    logging.info("Preprocessing data...")
    data, target_pd, features, target_lgd_ead = preprocess_data(data, config, is_training=True)

    # Save preprocessed data
    preprocessed_data_path = 'data/loan_data_2007_2014_final.csv'
    data.to_csv(preprocessed_data_path, index=False)
    logging.info(f"Saved preprocessed data to '{preprocessed_data_path}'")

    # Step 3: Create and save the scaler
    logging.info("Creating and saving scaler...")
    scaler = StandardScaler()
    feature_columns = config['features']['all']
    X = data[feature_columns]
    scaler.fit(X)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logging.info("Saved scaler to 'models/scaler.pkl'")

    # Step 4: Train the PD model
    logging.info("Training PD model...")
    pd_model, auc_roc = train_pd_model(data_path=preprocessed_data_path)
    logging.info(f"PD Model AUC-ROC: {auc_roc:.4f}")

    # Step 5: Train the LGD model
    logging.info("Training LGD model...")
    lgd_model, r2, mse = train_lgd_model(data_path=preprocessed_data_path)
    logging.info(f"LGD Model R²: {r2:.4f}, MSE: {mse:.4f}")

    # Step 6: Train the EAD model
    logging.info("Training EAD model...")
    ead_model, rmse = train_ead_model(data_path=preprocessed_data_path)
    logging.info(f"EAD Model RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()
