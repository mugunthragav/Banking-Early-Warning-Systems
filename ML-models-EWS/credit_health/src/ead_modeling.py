import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import pickle
import logging
from src.data_preprocessing import load_data, load_config, preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_ead_model(data_path='data/loan_data_2007_2014_preprocessed.csv'):
    try:
        config = load_config('config.yaml')
        data = load_data(data_path)
        data, defaulted_loans, _, defaulted_features = preprocess_data(data, config, is_training=True)

        features = config['features']['all']
        X = defaulted_features[features]
        y = defaulted_loans['CCF'].clip(0, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 1.0]
        }
        model = GridSearchCV(
            GradientBoostingRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        logging.info(f"Best GBM params: {model.best_params_}")

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        logging.info(f"EAD RMSE: {rmse:.4f}")

        with open('models/ead_model_gbm.pkl', 'wb') as f:
            pickle.dump(model.best_estimator_, f)
        with open('models/ead_feature_importance.pkl', 'wb') as f:
            pickle.dump(dict(zip(features, model.best_estimator_.feature_importances_)), f)
        logging.info("Saved EAD model and feature importance")

        return model, rmse

    except Exception as e:
        logging.error(f"Error training EAD model: {e}")
        raise


if __name__ == "__main__":
    train_ead_model()