import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import logging
import importlib.metadata
from src.data_preprocessing import load_data, load_config, preprocess_data

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pd_model(data_path='data/loan_data_2007_2014_preprocessed.csv'):
    try:
        logging.info(f"XGBoost version: {importlib.metadata.version('xgboost')}")
        config = load_config('config.yaml')
        data = load_data(data_path)

        data, _, features, _ = preprocess_data(data, config, is_training=True)
        features_list = config['features']['all']
        X = data[features_list]
        y = data['target_pd']  # Use the precomputed target_pd

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Impute NaNs
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)
        logging.info("Imputed NaNs in training and test sets")

        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        logging.info(f"SMOTE balanced training set: {X_train_smote.shape}")

        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'n_estimators': [100, 200, 300]
        }
        model = GridSearchCV(
            XGBClassifier(random_state=42, tree_method='hist', device='cpu'),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        model.fit(X_train_smote, y_train_smote)
        logging.info(f"Best XGBoost params: {model.best_params_}")

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_pred_proba)
        logging.info(f"PD AUC-ROC: {auc_roc:.4f}")

        with open('models/pd_model_xgboost.pkl', 'wb') as f:
            pickle.dump(model.best_estimator_, f)
        with open('models/pd_feature_importance.pkl', 'wb') as f:
            pickle.dump(dict(zip(features_list, model.best_estimator_.feature_importances_)), f)
        logging.info("Saved PD model and feature importance")

        return model, auc_roc

    except Exception as e:
        logging.error(f"Error training PD model: {e}")
        raise

if __name__ == "__main__":
    train_pd_model()