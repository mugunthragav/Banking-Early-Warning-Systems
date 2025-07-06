import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc, mean_squared_error, r2_score
from sklearn.metrics import classification_report
import pickle


def evaluate_classification_model(y_true, y_pred, y_proba, model_name):
    """Evaluate classification model and plot ROC curve."""
    print(f"\n{model_name} - Classification Report")
    print(classification_report(y_true, y_pred))

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')  # Save in project directory
    plt.close()

    return roc_auc


def evaluate_regression_model(y_true, y_pred, model_name):
    """Evaluate regression model and plot residuals."""
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{model_name} - Regression Metrics")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    residuals = y_true - y_pred
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.xlabel('Residuals')
    plt.title(f'Residuals Distribution - {model_name}')
    plt.savefig(f'{model_name}_residuals.png')  # Save in project directory
    plt.close()

    return mse, r2


def save_model(model, output_dir, filename):
    """Save model to file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved model: {filename}")


def save_results(result, output_dir):
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    result.to_csv(os.path.join(output_dir, 'credit_health_results.csv'), index=True)
    print(f"Saved results to {os.path.join(output_dir, 'credit_health_results.csv')}")


def calculate_expected_loss(data, pd_probs, lgd_dt, lgd_svr, ead_model, features):
    """Calculate Expected Loss."""
    try:
        # Ensure pd_probs has the correct columns
        pd_models = ['logistic_regression', 'xgboost', 'random_forest']
        if not all(col in pd_probs.columns for col in pd_models):
            raise ValueError(f"pd_probs missing columns: {pd_models}")

        # Scale features for LGD and EAD
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Predict LGD
        lgd_dt_pred = lgd_dt.predict(X_scaled)
        lgd_svr_pred = lgd_svr.predict(X_scaled)
        lgd_pred = (lgd_dt_pred + lgd_svr_pred) / 2  # Average predictions

        # Predict EAD
        ead_pred = ead_model.predict(X_scaled)

        # Calculate Expected Loss for each PD model
        result = pd.DataFrame(index=data.index)
        for model in pd_models:
            result[f'EL_{model}'] = pd_probs[model] * lgd_pred * ead_pred

        # Summary statistics
        summary = result.describe()

        return result, summary

    except Exception as e:
        print(f"Error calculating Expected Loss: {e}")
        raise