import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import pickle
import os
import logging
from src.data_preprocessing import load_data, load_config, preprocess_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create plots directory if it doesn't exist
if not os.path.exists('plots'):
    os.makedirs('plots')

def plot_eda(data):
    """Generate EDA plots for key features."""
    logging.info("Plotting distribution of funded_amnt...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['funded_amnt'], kde=True, bins=50)
    plt.title('Distribution of Funded Amount')
    plt.xlabel('Funded Amount ($)')
    plt.ylabel('Frequency')
    plt.savefig('plots/funded_amnt_distribution.png')
    plt.close()

    logging.info("Plotting distribution of int_rate...")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['int_rate'], kde=True, bins=50)
    plt.title('Distribution of Interest Rate')
    plt.xlabel('Interest Rate (%)')
    plt.ylabel('Frequency')
    plt.savefig('plots/int_rate_distribution.png')
    plt.close()

def plot_relationships(data):
    """Generate plots showing relationships between features and target."""
    logging.info("Plotting int_rate vs. loan_status...")
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='loan_status', y='int_rate', data=data)
    plt.title('Interest Rate vs. Loan Status')
    plt.xlabel('Loan Status')
    plt.ylabel('Interest Rate (%)')
    plt.xticks(rotation=45)
    plt.savefig('plots/int_rate_vs_loan_status.png')
    plt.close()

def plot_model_evaluation():
    """Generate model evaluation plots (ROC curve, feature importance)."""
    # Load models and preprocessed data
    config = load_config('config.yaml')
    data = load_data('data/loan_data_2007_2014_preprocessed.csv')
    data, defaulted_loans, features, _ = preprocess_data(data, config, is_training=True)

    # Extract target_pd from data
    target_pd = data['target_pd']

    # Verify target_pd
    logging.info(f"target_pd dtype: {target_pd.dtype}")
    logging.info(f"target_pd sample: {target_pd.head().to_list()}")
    if not pd.api.types.is_numeric_dtype(target_pd):
        raise ValueError("target_pd must be numeric (0s and 1s), but found non-numeric values.")

    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('models/pd_model_xgboost.pkl', 'rb') as f:
        pd_model = pickle.load(f)
    with open('models/pd_feature_importance.pkl', 'rb') as f:
        pd_importance = pickle.load(f)
    with open('models/lgd_feature_importance.pkl', 'rb') as f:
        lgd_importance = pickle.load(f)
    with open('models/ead_feature_importance.pkl', 'rb') as f:
        ead_importance = pickle.load(f)

    # Scale features
    X = features[config['features']['all']]
    X_scaled = scaler.transform(X)

    # ROC Curve for PD Model
    logging.info("Generating ROC curve for PD model...")
    y_pred_proba = pd_model.predict_proba(X_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(target_pd, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve for PD Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig('plots/roc_curve_pd.png')
    plt.close()

    # Feature Importance Plots
    # PD Feature Importance
    logging.info("Plotting PD feature importance...")
    plt.figure(figsize=(10, 6))
    pd_importance_df = pd.DataFrame(list(pd_importance.items()), columns=['Feature', 'Importance'])
    pd_importance_df = pd_importance_df.sort_values(by='Importance', ascending=False).head(10)
    sns.barplot(x='Importance', y='Feature', data=pd_importance_df)
    plt.title('Top 10 Features for PD Model')
    plt.savefig('plots/pd_feature_importance.png')
    plt.close()

    # LGD Feature Importance
    logging.info("Plotting LGD feature importance...")
    plt.figure(figsize=(10, 6))
    lgd_importance_df = pd.DataFrame(list(lgd_importance.items()), columns=['Feature', 'Importance'])
    lgd_importance_df = lgd_importance_df.sort_values(by='Importance', ascending=False).head(10)
    sns.barplot(x='Importance', y='Feature', data=lgd_importance_df)
    plt.title('Top 10 Features for LGD Model')
    plt.savefig('plots/lgd_feature_importance.png')
    plt.close()

    # EAD Feature Importance
    logging.info("Plotting EAD feature importance...")
    plt.figure(figsize=(10, 6))
    ead_importance_df = pd.DataFrame(list(ead_importance.items()), columns=['Feature', 'Importance'])
    ead_importance_df = ead_importance_df.sort_values(by='Importance', ascending=False).head(10)
    sns.barplot(x='Importance', y='Feature', data=ead_importance_df)
    plt.title('Top 10 Features for EAD Model')
    plt.savefig('plots/ead_feature_importance.png')
    plt.close()

def plot_metrics():
    """Generate plots for portfolio-level metrics from Basel III report."""
    logging.info("Loading data for portfolio metrics plot...")
    data = pd.read_csv('data/loan_data_with_predictions.csv')

    # Portfolio Metrics
    metrics = {
        'Average PD': data['PD_Probability'].mean(),
        'Average LGD': data['LGD_Prediction'].mean(),
        'Average EAD': data['EAD_Prediction'].mean(),
        'Total Expected Loss': data['Expected_Loss'].sum(),
        'Total RWA': data['RWA'].sum()
    }

    logging.info("Plotting portfolio metrics...")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.values()), y=list(metrics.keys()))
    plt.title('Portfolio-Level Metrics')
    plt.xlabel('Value')
    plt.savefig('plots/portfolio_metrics.png')
    plt.close()

def main():
    """Generate all plots."""
    print("Generating EDA plots...")
    data = load_data('data/loan_data_2007_2014_preprocessed.csv')
    plot_eda(data)

    print("Generating relationship plots...")
    plot_relationships(data)

    print("Generating model evaluation plots...")
    plot_model_evaluation()

    print("Generating metrics plots...")
    plot_metrics()

    print("All plots saved to the 'plots/' directory.")

if __name__ == "__main__":
    main()