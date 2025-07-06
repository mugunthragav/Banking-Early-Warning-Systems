import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# Load the trained models and scaler
scaler = pickle.load(open('models/scaler.pkl', 'rb'))
pd_model = pickle.load(open('models/pd_model_xgboost.pkl', 'rb'))
lgd_model = pickle.load(open('models/lgd_model_dt.pkl', 'rb'))
ead_model = pickle.load(open('models/ead_model_gbm.pkl', 'rb'))
pd_importance = pickle.load(open('models/pd_feature_importance.pkl', 'rb'))
lgd_importance = pickle.load(open('models/lgd_feature_importance.pkl', 'rb'))
ead_importance = pickle.load(open('models/ead_feature_importance.pkl', 'rb'))

# Define the expected features based on config.yaml
EXPECTED_FEATURES = [
    'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
    'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim', 'installment',
    'mths_since_last_delinq', 'mths_since_last_record', 'term_int', 'emp_length_int',
    'mths_since_issue_d', 'mths_since_earliest_cr_line', 'revol_bal', 'revol_util',
    'grade_A', 'grade_B', 'grade_C', 'grade_D', 'grade_E', 'grade_F', 'grade_G',
    'home_ownership_MORTGAGE', 'home_ownership_OWN', 'home_ownership_RENT',
    'purpose_car', 'purpose_credit_card', 'purpose_debt_consolidation',
    'purpose_home_improvement', 'purpose_house', 'purpose_major_purchase', 'purpose_medical',
    'purpose_moving', 'purpose_other', 'purpose_small_business', 'purpose_vacation',
    'purpose_wedding',
    'initial_list_status_f', 'initial_list_status_w'
]

# Streamlit app layout
st.title("Loan Risk Prediction App")
st.write("This app predicts loan risk metrics and provides portfolio-level insights.")

# Tabs for Individual Prediction and Portfolio Insights
tab1, tab2 = st.tabs(["Individual Loan Prediction", "Portfolio Insights"])

# Tab 1: Individual Loan Prediction
with tab1:
    st.header("Individual Loan Prediction")
    st.write("Enter the loan details below to predict the risk metrics.")

    # Input fields
    loan_id = st.number_input("Loan ID", value=9999999)
    funded_amnt = st.number_input("Funded Amount", value=15000.0)
    int_rate = st.number_input("Interest Rate (%)", value=25.0)
    grade = st.selectbox("Grade", ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=0)
    dti = st.number_input("Debt-to-Income Ratio (DTI)", value=35.0)
    home_ownership = st.selectbox("Home Ownership", ['RENT', 'OWN', 'MORTGAGE'], index=0)
    purpose = st.selectbox("Purpose", [
        'car', 'credit_card', 'debt_consolidation', 'home_improvement', 'house',
        'major_purchase', 'medical', 'moving', 'other', 'small_business', 'vacation', 'wedding'
    ], index=9)  # Default to 'small_business'
    initial_list_status = st.selectbox("Initial List Status", ['f', 'w'], index=0)
    term = st.selectbox("Term", [' 36 months', ' 60 months'], index=0)
    annual_inc = st.number_input("Annual Income", value=20000.0)
    emp_length = st.selectbox("Employment Length", [
        '< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
        '6 years', '7 years', '8 years', '9 years', '10+ years'
    ], index=0)
    delinq_2yrs = st.number_input("Delinquencies in Last 2 Years", value=2)
    inq_last_6mths = st.number_input("Inquiries in Last 6 Months", value=4)
    open_acc = st.number_input("Open Accounts", value=5)
    pub_rec = st.number_input("Public Records", value=1)
    total_acc = st.number_input("Total Accounts", value=10)
    acc_now_delinq = st.number_input("Accounts Now Delinquent", value=0)
    total_rev_hi_lim = st.number_input("Total Revolving High Limit", value=30000)
    installment = st.number_input("Installment", value=500.0)
    mths_since_last_delinq = st.number_input("Months Since Last Delinquency", value=12)
    mths_since_last_record = st.number_input("Months Since Last Record", value=24)
    revol_bal = st.number_input("Revolving Balance", value=10000)
    revol_util = st.number_input("Revolving Utilization (%)", value=90.0)
    issue_d = st.date_input("Issue Date", value=datetime(2019, 1, 1))
    earliest_cr_line = st.date_input("Earliest Credit Line", value=datetime(2010, 1, 1))

    # Predict button
    if st.button("Predict"):
        # Create a dictionary with the input data
        input_data = {
            'loan_id': loan_id,
            'funded_amnt': funded_amnt,
            'int_rate': int_rate,
            'grade': grade,
            'dti': dti,
            'home_ownership': home_ownership,
            'purpose': purpose,
            'initial_list_status': initial_list_status,
            'term': term,
            'annual_inc': annual_inc,
            'emp_length': emp_length,
            'delinq_2yrs': delinq_2yrs,
            'inq_last_6mths': inq_last_6mths,
            'open_acc': open_acc,
            'pub_rec': pub_rec,
            'total_acc': total_acc,
            'acc_now_delinq': acc_now_delinq,
            'total_rev_hi_lim': total_rev_hi_lim,
            'installment': installment,
            'mths_since_last_delinq': mths_since_last_delinq,
            'mths_since_last_record': mths_since_last_record,
            'revol_bal': revol_bal,
            'revol_util': revol_util,
            'issue_d': issue_d.strftime('%Y-%m-%d'),
            'earliest_cr_line': earliest_cr_line.strftime('%Y-%m-%d')
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Feature engineering (match src/data_preprocessing.py exactly)
        # 1. Convert term to term_int
        input_df['term_int'] = input_df['term'].str.extract('(\d+)').astype(int)

        # 2. Convert emp_length to emp_length_int
        emp_length_mapping = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9, '10+ years': 10
        }
        input_df['emp_length_int'] = input_df['emp_length'].map(emp_length_mapping)

        # 3. Calculate months since issue_d and earliest_cr_line using the current date
        current_date = pd.to_datetime(datetime.today().date())  # Use today's date dynamically
        input_df['issue_d'] = pd.to_datetime(input_df['issue_d'])
        input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'])
        input_df['mths_since_issue_d'] = ((current_date - input_df['issue_d']) / pd.Timedelta(days=30)).astype(int)
        input_df['mths_since_earliest_cr_line'] = ((current_date - input_df['earliest_cr_line']) / pd.Timedelta(days=30)).astype(int)

        # 4. Handle missing values (match training)
        numerical_cols = [
            'funded_amnt', 'int_rate', 'annual_inc', 'dti', 'delinq_2yrs', 'inq_last_6mths',
            'open_acc', 'pub_rec', 'total_acc', 'acc_now_delinq', 'total_rev_hi_lim', 'installment',
            'mths_since_last_delinq', 'mths_since_last_record', 'revol_bal', 'revol_util',
            'term_int', 'emp_length_int', 'mths_since_issue_d', 'mths_since_earliest_cr_line'
        ]
        median_values = {
            'funded_amnt': 15000, 'int_rate': 12.0, 'annual_inc': 60000, 'dti': 18.0,
            'delinq_2yrs': 0, 'inq_last_6mths': 0, 'open_acc': 10, 'pub_rec': 0,
            'total_acc': 20, 'acc_now_delinq': 0, 'total_rev_hi_lim': 30000, 'installment': 400,
            'mths_since_last_delinq': 30, 'mths_since_last_record': 60, 'term_int': 36,
            'emp_length_int': 5, 'mths_since_issue_d': 60, 'mths_since_earliest_cr_line': 120,
            'revol_bal': 10000, 'revol_util': 50.0
        }
        for col in numerical_cols:
            input_df[col] = input_df[col].fillna(median_values[col])

        # 5. Create dummy variables (match training exactly)
        categorical_cols = ['grade', 'home_ownership', 'purpose', 'initial_list_status']
        input_df = pd.get_dummies(input_df, columns=categorical_cols, dummy_na=False)

        # 6. Ensure all expected dummy variables are present
        for feature in EXPECTED_FEATURES:
            if feature not in input_df.columns and feature.startswith(('grade_', 'home_ownership_', 'purpose_', 'initial_list_status_')):
                input_df[feature] = 0

        # 7. Select only the expected features
        input_df = input_df[EXPECTED_FEATURES]

        # 8. Scale the input data
        input_scaled = scaler.transform(input_df)

        # 9. Make predictions
        pd_pred = pd_model.predict_proba(input_scaled)[:, 1][0]  # Probability of default
        lgd_pred = lgd_model.predict(input_scaled)[0]  # Loss given default
        ead_pred = ead_model.predict(input_scaled)[0]  # Exposure at default (CCF)
        ead_pred_amount = ead_pred * funded_amnt  # Convert CCF to amount

        # 10. Calculate expected loss
        expected_loss = pd_pred * lgd_pred * ead_pred_amount

        # 11. Confidence scores (approximated intervals)
        pd_conf = (max(0, pd_pred - 0.1), min(1, pd_pred + 0.1))
        lgd_conf = (max(0, lgd_pred - 0.05), min(1, lgd_pred + 0.05))
        ead_conf = (ead_pred_amount * 0.9, ead_pred_amount * 1.1)

        # 12. Decision factors (top 3 features)
        pd_top = sorted(pd_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        lgd_top = sorted(lgd_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        ead_top = sorted(ead_importance.items(), key=lambda x: x[1], reverse=True)[:3]

        # Display predictions
        st.write("### Prediction Results")
        will_default = "Yes" if pd_pred > 0.5 else "No"
        st.write(f"**Loan ID**: {loan_id}")
        st.write(f"**Borrower will default**: {will_default} ({pd_pred:.1%} probability, Confidence: {pd_conf[0]:.1%}–{pd_conf[1]:.1%})")
        st.write(f"**Potential loss**: {lgd_pred:.1%} (${funded_amnt * lgd_pred:,.2f}, Confidence: {lgd_conf[0]:.1%}–{lgd_conf[1]:.1%})")
        st.write(f"**Exposure at default**: ${ead_pred_amount:,.2f} (Confidence: ${ead_conf[0]:,.2f}–${ead_conf[1]:,.2f})")
        st.write(f"**Expected loss**: ${expected_loss:,.2f}")

        st.subheader("Decision Factors")
        st.write("**PD Factors**:")
        for feature, importance in pd_top:
            st.write(f"- {feature}: {importance:.4f}")
        st.write("**LGD Factors**:")
        for feature, importance in lgd_top:
            st.write(f"- {feature}: {importance:.4f}")
        st.write("**EAD Factors**:")
        for feature, importance in ead_top:
            st.write(f"- {feature}: {importance:.4f}")

        # 13. Save predictions to CSV
        predictions = pd.DataFrame({
            'loan_id': [loan_id],
            'pd': [pd_pred],
            'lgd': [lgd_pred],
            'ead': [ead_pred_amount],
            'expected_loss': [expected_loss]
        })
        if not os.path.exists('results'):
            os.makedirs('results')
        predictions_file = 'results/loan_risk_predictions.csv'
        if os.path.exists(predictions_file):
            predictions.to_csv(predictions_file, mode='a', header=False, index=False)
        else:
            predictions.to_csv(predictions_file, mode='w', header=True, index=False)
        st.write(f"Predictions saved to {predictions_file}")

# Tab 2: Portfolio Insights
with tab2:
    st.header("Portfolio Insights")
    st.write("View portfolio-level risk metrics and generate the Basel III report.")

    # Display existing Basel III report if available
    basel_report_file = 'basel_report.txt'
    if os.path.exists(basel_report_file):
        with open(basel_report_file, 'r') as f:
            basel_report_content = f.read()
        st.subheader("Latest Basel III Report")
        st.text(basel_report_content)
    else:
        st.info("No Basel III report available. Generate a new report below.")

    # Button to generate Basel III report
    if st.button("Generate Basel III Report"):
        try:
            from generate_basel_report import generate_basel_report
            with st.spinner("Generating Basel III report..."):
                generate_basel_report()
            st.success("Basel III report generated successfully!")
            # Display the newly generated report
            with open(basel_report_file, 'r') as f:
                basel_report_content = f.read()
            st.subheader("Updated Basel III Report")
            st.text(basel_report_content)
        except Exception as e:
            st.error(f"Error generating Basel III report: {str(e)}")

    # Optionally, display portfolio metrics from loan_data_with_predictions.csv
    predictions_file = 'data/loan_data_with_predictions.csv'
    if os.path.exists(predictions_file):
        portfolio_df = pd.read_csv(predictions_file)
        st.subheader("Portfolio Risk Metrics")
        st.write(f"**Total Loans Analyzed**: {len(portfolio_df)}")
        st.write(f"**Average Probability of Default**: {portfolio_df['PD_Probability'].mean():.4f}")
        st.write(f"**Average Loss Given Default**: {portfolio_df['LGD_Prediction'].mean():.4f}")
        st.write(f"**Average Exposure at Default**: ${portfolio_df['EAD_Prediction'].mean():,.2f}")
        st.write(f"**Total Expected Loss**: ${portfolio_df['Expected_Loss'].sum():,.2f}")
        st.write(f"**Total Risk-Weighted Assets (RWA)**: ${portfolio_df['RWA'].sum():,.2f}")
        st.write(f"**Capital Requirement (8% of RWA)**: ${(portfolio_df['RWA'].sum() * 0.08):,.2f}")