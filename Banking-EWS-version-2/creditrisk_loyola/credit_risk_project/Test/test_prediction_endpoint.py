from fastapi.testclient import TestClient
from app.main import app
import json

client = TestClient(app)

def test_prediction_endpoint():
    # Test data
    test_data = {
        "applications": [
            {
                "loan_amnt": 5000,
                "funded_amnt": 5000,
                "funded_amnt_inv": 4975.0,
                "term": "36 months",
                "int_rate": 10.65,
                "installment": 162.87,
                "grade": "B",
                "emp_length": 10,
                "home_ownership": "RENT",
                "annual_inc": 24000,
                "verification_status": "Verified",
                "dti": 27.65,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "mths_since_last_delinq": None,
                "open_acc": 3,
                "pub_rec": 0,
                "revol_bal": 13648,
                "revol_util": 83.7,
                "total_acc": 9,
                "initial_list_status": "f",
                "out_prncp": 0,
                "total_pymnt": 0,
                "total_rec_prncp": 0,
                "total_rec_int": 0,
                "total_rec_late_fee": 0,
                "recoveries": 0,
                "collection_recovery_fee": 0,
                "last_pymnt_amnt": 0,
                "tot_coll_amt": 0,
                "tot_cur_bal": 13648,
                "total_rev_hi_lim": 16300,
                "mths_since_earliest_cr_line": 60,
                "purpose": "debt_consolidation"
            }
        ]
    }

    print("\nSending test request to /predict/batch_json endpoint...")
    response = client.post("/api/v1/predict/batch_json", json=test_data)
    print(f"\nResponse status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nSuccessful response:")
        print(f"Cumulative Expected Loss: {result.get('cumulative_expected_loss')}")
        print(f"Credit Risk Percentage: {result.get('credit_risk_percentage')}")
        print(f"Number of results: {len(result.get('results', []))}")
        
        for item in result.get('results', []):
            print(f"\nApplication DB ID: {item.get('application_db_id')}")
            print(f"Status Message: {item.get('status_message')}")
            print(f"PD ML Probability: {item.get('pd_ml_probability')}")
            print(f"Expected Loss ML: {item.get('expected_loss_ml')}")
    else:
        print("\nError response:")
        print(response.text)

if __name__ == "__main__":
    print("Testing prediction endpoint with sample data...")
    test_prediction_endpoint()
