from sqlalchemy.orm import Session
from app.database.db_session import SessionLocal
from app.database.model import ApplicationLog
from datetime import datetime

def test_data_insertion():
    db = SessionLocal()
    try:
        # Create a sample application log entry
        test_entry = ApplicationLog(
            created_at=datetime.now(),
            status_message="Test entry",
            loan_amnt=5000,
            funded_amnt=5000,
            funded_amnt_inv=4975.0,
            term="36 months",
            int_rate=10.65,
            installment=162.87,
            grade="B",
            emp_length=10,
            home_ownership="RENT",
            annual_inc=24000,
            verification_status="Verified",
            dti=27.65,
            delinq_2yrs=0,
            inq_last_6mths=1,
            pd_ml_probability=0.15,
            pd_ml_prediction=0,
            probability_of_repayment=0.85,
            lgd_ml_ann=0.45,
            recovery_rate_ml=0.55,
            ead_ml_meta=4975.0,
            expected_loss_ml=335.81,
            ai_interpretation_text="Test interpretation"
        )

        # Add and commit the entry
        db.add(test_entry)
        db.commit()
        db.refresh(test_entry)
        print(f"Successfully inserted test entry with ID: {test_entry.id}")

        # Verify we can read it back
        inserted_entry = db.query(ApplicationLog).filter(ApplicationLog.id == test_entry.id).first()
        print("\nRetrieved entry details:")
        print(f"ID: {inserted_entry.id}")
        print(f"Loan Amount: {inserted_entry.loan_amnt}")
        print(f"Grade: {inserted_entry.grade}")
        print(f"PD ML Probability: {inserted_entry.pd_ml_probability}")
        print(f"Expected Loss ML: {inserted_entry.expected_loss_ml}")

    except Exception as e:
        print(f"Error during test: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    print("Testing database insertion...")
    test_data_insertion()
