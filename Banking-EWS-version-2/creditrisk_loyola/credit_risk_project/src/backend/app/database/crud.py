# backend/app/database/crud.py

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, Any, Optional, List
import traceback
from . import model
from ..core import config

def create_application_log_entry(db: Session, application_data: Dict[str, Any]) -> model.ApplicationLog:
    try:
        print(f"[Database] Creating ApplicationLog entry with data:")
        print(f"[Database] Loan amount: {application_data.get('loan_amnt')}")
        print(f"[Database] Status message: {application_data.get('status_message')}")
        
        db_log_entry = model.ApplicationLog(**application_data)
        db.add(db_log_entry)
        db.commit()
        db.refresh(db_log_entry)
        print(f"[Database] Successfully created entry with ID: {db_log_entry.id}")
        return db_log_entry
    except Exception as e:
        print(f"[Database] Error creating application log entry: {str(e)}")
        print(f"[Database] Error type: {type(e)}")
        db.rollback()
        raise

def get_application_log_by_id(db: Session, application_id: int) -> Optional[model.ApplicationLog]:
    """
    Retrieves a specific application log by its database ID.
    """
    try:
        return db.query(model.ApplicationLog).filter(model.ApplicationLog.id == application_id).first()
    except Exception as e:
        print(f"Error retrieving application log by ID {application_id}: {e}")
        return None

def get_application_logs(db: Session, skip: int = 0, limit: int = 100) -> List[model.ApplicationLog]:
    """
    Retrieves a list of application logs with pagination.
    """
    try:
        return db.query(model.ApplicationLog).offset(skip).limit(limit).all()
    except Exception as e:
        print(f"Error retrieving application logs: {e}")
        return []
