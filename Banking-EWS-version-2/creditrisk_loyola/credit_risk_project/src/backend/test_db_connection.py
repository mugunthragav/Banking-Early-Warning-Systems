# backend/test_db_connection.py
import sys
import os

# Add the parent directory to the Python path to allow relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database.db_session import engine, Base, SessionLocal
from app.database.model import ApplicationLog  # Import your models
from sqlalchemy import text # For simple query

def test_db_connection():
    print(f"Database URL: {engine.url}")
    try:
        # Attempt to connect and execute a simple query
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))
            print("Database connection successful!")
            print(f"Query result: {result.scalar()}")

        # Attempt to create tables (if not exists)
        print("Attempting to create tables...")
        Base.metadata.create_all(bind=engine)
        print("Tables checked/created successfully.")

        # Attempt to get a session
        with SessionLocal() as session:
            print("Database session obtained successfully.")
            # You can add a simple query here, e.g., to check if any data exists
            # count = session.query(ApplicationLog).count()
            # print(f"Number of application logs: {count}")

    except Exception as e:
        print(f"Database connection failed: {e}")
        print("Please check your database server status and connection details in .env/config.py.")

if __name__ == "__main__":
    test_db_connection()