# backend/app/database/db_session.py

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator

from ..core import config # To get SQLALCHEMY_DATABASE_URL

# Create the SQLAlchemy engine
# The connect_args is specific to SQLite to allow a single connection for a thread.
# For other databases like PostgreSQL or MySQL, you typically don't need connect_args.
engine_args = {}
if config.SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

try:
    engine = create_engine(
        config.SQLALCHEMY_DATABASE_URL,
        **engine_args,
        echo=True,  # Set to True to log all SQL queries; useful for debugging
        # For production, you might want to add pool_pre_ping=True and other pool settings
        # pool_pre_ping=True,
        # pool_recycle=3600, # For example, recycle connections every hour
    )
except Exception as e:
    print(f"Error creating database engine: {e}")
    print(f"Database URL used: {config.SQLALCHEMY_DATABASE_URL}")
    # Depending on your application's needs, you might want to exit or raise the error.
    # For now, we'll let it proceed, but it will likely fail later if the engine isn't created.
    engine = None


# Create a SessionLocal class
# This will be used to create individual database sessions.
try:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception as e:
    print(f"Error creating SessionLocal: {e}")
    # Handle error if engine is None
    if engine is None:
        print("Cannot create SessionLocal because the database engine failed to initialize.")
    SessionLocal = None # type: ignore


# Create a Base class for declarative models
# Your database models will inherit from this class.
Base = declarative_base()


# Dependency function to get a DB session
def get_db() -> Generator:
    """
    FastAPI dependency that provides a SQLAlchemy database session.
    It ensures the session is closed after the request is finished.
    """
    if SessionLocal is None:
        print("ERROR: Database session (SessionLocal) is not configured. Check DB connection.")
        # This will likely cause a 500 error in routes trying to use it.
        # Consider raising an exception or handling this more gracefully based on app requirements.
        yield None # Or raise HTTPException(status_code=503, detail="Database not configured")
        return

    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        # Handle exceptions that might occur when creating a session or during its use.
        print(f"Error in get_db session management: {e}")
        # Depending on the error, you might want to rollback if a transaction was started.
        # However, SessionLocal is configured with autocommit=False, autoflush=False,
        # so explicit commit/rollback is usually handled in CRUD operations.
        if db:
            try:
                db.rollback() # Attempt to rollback if an error occurs within the session usage
            except Exception as rb_exc:
                print(f"Error during rollback in get_db: {rb_exc}")
        raise # Re-raise the exception to be handled by FastAPI's error handling
    finally:
        if db:
            try:
                db.close()
            except Exception as close_exc:
                print(f"Error closing database session in get_db: {close_exc}")

# You can also add a function here to create all tables based on your models.
# This is often called once when the application starts up (e.g., in main.py).
def create_db_tables():
    """Creates all database tables defined by models inheriting from Base."""
    if engine is None:
        print("ERROR: Database engine is not initialized. Cannot create tables.")
        return
    try:
        print("Attempting to create database tables...")
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created successfully.")
    except Exception as e:
        print(f"Error creating database tables: {e}")
        print("Please ensure the database server is running and accessible, and the database URL is correct.")

