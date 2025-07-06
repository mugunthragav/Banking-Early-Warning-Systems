# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1 import predict_router  # Your API router
from app.core import config  # Your application config
from app.database.db_session import engine, Base  # For creating tables on startup (dev only)

# Optional: Create database tables if they don't exist (for development)
# In production, you would typically use Alembic migrations.
def create_startup_db_tables():
    try:
        print("Attempting to create database tables defined in models...")
        Base.metadata.create_all(bind=engine)
        print("Database tables checked/created.")
    except Exception as e:
        print(f"Error creating database tables on startup: {e}")
        print("Ensure database is accessible and correctly configured.")

# Call it on startup if desired (e.g., for local development)
# create_startup_db_tables() # Uncomment to run on startup

app = FastAPI(
    title=config.APP_NAME,
    version="1.0.0",
    description="Credit Risk Prediction Suite API with ML, Lookup Scoring, and AI Interpretation"
)

origins=[
    "http://localhost:4200",  # Angular dev server
    "http://127.0.0.1:4200"
]

# --- CORS Middleware ---
# Allows your Angular frontend (running on a different port) to communicate with the backend.
# if config.CORS_ALLOWED_ORIGINS:
#     app.add_middleware(
#         CORSMiddleware,
#         allow_origins=[str(origin) for origin in config.CORS_ALLOWED_ORIGINS],  # Origins should be strings
#         allow_credentials=True,
#         allow_methods=["*"],   # Allows all methods (GET, POST, etc.)
#         allow_headers=["*"],   # Allows all headers
#     )
# else:
#     # Fallback if no origins are specified, allow all for local dev convenience (not for production)
#     print("Warning: CORS_ALLOWED_ORIGINS not set in config. Allowing all origins for development.")
app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# --- Include API Routers ---
app.include_router(predict_router.router, prefix=config.API_V1_STR)

# --- Root Endpoint (Optional) ---
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": f"Welcome to the {config.APP_NAME}!"}

# --- Lifespan Events (Optional: for resource initialization/cleanup) ---
# @app.on_event("startup")
# async def startup_event():
#     print("Application startup: Initializing resources...")
#     # create_startup_db_tables() # Alternative place to create tables

# @app.on_event("shutdown")
# async def shutdown_event():
#     print("Application shutdown: Cleaning up resources...")

# To run (from backend/ directory, assuming main.py is in backend/app/):
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
