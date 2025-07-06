import os
import logging
from pathlib import Path
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Dynamic path resolution
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from routers import credit_router, liquidity_router, market_router

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, 'logs', 'combined_risk_api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Combined Risk API",
    description="Unified API for Credit, Liquidity, and Market Health Risk Analysis",
    version="1.0.0"
)

# Configure CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with specific prefixes
app.include_router(credit_router.router, prefix="/credit")
app.include_router(liquidity_router.router, prefix="/liquidity")
app.include_router(market_router.router, prefix="/market")

# Health check endpoint for the combined API
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "components": {
            "credit": "operational (mocked)",
            "liquidity": "operational",
            "market": "operational"
        }
    }

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Combined Risk API on http://0.0.0.0:8000 with CORS for origins: {ALLOWED_ORIGINS}")
    uvicorn.run(app, host="0.0.0.0", port=8000)