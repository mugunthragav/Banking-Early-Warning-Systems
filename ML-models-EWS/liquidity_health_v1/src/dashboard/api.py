import os
import sys
import pandas as pd
from datetime import timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import numpy as np
from io import StringIO
import csv

# Set BASE_DIR relative to script location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'src'))

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

from src.core.lcr_models import LCRModels
from src.core.nsfr_models import NSFRModels
from src.simulation_engine import SimulationEngine

# Initialize models and simulation engine
lcr_calc = LCRModels()
nsfr_calc = NSFRModels()
sim_engine = SimulationEngine(os.path.join(DATA_DIR, 'transaction_data.csv'))

# Load and preprocess data
df = pd.read_csv(os.path.join(DATA_DIR, 'transaction_data.csv'))
for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y']:
    df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
    if not df['date'].isna().all():
        break
if df['date'].isna().all():
    raise ValueError("Failed to parse 'date' column. Check the date format.")
df.set_index('date', inplace=True)

# Prepare historical LCR and NSFR
lcr_series = df['lcr'].dropna() if 'lcr' in df.columns else pd.Series()
if len(lcr_series) == 0:
    avg_outflow = df['outflows'].mean()
    df['net_outflow'] = df['outflows'].rolling(window=30, min_periods=30).sum() - df['inflows'].rolling(window=30, min_periods=30).sum()
    df['net_outflow'] = np.where(df['net_outflow'] <= 0, avg_outflow * 0.05, df['net_outflow'])
    df['lcr'] = df['hqla_value'] / df['net_outflow'] * 100
    lcr_series = df['lcr'].dropna()

nsfr_series = df['nsfr'].dropna() if 'nsfr' in df.columns else pd.Series()
if len(nsfr_series) == 0:
    df['nsfr'] = np.where(df['required_funding'] != 0,
                         df['stable_funding'].rolling(window=365, min_periods=365).mean() /
                         df['required_funding'].rolling(window=365, min_periods=365).mean() * 100,
                         np.nan)
    nsfr_series = df['nsfr'].dropna()

end_date = df.index.max()
start_date = end_date - timedelta(days=365)
lcr_historical = lcr_series[lcr_series.index >= start_date]
nsfr_historical = nsfr_series[nsfr_series.index >= start_date]

# Train models if not pre-trained
if not os.path.exists(os.path.join(MODEL_DIR, "best_lcr_model.pkl")) or not os.path.exists(os.path.join(MODEL_DIR, "best_nsfr_model.pkl")):
    lcr_calc.get_best_model()
    nsfr_calc.get_best_model()

# Initialize FastAPI app
app = FastAPI(title="Liquidity & Stability API", description="API for analyzing liquidity and stability metrics (LCR, NSFR) with forecasting and stress testing.")

# Pydantic model for stress simulation request
class StressSimulationRequest(BaseModel):
    hqla_adjustment: float = 0.0
    outflow_adjustment: float = 0.0
    inflow_adjustment: float = 0.0
    stable_funding_adjustment: float = 0.0
    required_funding_adjustment: float = 0.0

# LCR Endpoints
@app.get("/lcr/historical", summary="Get historical LCR data")
async def get_lcr_historical():
    data = [{"date": idx.strftime('%Y-%m-%d'), "lcr": float(round(val, 2))} for idx, val in lcr_historical.items()]
    return {"historical_lcr": data}

@app.get("/lcr/forecast", summary="Get LCR forecast")
async def get_lcr_forecast(periods: int = 1):
    if periods <= 0:
        raise HTTPException(status_code=400, detail="Forecast periods must be greater than 0")
    steps = 30 * periods  # 30 days per period, matching dashboard logic
    lcr_forecast = lcr_calc.forecast_lcr(steps=steps).astype(float)  # Convert to float
    data = [{"date": idx.strftime('%Y-%m-%d'), "lcr": float(round(val, 2))} for idx, val in lcr_forecast.items()]
    return {
        "forecast_lcr": data,
        "range": {
            "min": float(round(lcr_forecast.min(), 2)),
            "max": float(round(lcr_forecast.max(), 2)),
            "avg": float(round(lcr_forecast.mean(), 2))
        }
    }

@app.get("/lcr/low-liquidity-accounts", summary="Get accounts with low LCR")
async def get_low_liquidity_accounts():
    low_liquidity_accounts = df[df['lcr'] < 100].groupby('account_id').size().reset_index(name='low_lcr_count') if 'account_id' in df.columns else pd.DataFrame()
    return {"low_liquidity_accounts": low_liquidity_accounts.to_dict('records')}

@app.get("/lcr/low-liquidity-divisions", summary="Get divisions with low LCR")
async def get_low_liquidity_divisions():
    low_liquidity_divisions = df[df['lcr'] < 100].groupby('division').size().reset_index(name='low_lcr_count') if 'division' in df.columns else pd.DataFrame()
    return {"low_liquidity_divisions": low_liquidity_divisions.to_dict('records')}

@app.get("/lcr/alerts", response_class=PlainTextResponse, summary="Export LCR alerts as CSV")
async def export_lcr_alerts():
    low_lcr_data = df[df['lcr'] < 100]
    if not low_lcr_data.empty:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Date', 'Account ID', 'Division', 'LCR (%)', 'Recommendation'])
        for index, row in low_lcr_data.iterrows():
            recommendation = f"Increase HQLA by {max(100 - row['lcr'], 10)}% or reduce outflows by {max(100 - row['lcr'], 10)}% for account {row['account_id']} in division {row['division']}."
            writer.writerow([index.strftime('%Y-%m-%d'), row['account_id'], row['division'], float(round(row['lcr'], 2)), recommendation])
        return output.getvalue()
    return "No alerts to export."

# NSFR Endpoints
@app.get("/nsfr/historical", summary="Get historical NSFR data")
async def get_nsfr_historical():
    data = [{"date": idx.strftime('%Y-%m-%d'), "nsfr": float(round(val, 2))} for idx, val in nsfr_historical.items()]
    return {"historical_nsfr": data}

@app.get("/nsfr/forecast", summary="Get NSFR forecast")
async def get_nsfr_forecast(days: int = 365):
    if days <= 0:
        raise HTTPException(status_code=400, detail="Forecast days must be greater than 0")
    nsfr_forecast = nsfr_calc.forecast_nsfr(steps=days).astype(float)  # Convert to float
    data = [{"date": idx.strftime('%Y-%m-%d'), "nsfr": float(round(val, 2))} for idx, val in nsfr_forecast.items()]
    return {
        "forecast_nsfr": data,
        "range": {
            "min": float(round(nsfr_forecast.min(), 2)),
            "max": float(round(nsfr_forecast.max(), 2)),
            "avg": float(round(nsfr_forecast.mean(), 2))
        }
    }

# Stress Simulation Endpoint
@app.post("/stress-simulation", summary="Run stress simulation with adjustments")
async def run_stress_simulation(request: StressSimulationRequest):
    simulations = sim_engine.run_stress_simulation(
        hqla_adjustment=request.hqla_adjustment,
        outflow_adjustment=request.outflow_adjustment,
        inflow_adjustment=request.inflow_adjustment,
        stable_funding_adjustment=request.stable_funding_adjustment,
        required_funding_adjustment=request.required_funding_adjustment
    )
    results = {}
    for days, series in simulations.items():
        series = series.astype(float)  # Convert to float
        results[f"{days}_days"] = {
            "min": float(round(series.min(), 2)),
            "max": float(round(series.max(), 2)),
            "avg": float(round(series.mean(), 2)),
            "series": [{"date": idx.strftime('%Y-%m-%d'), "value": float(round(val, 2))} for idx, val in series.items()]
        }
    return {"stress_simulation": results}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="localhost", port=8000)