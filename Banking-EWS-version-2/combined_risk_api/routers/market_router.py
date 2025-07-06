import os
import json
import logging
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse  # Added import
from pydantic import BaseModel
import numpy as np
import plotly.graph_objects as go

# Logger setup with UTF-8 encoding
logger = logging.getLogger(__name__)
if not logger.handlers:  # Prevent duplicate handlers
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    try:
        handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Dynamic project root
project_root = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(project_root / "models" / "all_commodities_model.json"))

router = APIRouter(
    prefix="/market",
    tags=["Market Risk"]
)

# Pydantic Models
class SymbolInfo(BaseModel):
    symbol: str
    description: str

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class VaRMetric(BaseModel):
    method: str
    var: float
    revenue_impact: float = 0.0
    capital_impact: float = 0.0
    liquidity_impact: float = 0.0

class StressTestMetric(BaseModel):
    scenario: str
    loss: float

class BacktestMetric(BaseModel):
    metric: str
    value: float

class SymbolMetricsResponse(BaseModel):
    symbol: str
    description: str
    portfolio_var: float
    feature_importance: List[FeatureImportance]
    var_metrics: List[VaRMetric]
    stress_metrics: List[StressTestMetric]
    backtest_metrics: List[BacktestMetric]
    text_output: str

class HeatmapData(BaseModel):
    scenario_metric: str
    value: float

class HeatmapResponse(BaseModel):
    symbol: str
    description: str
    heatmap_data: List[HeatmapData]
    plotly_json: dict

class SymbolVisualizationsResponse(BaseModel):
    feature_importance: dict
    mc_var_distribution: dict

# Helper Functions
def load_model_data():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at: {MODEL_PATH}")
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    with open(MODEL_PATH, 'r') as f:
        return json.load(f)

def convert_numpy_to_python(obj):
    """Recursively convert NumPy arrays to Python lists in a dictionary or list."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert NumPy scalars to Python scalars
    return obj

# Endpoints
@router.get("/symbols", response_model=List[SymbolInfo])
async def get_symbols():
    try:
        model_data = load_model_data()
        symbols = [
            SymbolInfo(symbol=symbol, description=data.get("description", "Unknown"))
            for symbol, data in model_data.items()
        ]
        return symbols
    except Exception as e:
        logger.error(f"Error fetching symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch symbols: {str(e)}")

@router.get("/metrics/{symbol}", response_model=SymbolMetricsResponse)
async def get_metrics(symbol: str):
    try:
        model_data = load_model_data()
        if symbol not in model_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        data = model_data[symbol]
        feature_importance = [
            FeatureImportance(feature=k, importance=v)
            for k, v in data.get("mdi_imp", {}).items()
        ]
        var_metrics = [
            VaRMetric(method=k, var=v.get("var", 0), revenue_impact=v.get("revenue_impact", 0))
            for k, v in data.get("monte_carlo_var", {}).items()
        ]
        stress_metrics = [
            StressTestMetric(scenario=k, loss=v.get("loss", 0))
            for k, v in data.get("stress_results", {}).items()
        ]
        backtest_metrics = [
            BacktestMetric(metric=k, value=v)
            for k, v in data.get("backtest_results", {}).items()
        ]
        text_output = (
            f"=== Feature Importance ===\n" +
            "\n".join([f"{fi.feature}: {fi.importance:.4f}" for fi in feature_importance]) +
            f"\n=== Portfolio VaR ===\n{data.get('portfolio_var', 0):.2f}"
        )

        return SymbolMetricsResponse(
            symbol=symbol,
            description=data.get("description", "Unknown"),
            portfolio_var=data.get("portfolio_var", 0),
            feature_importance=feature_importance,
            var_metrics=var_metrics,
            stress_metrics=stress_metrics,
            backtest_metrics=backtest_metrics,
            text_output=text_output
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch metrics: {str(e)}")

@router.get("/visualizations/{symbol}", response_model=SymbolVisualizationsResponse)
async def get_visualizations(symbol: str):
    try:
        model_data = load_model_data()
        if symbol not in model_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        data = model_data[symbol]
        mdi_imp = data.get("mdi_imp", {})
        monte_carlo_var = data.get("monte_carlo_var", {})

        # Feature Importance Bar Plot
        feature_importance_data = [
            go.Bar(
                x=np.array(list(mdi_imp.values())),
                y=list(mdi_imp.keys()),
                orientation='h'
            )
        ]
        feature_importance_layout = {
            "title": {"text": f"Feature Importance for {symbol}"},
            "xaxis": {"title": "Importance"},
            "yaxis": {"title": "Feature"}
        }
        feature_importance_plot = {
            "data": [convert_numpy_to_python(trace.to_plotly_json()) for trace in feature_importance_data],
            "layout": feature_importance_layout
        }

        # Monte Carlo VaR Distribution Histogram
        mc_var_values = np.array([v.get("var", 0) for v in monte_carlo_var.values()])
        mc_var_distribution_data = [
            go.Histogram(
                x=mc_var_values,
                nbinsx=30
            )
        ]
        mc_var_distribution_layout = {
            "title": {"text": f"MC VaR Values Distribution for {symbol}"},
            "xaxis": {"title": "VaR"},
            "yaxis": {"title": "Frequency"}
        }
        mc_var_distribution_plot = {
            "data": [convert_numpy_to_python(trace.to_plotly_json()) for trace in mc_var_distribution_data],
            "layout": mc_var_distribution_layout
        }

        return SymbolVisualizationsResponse(
            feature_importance=feature_importance_plot,
            mc_var_distribution=mc_var_distribution_plot
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching visualizations for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch visualizations: {str(e)}")

@router.get("/heatmap/{symbol}", response_model=HeatmapResponse)
async def get_heatmap(symbol: str):
    try:
        model_data = load_model_data()
        if symbol not in model_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        data = model_data[symbol]
        heatmap_data = []
        z_values = []

        for metric, values in data.get("monte_carlo_var", {}).items():
            heatmap_data.append(HeatmapData(scenario_metric=metric, value=values.get("var", 0)))
            z_values.append(values.get("var", 0))
        for scenario, values in data.get("stress_results", {}).items():
            heatmap_data.append(HeatmapData(scenario_metric=scenario, value=values.get("loss", 0)))
            z_values.append(values.get("loss", 0))

        plotly_json = {
            "data": [
                {
                    "type": "heatmap",
                    "z": [z_values],
                    "x": [m.scenario_metric for m in heatmap_data],
                    "y": [symbol]
                }
            ],
            "layout": {
                "title": {"text": f"Risk Exposure Heat Map for {symbol}"},
                "xaxis": {"title": "Scenario/Metric"},
                "yaxis": {"title": "Symbol"}
            }
        }

        return HeatmapResponse(
            symbol=symbol,
            description=data.get("description", "Unknown"),
            heatmap_data=heatmap_data,
            plotly_json=plotly_json
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching heatmap for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch heatmap: {str(e)}")

@router.get("/report/{symbol}")
async def get_report(symbol: str):
    try:
        from app.report import generate_report
        model_data = load_model_data()
        if symbol not in model_data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        config = {"report": {"output_dir": str(project_root / "reports")}}
        report_path = generate_report({symbol: model_data[symbol]}, config)
        return FileResponse(
            report_path,
            media_type="application/pdf",
            filename=f"{symbol}_report.pdf"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate report: {str(e)}")