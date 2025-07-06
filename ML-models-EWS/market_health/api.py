from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import logging
from logging import StreamHandler
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_risk.log'),
        StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Market Health API", description="API for accessing market risk metrics and reports")

# Load model results
try:
    model_file_path = Path('models/all_commodities_model.json')
    if model_file_path.exists():
        with open(model_file_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results: {list(results.keys())}")
        if results:
            first_symbol = list(results.keys())[0]
            logger.info(f"Structure for first symbol ({first_symbol}): {list(results[first_symbol].keys())}")
    else:
        logger.warning(f"Model results file not found at {model_file_path}. Starting with empty results.")
        results = {}
except Exception as e:
    logger.error(f"Failed to load model results from {model_file_path}: {e}")
    results = {}

# Symbol descriptions
symbol_descriptions = {
    '@CL#C': 'Energy',
    '@C#C': 'Agriculture',
    '@ES': 'Equity Index',
    '@NG': 'Energy',
    '@SI': 'Metals',
    '@TY': 'Interest Rate',
    '@AD': 'Currencies',
    '@EC': 'Currencies',
    '@6E': 'Currencies',
    '@GC': 'Metals',
    '@LE': 'Equity Index',
    '@NQ': 'Equity Index',
    '@SB': 'Equity Index',
    '@YM': 'Equity Index'
}

# Prepare data
expected_symbols = list(symbol_descriptions.keys())
loaded_symbols = list(results.keys())
symbols = [s for s in expected_symbols if s in loaded_symbols and results[s] is not None]

for s in symbols:
    results[s].setdefault('mdi_imp', {})
    results[s].setdefault('monte_carlo_var', {})
    results[s].setdefault('historical_var', {})
    results[s].setdefault('stress_results', {})
    results[s].setdefault('backtest_results', {})
    results[s].setdefault('portfolio_var', None)

feature_importance_data = {s: pd.Series(results[s].get('mdi_imp', {})).sort_values(ascending=True) for s in symbols}

var_data = {}
all_var_keys = [f'{method}_{conf}_{h}d' for method in ['mc_var', 'hist_var'] for conf in [0.95, 0.99] for h in [1, 10]]

for s in symbols:
    mc_var = results[s].get('monte_carlo_var', {})
    hist_var = results[s].get('historical_var', {})
    combined_var_results = {**mc_var, **hist_var}
    data_list_for_symbol = []
    for key in all_var_keys:
        var_info = combined_var_results.get(key, {})
        data_list_for_symbol.append({
            'index': key,
            'var': var_info.get('var', 0),
            'revenue_impact': var_info.get('revenue_impact', 0),
            'capital_impact': var_info.get('capital_impact', 0),
            'liquidity_impact': var_info.get('liquidity_impact', 0)
        })
    var_data[s] = pd.DataFrame(data_list_for_symbol).set_index('index')

stress_data = {}
default_stress_scenarios_keys = ['2008_Crash', '1987_Crash', 'COVID_Drop', 'Rate_Hike', 'Geopolitical',
                                 'Liquidity_Shortage']

for s in symbols:
    stress_results = results[s].get('stress_results', {})
    scenarios_keys = list(stress_results.keys()) if stress_results else default_stress_scenarios_keys
    data_list_for_symbol = []
    for key in scenarios_keys:
        scenario_info = stress_results.get(key, {})
        data_list_for_symbol.append({
            'index': key,
            'final_value': scenario_info.get('final_value', 0),
            'loss': scenario_info.get('loss', 0),
            'drawdown': scenario_info.get('drawdown', 0),
            'margin_shortfall': scenario_info.get('margin_shortfall', 0),
            'revenue_impact': scenario_info.get('revenue_impact', 0),
            'capital_impact': scenario_info.get('capital_impact', 0),
            'liquidity_impact': scenario_info.get('liquidity_impact', 0)
        })
    stress_data[s] = pd.DataFrame(data_list_for_symbol).set_index('index')

backtest_data = {}
all_backtest_metrics_keys = all_var_keys

for s in symbols:
    backtest_results = results[s].get('backtest_results', {})
    metrics_keys = list(backtest_results.keys()) if backtest_results else all_backtest_metrics_keys
    data_list_for_symbol = []
    for key in metrics_keys:
        metric_info = backtest_results.get(key, {})
        data_list_for_symbol.append({
            'index': key,
            'exceedances': metric_info.get('exceedances', 0),
            'coverage': metric_info.get('coverage', None)
        })
    backtest_data[s] = pd.DataFrame(data_list_for_symbol).set_index('index')


def generate_heatmap(symbol: str, results_dict: Dict[str, Any]) -> tuple[go.Figure, pd.DataFrame]:
    """Generate a heatmap for risk exposure."""
    try:
        symbol_results = results_dict.get(symbol, {})
        mc_var_results = symbol_results.get('monte_carlo_var', {})
        stress_results = symbol_results.get('stress_results', {})
        var_values = {k: v.get('var', 0) for k, v in mc_var_results.items()}
        stress_values = {k: v.get('loss', 0) for k, v in stress_results.items()}
        heatmap_dict = {**var_values, **stress_values}
        if not heatmap_dict:
            return go.Figure().update_layout(
                title=f'Risk Exposure Heat Map for {symbol} ({symbol_descriptions.get(symbol, "Unknown Symbol")})<br>(Data Not Available)'), pd.DataFrame()
        data = pd.DataFrame(list(heatmap_dict.items()), columns=['Scenario/Metric', 'Value']).set_index(
            'Scenario/Metric')
        fig = px.imshow(
            data.values,
            labels=dict(x="Metric", y="Scenario/Metric", color="Value"),
            x=['Loss/Value ($)'],
            y=list(data.index),
            title=f'Risk Exposure Heat Map for {symbol} ({symbol_descriptions.get(symbol, "Unknown Symbol")})',
            color_continuous_scale='Reds',
            text_auto=True
        )
        fig.update_layout(
            xaxis={'side': 'top'},
            title={'x': 0.5, 'xanchor': 'center'}
        )
        return fig, data
    except Exception as e:
        logger.error(f"Heatmap generation failed for {symbol}: {e}")
        return go.Figure().update_layout(
            title=f'Risk Exposure Heat Map for {symbol} ({symbol_descriptions.get(symbol, "Unknown Symbol")})<br>(Error Loading Data: {e})'), pd.DataFrame()


# Pydantic models for response validation
class SymbolInfo(BaseModel):
    symbol: str
    description: str


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class VaRMetric(BaseModel):
    method: str
    var: float
    revenue_impact: float
    capital_impact: float
    liquidity_impact: float


class StressTestMetric(BaseModel):
    scenario: str
    final_value: float
    loss: float
    drawdown: float
    margin_shortfall: float
    revenue_impact: float
    capital_impact: float
    liquidity_impact: float


class BacktestMetric(BaseModel):
    metric: str
    exceedances: int
    coverage: Optional[float]


class HeatmapData(BaseModel):
    scenario_metric: str
    value: float


class SymbolMetricsResponse(BaseModel):
    symbol: str
    description: str
    portfolio_var: Optional[float]
    feature_importance: List[FeatureImportance]
    var_metrics: List[VaRMetric]
    stress_metrics: List[StressTestMetric]
    backtest_metrics: List[BacktestMetric]
    text_output: str


class HeatmapResponse(BaseModel):
    symbol: str
    description: str
    heatmap_data: List[HeatmapData]
    plotly_json: Dict[str, Any]


# Endpoints
@app.get("/symbols", response_model=List[SymbolInfo], summary="Get available symbols")
async def get_symbols():
    """Retrieve the list of available symbols with their descriptions."""
    try:
        return [{"symbol": s, "description": symbol_descriptions.get(s, "Unknown Symbol")} for s in symbols]
    except Exception as e:
        logger.error(f"Failed to retrieve symbols: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving symbols: {str(e)}")


@app.get("/metrics/{symbol}", response_model=SymbolMetricsResponse, summary="Get all metrics for a symbol")
async def get_metrics(symbol: str):
    """Retrieve all risk metrics for a specific symbol."""
    try:
        if symbol not in symbols or results.get(symbol) is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or no data available")

        result = results[symbol]

        # Feature Importance
        mdi_imp_data = pd.Series(result.get('mdi_imp', {})).sort_values(ascending=True)
        feature_importance = [{"feature": k, "importance": v} for k, v in mdi_imp_data.items()]

        # Portfolio VaR
        portfolio_var = result.get('portfolio_var', None)

        # VaR Metrics
        var_df = var_data.get(symbol, pd.DataFrame())
        var_metrics = var_df.reset_index().to_dict('records')
        var_metrics = [{
            "method": row['index'],
            "var": row['var'],
            "revenue_impact": row['revenue_impact'],
            "capital_impact": row['capital_impact'],
            "liquidity_impact": row['liquidity_impact']
        } for row in var_metrics]

        # Stress Test Metrics
        stress_df = stress_data.get(symbol, pd.DataFrame())
        stress_metrics = stress_df.reset_index().to_dict('records')
        stress_metrics = [{
            "scenario": row['index'],
            "final_value": row['final_value'],
            "loss": row['loss'],
            "drawdown": row['drawdown'],
            "margin_shortfall": row['margin_shortfall'],
            "revenue_impact": row['revenue_impact'],
            "capital_impact": row['capital_impact'],
            "liquidity_impact": row['liquidity_impact']
        } for row in stress_metrics]

        # Backtest Metrics
        backtest_df = backtest_data.get(symbol, pd.DataFrame())
        backtest_metrics = backtest_df.reset_index().to_dict('records')
        backtest_metrics = [{
            "metric": row['index'],
            "exceedances": row['exceedances'],
            "coverage": row['coverage']
        } for row in backtest_metrics]

        # Text Output
        text_output = []
        text_output.append("=== Feature Importance ===")
        if not mdi_imp_data.empty:
            for feature, importance in mdi_imp_data.items():
                text_output.append(f"{feature}: {importance:.4f}")
        else:
            text_output.append("No feature importance data available.")
        text_output.append("")

        text_output.append("=== Portfolio VaR ===")
        text_output.append(
            f"Portfolio VaR (Batch): ${portfolio_var:.2f}" if portfolio_var is not None else "Portfolio VaR (Batch): N/A")
        text_output.append("")

        text_output.append("=== VaR Metrics ===")
        if var_metrics:
            text_output.append("Method | VaR ($) | Revenue Impact ($) | Capital Impact ($) | Liquidity Impact ($)")
            for row in var_metrics:
                text_output.append(
                    f"{row['method']} | {row['var']:.2f} | {row['revenue_impact']:.2f} | {row['capital_impact']:.2f} | {row['liquidity_impact']:.2f}")
        else:
            text_output.append("No VaR metrics data available.")
        text_output.append("")

        text_output.append("=== Stress Test Metrics ===")
        if stress_metrics:
            text_output.append(
                "Scenario | Final Value ($) | Loss ($) | Drawdown (%) | Margin Shortfall ($) | Revenue Impact ($) | Capital Impact ($) | Liquidity Impact ($)")
            for row in stress_metrics:
                text_output.append(
                    f"{row['scenario']} | {row['final_value']:.2f} | {row['loss']:.2f} | {row['drawdown']:.2f} | {row['margin_shortfall']:.2f} | {row['revenue_impact']:.2f} | {row['capital_impact']:.2f} | {row['liquidity_impact']:.2f}")
        else:
            text_output.append("No Stress Test metrics data available.")
        text_output.append("")

        text_output.append("=== Backtest Metrics ===")
        if backtest_metrics:
            text_output.append("Metric | Exceedances | Coverage (%)")
            for row in backtest_metrics:
                coverage_str = f"{row['coverage']:.2f}" if row['coverage'] is not None else "N/A"
                text_output.append(f"{row['metric']} | {row['exceedances']} | {coverage_str}")
        else:
            text_output.append("No Backtest metrics data available.")
        text_output.append("")

        text_output_str = "\n".join(text_output)

        return {
            "symbol": symbol,
            "description": symbol_descriptions.get(symbol, "Unknown Symbol"),
            "portfolio_var": portfolio_var,
            "feature_importance": feature_importance,
            "var_metrics": var_metrics,
            "stress_metrics": stress_metrics,
            "backtest_metrics": backtest_metrics,
            "text_output": text_output_str
        }
    except Exception as e:
        logger.error(f"Failed to retrieve metrics for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving metrics: {str(e)}")


@app.get("/visualizations/{symbol}", response_model=Dict[str, Any], summary="Get visualizations for a symbol")
async def get_visualizations(symbol: str):
    """Retrieve Plotly JSON for feature importance and VaR distribution visualizations."""
    try:
        if symbol not in symbols or results.get(symbol) is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or no data available")

        result = results[symbol]

        # Feature Importance Chart
        mdi_imp_data = pd.Series(result.get('mdi_imp', {})).sort_values(ascending=True)
        if mdi_imp_data.empty or mdi_imp_data.sum() == 0:
            fig1 = go.Figure().update_layout(title=f'Feature Importance for {symbol}<br>(Data Not Available)')
        else:
            fig1 = px.bar(
                x=mdi_imp_data.values,
                y=mdi_imp_data.index,
                orientation='h',
                title=f'Feature Importance for {symbol}',
                labels={'x': 'Importance', 'y': 'Features'},
                color=mdi_imp_data.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig1.update_layout(height=400, showlegend=False, coloraxis_colorbar_title="Importance")

        # MC VaR Distribution Chart
        mc_var_values_dict = result.get('monte_carlo_var', {})
        mc_var_values = [v.get('var', 0) for v in mc_var_values_dict.values()]
        if not mc_var_values_dict or all(v.get('var', 0) == 0 for v in mc_var_values_dict.values()):
            fig2 = go.Figure().update_layout(title=f'MC VaR Values Distribution for {symbol}<br>(Data Not Available)')
        else:
            fig2 = px.histogram(
                x=mc_var_values,
                nbins=len(mc_var_values) if len(mc_var_values) < 20 else 20,
                title=f'MC VaR Values Distribution for {symbol}',
                labels={'x': 'Calculated VaR ($)', 'y': 'Count'},
                color_discrete_sequence=['#1f77b4']
            )
            fig2.update_layout(height=400, showlegend=False)

        return {
            "feature_importance": fig1.to_dict(),
            "mc_var_distribution": fig2.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to generate visualizations for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating visualizations: {str(e)}")


@app.get("/heatmap/{symbol}", response_model=HeatmapResponse, summary="Get heatmap data and visualization")
async def get_heatmap(symbol: str):
    """Retrieve heatmap data and Plotly JSON for a specific symbol."""
    try:
        if symbol not in symbols or results.get(symbol) is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or no data available")

        fig, heatmap_data = generate_heatmap(symbol, results)
        heatmap_data_list = [{"scenario_metric": idx, "value": row['Value']} for idx, row in heatmap_data.iterrows()]

        return {
            "symbol": symbol,
            "description": symbol_descriptions.get(symbol, "Unknown Symbol"),
            "heatmap_data": heatmap_data_list,
            "plotly_json": fig.to_dict()
        }
    except Exception as e:
        logger.error(f"Failed to generate heatmap for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating heatmap: {str(e)}")


@app.get("/report/{symbol}", response_class=FileResponse, summary="Generate and download report")
async def get_report(symbol: str):
    """Generate and download a PDF report for a specific symbol."""
    try:
        if symbol not in symbols or results.get(symbol) is None:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found or no data available")

        report_dir = 'reports'
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        report_path = Path(report_dir) / f"{symbol}_report.pdf"

        from report import generate_report  # Assuming report.py is in the same directory
        generate_report({symbol: results[symbol]}, {'report': {'output_dir': report_dir, 'format': 'pdf'}})

        if not report_path.exists():
            raise HTTPException(status_code=500, detail="Report generation failed")

        return FileResponse(
            path=str(report_path),
            filename=f"{symbol}_report.pdf",
            media_type='application/pdf'
        )
    except Exception as e:
        logger.error(f"Report generation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


if __name__ == '__main__':
    import uvicorn

    Path('logs').mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)