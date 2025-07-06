import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import logging
from logging import StreamHandler # Import StreamHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_risk.log'),
        StreamHandler() # StreamHandler is now imported
    ]
)
logger = logging.getLogger(__name__)

# Initialize the Dash app
app = dash.Dash(__name__)

# Load the model results
try:
    model_file_path = Path('models/all_commodities_model.json')
    if model_file_path.exists():
        with open(model_file_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded results: {list(results.keys())}")
        if results:
             first_symbol = list(results.keys())[0]
             logger.info(f"Structure for first symbol ({first_symbol}): {list(results[first_symbol].keys())}")
             if 'monte_carlo_var' in results[first_symbol]:
                  logger.info(f"Sample MC VaR for {first_symbol}: {results[first_symbol]['monte_carlo_var'].get('mc_var_0.95_1d')}")
             if 'portfolio_var' in results[first_symbol]:
                  logger.info(f"Sample Portfolio VaR for {first_symbol}: {results[first_symbol]['portfolio_var']}")
             if 'backtest_results' in results[first_symbol]:
                  logger.info(f"Sample Backtest for {first_symbol}: {results[first_symbol]['backtest_results'].get('mc_var_0.95_1d')}")

    else:
        logger.warning(f"Model results file not found at {model_file_path}. Starting with empty results.")
        results = {}

except Exception as e:
    logger.error(f"Failed to load model results from {model_file_path}: {e}")
    results = {}

# Define symbol descriptions (keeping this hardcoded for simplicity as in original)
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

# Prepare data with fallback for missing keys before layout is defined
# This is done once when the app starts
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
default_stress_scenarios_keys = ['2008_Crash', '1987_Crash', 'COVID_Drop', 'Rate_Hike', 'Geopolitical', 'Liquidity_Shortage']

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


def generate_heatmap(symbol, results_dict):
    """Generate a heatmap for risk exposure."""
    try:
        symbol_results = results_dict.get(symbol, {})
        mc_var_results = symbol_results.get('monte_carlo_var', {})
        stress_results = symbol_results.get('stress_results', {})

        var_values = {k: v.get('var', 0) for k, v in mc_var_results.items()}
        stress_values = {k: v.get('loss', 0) for k, v in stress_results.items()}

        heatmap_dict = {**var_values, **stress_values}

        if not heatmap_dict:
             return go.Figure().update_layout(title=f'Risk Exposure Heat Map for {symbol} ({symbol_descriptions.get(symbol, "Unknown Symbol")})<br>(Data Not Available)'), pd.DataFrame()

        data = pd.DataFrame(list(heatmap_dict.items()), columns=['Scenario/Metric', 'Value']).set_index('Scenario/Metric')

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
            title={'x':0.5, 'xanchor': 'center'}
        )
        return fig, data
    except Exception as e:
        logger.error(f"Heatmap generation failed for {symbol}: {e}")
        return go.Figure().update_layout(title=f'Risk Exposure Heat Map for {symbol} ({symbol_descriptions.get(symbol, "Unknown Symbol")})<br>(Error Loading Data: {e})'), pd.DataFrame()


# Define common formatting rules outside the layout for readability
currency_format = dash_table.Format.Format(
    scheme=dash_table.Format.Scheme.fixed, precision=2).symbol('$')

# Corrected percentage format using symbol_suffix
percentage_format = dash_table.Format.Format(
    scheme=dash_table.Format.Scheme.fixed, precision=2).symbol_suffix('%')

integer_format = dash_table.Format.Format(
    scheme=dash_table.Format.Scheme.fixed, precision=0)


# App layout
app.layout = html.Div([
    html.H1("Market Risk Dashboard"),
    dcc.Tabs([
        dcc.Tab(label='Dashboard', children=[
            html.Div([
                dcc.Dropdown(
                    id='symbol-dropdown',
                    options=[{'label': f"{s} ({symbol_descriptions.get(s, 'Unknown Symbol')})", 'value': s} for s in symbols],
                    value=symbols[0] if symbols else None,
                    style={'width': '50%', 'min-width': '200px'}
                ),
            ], style={'margin': '10px 0'}),

            html.Div(id='text-output', style={'margin': '20px'}),

            html.Div(id='portfolio-var-output', style={'margin': '10px 20px', 'fontSize': '1.1em', 'fontWeight': 'bold'}),


            html.H3("Feature Importance", style={'margin': '20px 0 10px 20px'}),
            dcc.Graph(id='feature-importance-chart'),

            html.H3("MC VaR Values Distribution", style={'margin': '20px 0 10px 20px'}),
            dcc.Graph(id='mc-var-distribution-chart'),

            html.H3("Risk Exposure Heat Map", style={'margin': '20px 0 10px 20px'}),
            dcc.Graph(id='heatmap-chart'),

            html.H3("Value at Risk (VaR) Table", style={'margin': '20px 0 10px 20px'}),
            dash_table.DataTable(
                id='var-table',
                style_table={'overflowX': 'auto', 'margin': '0 20px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_data_conditional=[
                    # Currency columns in VaR table
                    {'if': {'column_id': col, 'column_type': 'numeric'}, 'format': currency_format} # Corrected 'type' to 'column_type'
                    for col in ['var', 'revenue_impact', 'capital_impact', 'liquidity_impact']
                ]
            ),

            html.H3("Stress Test Table", style={'margin': '20px 0 10px 20px'}),
            dash_table.DataTable(
                id='stress-table',
                style_table={'overflowX': 'auto', 'margin': '0 20px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                style_data_conditional=[
                    # Currency columns in Stress table
                    {'if': {'column_id': col, 'column_type': 'numeric'}, 'format': currency_format} # Corrected 'type' to 'column_type'
                    for col in ['final_value', 'loss', 'margin_shortfall', 'revenue_impact', 'capital_impact', 'liquidity_impact']
                ] + [
                    # Percentage columns in Stress table
                    {'if': {'column_id': 'drawdown'}, 'format': percentage_format}
                ]
            ),

            html.H3("Backtest Table", style={'margin': '20px 0 10px 20px'}),
            dash_table.DataTable(
                id='backtest-table',
                style_table={'overflowX': 'auto', 'margin': '0 20px'},
                style_header={'backgroundColor': 'lightgrey', 'fontWeight': 'bold'},
                style_cell={'textAlign': 'center', 'padding': '5px'},
                 style_data_conditional=[
                    # Currency columns in Backtest table
                    {'if': {'column_id': col, 'column_type': 'numeric'}, 'format': currency_format} # Corrected 'type' to 'column_type'
                    for col in ['var', 'revenue_impact', 'capital_impact', 'liquidity_impact'] # These columns might appear if VaR details are shown per metric
                ] + [
                    # Integer column in Backtest table
                    {'if': {'column_id': 'exceedances'}, 'format': integer_format},
                    # Percentage column in Backtest table
                    {'if': {'column_id': 'coverage'}, 'format': percentage_format}
                ]
            ),

            html.Div([
                html.Button("Download Report", id="btn-download"),
                dcc.Download(id="download-report"),
            ], style={'margin': '20px'}),

            html.H3("Text Outputs (Copyable)", style={'margin': '20px 0 10px 20px'}),
            html.Pre(id='text-data-output', style={'backgroundColor': '#f0f0f0', 'padding': '10px', 'whiteSpace': 'pre-wrap', 'margin': '0 20px'})
        ]),
        dcc.Tab(label='Instructions', children=[
            html.Div([
                html.H3("Dashboard Instructions", style={'margin-top': '20px'}),
                html.P([
                    "This dashboard visualizes market risk metrics for various assets. Key features:",
                    html.Ul([
                        html.Li("VaR: Monte Carlo and Historical VaR at 95% and 99% confidence levels, with financial impacts."),
                        html.Li("Portfolio VaR: Value at Risk calculation for the portfolio (currently, the selected batch of symbols with equal weights)."),
                        html.Li("Stress Testing: Scenario analysis (e.g., 2008 Crash, COVID Drop) with financial impacts."),
                        html.Li("Feature Importance: Factors influencing the ML model (e.g., Volatility, Momentum)."),
                        html.Li("Heat Map: Visualizes risk exposures across scenarios and key VaR metrics."),
                        html.Li("Backtesting: Compares current VaR predictions with historical exceedances over rolling periods."),
                        html.Li("Text Outputs: Copyable text format of all data at the bottom of the Dashboard tab.")
                    ])
                ], style={'margin-left': '20px'})
            ], style={'margin': '0 20px'})
        ])
    ])
])

@app.callback(
    [
        Output('text-output', 'children'),
        Output('var-table', 'data'),
        Output('var-table', 'columns'),
        Output('stress-table', 'data'),
        Output('stress-table', 'columns'),
        Output('backtest-table', 'data'),
        Output('backtest-table', 'columns'),
        Output('download-report', 'data'),
        Output('text-data-output', 'children'),
        Output('portfolio-var-output', 'children'),
        Output('feature-importance-chart', 'figure'),
        Output('mc-var-distribution-chart', 'figure'),
        Output('heatmap-chart', 'figure')
    ],
    [
        Input('symbol-dropdown', 'value'),
        Input('btn-download', 'n_clicks')
    ],
    prevent_initial_call=True
)
def update_output(selected_symbol, n_clicks):
    try:
        if not selected_symbol or selected_symbol not in results or results[selected_symbol] is None:
            empty_figure = go.Figure().update_layout(title='No Data Available')
            empty_table_data = []
            empty_table_columns = []
            return [], empty_table_data, empty_table_columns, \
                   empty_table_data, empty_table_columns, empty_table_data, empty_table_columns, \
                   None, "No data available for the selected symbol.", "Portfolio VaR (Batch): N/A", \
                   empty_figure, empty_figure, empty_figure


        result = results[selected_symbol]

        text = [
            html.H3(f"Results for {selected_symbol} ({symbol_descriptions.get(selected_symbol, 'Unknown Symbol')})"),
            html.P(f"Risk analysis metrics including VaR, Stress Tests, and Backtesting."),
        ]

        portfolio_var_value = result.get('portfolio_var', None)
        if portfolio_var_value is not None:
             portfolio_var_text = f"Portfolio VaR (Batch): ${portfolio_var_value:.2f}"
        else:
             portfolio_var_text = "Portfolio VaR (Batch): N/A (Data insufficient or calculation skipped)"


        mdi_imp_data = pd.Series(result.get('mdi_imp', {})).sort_values(ascending=True)
        if mdi_imp_data.empty or mdi_imp_data.sum() == 0:
             fig1 = go.Figure().update_layout(title=f'Feature Importance for {selected_symbol}<br>(Data Not Available)')
        else:
            fig1 = px.bar(
                x=mdi_imp_data.values,
                y=mdi_imp_data.index,
                orientation='h',
                title=f'Feature Importance for {selected_symbol}',
                labels={'x': 'Importance', 'y': 'Features'},
                color=mdi_imp_data.values,
                color_continuous_scale=px.colors.sequential.Viridis
            )
            fig1.update_layout(height=400, showlegend=False, coloraxis_colorbar_title="Importance")


        mc_var_values_dict = result.get('monte_carlo_var', {})
        mc_var_values = [v.get('var', 0) for v in mc_var_values_dict.values()]
        if not mc_var_values_dict or all(v.get('var', 0) == 0 for v in mc_var_values_dict.values()):
            fig2 = go.Figure().update_layout(title=f'MC VaR Values Distribution for {selected_symbol}<br>(Data Not Available)')
        else:
            fig2 = px.histogram(
                x=mc_var_values,
                nbins=len(mc_var_values) if len(mc_var_values) < 20 else 20,
                title=f'MC VaR Values Distribution for {selected_symbol}',
                labels={'x': 'Calculated VaR ($)', 'y': 'Count'},
                color_discrete_sequence=['#1f77b4']
            )
            fig2.update_layout(height=400, showlegend=False)

        fig3, heatmap_data = generate_heatmap(selected_symbol, results)

        var_df = var_data.get(selected_symbol, pd.DataFrame())
        var_columns = [
            {'name': 'Method', 'id': 'index', 'type': 'text'},
            {'name': 'VaR ($)', 'id': 'var', 'type': 'numeric'},
            {'name': 'Revenue Impact ($)', 'id': 'revenue_impact', 'type': 'numeric'},
            {'name': 'Capital Impact ($)', 'id': 'capital_impact', 'type': 'numeric'},
            {'name': 'Liquidity Impact ($)', 'id': 'liquidity_impact', 'type': 'numeric'}
        ]
        var_data_list = var_df.reset_index().to_dict('records')


        stress_df = stress_data.get(selected_symbol, pd.DataFrame())
        stress_columns = [
            {'name': 'Scenario', 'id': 'index', 'type': 'text'},
            {'name': 'Final Value ($)', 'id': 'final_value', 'type': 'numeric'},
            {'name': 'Loss ($)', 'id': 'loss', 'type': 'numeric'},
            {'name': 'Drawdown (%)', 'id': 'drawdown', 'type': 'numeric'},
            {'name': 'Margin Shortfall ($)', 'id': 'margin_shortfall', 'type': 'numeric'},
            {'name': 'Revenue Impact ($)', 'id': 'revenue_impact', 'type': 'numeric'},
            {'name': 'Capital Impact ($)', 'id': 'capital_impact', 'type': 'numeric'},
            {'name': 'Liquidity Impact ($)', 'id': 'liquidity_impact', 'type': 'numeric'}
        ]
        stress_data_list = stress_df.reset_index().to_dict('records')

        backtest_df = backtest_data.get(selected_symbol, pd.DataFrame())
        backtest_columns = [
            {'name': 'Metric', 'id': 'index', 'type': 'text'},
            {'name': 'Exceedances', 'id': 'exceedances', 'type': 'numeric'},
            {'name': 'Coverage (%)', 'id': 'coverage', 'type': 'numeric'}
        ]
        backtest_data_list = backtest_df.reset_index().to_dict('records')

        text_output = []

        text_output.append("=== Feature Importance ===")
        if not mdi_imp_data.empty:
             for feature, importance in mdi_imp_data.items():
                 text_output.append(f"{feature}: {importance:.4f}")
        else:
             text_output.append("No feature importance data available.")
        text_output.append("")

        text_output.append("=== Portfolio VaR ===")
        text_output.append(portfolio_var_text)
        text_output.append("")


        text_output.append("=== MC VaR Values Distribution ===")
        if not mc_var_values_dict:
             text_output.append("No MC VaR values data available.")
        else:
            text_output.append("Monte Carlo VaR Values (from calculated metrics):")
            mc_vars_raw = result.get('monte_carlo_var', {})
            for key in sorted(mc_vars_raw.keys()):
                 value_info = mc_vars_raw[key]
                 text_output.append(f"{key}: ${value_info.get('var', 0.0):.2f}")
        text_output.append("")

        text_output.append("=== Risk Exposure Heat Map ===")
        if not heatmap_data.empty:
            text_output.append("Scenarios/Metrics vs. Loss/Value ($):")
            for scenario, row in heatmap_data.iterrows():
                 text_output.append(f"{scenario}: Loss/Value = ${row.iloc[0]:.2f}")
        else:
            text_output.append("No heatmap data available.")
        text_output.append("")

        text_output.append("=== VaR Table ===")
        if var_data_list:
            text_output.append("Method | VaR ($) | Revenue Impact ($) | Capital Impact ($) | Liquidity Impact ($)")
            for row in var_data_list:
                 text_output.append(f"{row.get('index', 'N/A')} | {row.get('var', 0.0):.2f} | {row.get('revenue_impact', 0.0):.2f} | {row.get('capital_impact', 0.0):.2f} | {row.get('liquidity_impact', 0.0):.2f}")
        else:
            text_output.append("No VaR table data available.")
        text_output.append("")

        text_output.append("=== Stress Test Table ===")
        if stress_data_list:
            text_output.append("Scenario | Final Value ($) | Loss ($) | Drawdown (%) | Margin Shortfall ($) | Revenue Impact ($) | Capital Impact ($) | Liquidity Impact ($)")
            for row in stress_data_list:
                text_output.append(f"{row.get('index', 'N/A')} | {row.get('final_value', 0.0):.2f} | {row.get('loss', 0.0):.2f} | {row.get('drawdown', 0.0):.2f} | {row.get('margin_shortfall', 0.0):.2f} | {row.get('revenue_impact', 0.0):.2f} | {row.get('capital_impact', 0.0):.2f} | {row.get('liquidity_impact', 0.0):.2f}")
        else:
            text_output.append("No Stress Test table data available.")
        text_output.append("")

        text_output.append("=== Backtest Table ===")
        if backtest_data_list:
            text_output.append("Metric | Exceedances | Coverage (%)")
            for row in backtest_data_list:
                 coverage_str = f"{row.get('coverage', 0.0):.2f}" if row.get('coverage') is not None else "N/A"
                 text_output.append(f"{row.get('index', 'N/A')} | {row.get('exceedances', 0)} | {coverage_str}")
        else:
            text_output.append("No Backtest table data available.")
        text_output.append("")

        text_output_str = "\n".join(text_output)

        report_data = None
        if n_clicks and result:
            report_dir = 'reports'
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            report_path = Path(report_dir) / f"{selected_symbol}_report.pdf"
            try:
                generate_report({selected_symbol: result}, {'report': {'output_dir': report_dir, 'format': 'pdf'}})
                report_data = dcc.send_file(str(report_path))
            except Exception as e:
                 logger.error(f"Report generation failed for download: {e}")


        return text, var_data_list, var_columns, stress_data_list, stress_columns, \
               backtest_data_list, backtest_columns, report_data, text_output_str, portfolio_var_text, \
               fig1, fig2, fig3


    except Exception as e:
        logger.error(f"Dashboard update failed: {e}")
        empty_figure = go.Figure().update_layout(title=f"Error loading data: {e}")
        empty_table_data = []
        empty_table_columns = []
        return [], empty_table_data, empty_table_columns, \
               empty_table_data, empty_table_columns, empty_table_data, empty_table_columns, \
               None, f"Error: {str(e)}", "Portfolio VaR (Batch): Error", \
               empty_figure, empty_figure, empty_figure


if __name__ == '__main__':
    Path('logs').mkdir(parents=True, exist_ok=True)
    app.run(debug=True)