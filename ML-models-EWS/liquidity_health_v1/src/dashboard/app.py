import os
import sys
import pandas as pd
from datetime import timedelta
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, dash_table
import numpy as np
import csv
from io import StringIO

# Debug execution context
print(f"Script location: {__file__}")
print(f"Current working directory: {os.getcwd()}")

# Set BASE_DIR relative to script location
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
print(f"Calculated BASE_DIR: {BASE_DIR}")

# Add BASE_DIR and src to sys.path
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'src'))
print(f"Updated sys.path: {sys.path}")

DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

print(f"DATA_DIR: {DATA_DIR}")
print(f"MODEL_DIR: {MODEL_DIR}")
print(f"Contents of src/core: {os.listdir(os.path.join(BASE_DIR, 'src', 'core')) if os.path.exists(os.path.join(BASE_DIR, 'src', 'core')) else 'Directory not found'}")

from src.core.lcr_models import LCRModels
from src.core.nsfr_models import NSFRModels
from src.simulation_engine import SimulationEngine

lcr_calc = LCRModels()
nsfr_calc = NSFRModels()
sim_engine = SimulationEngine(os.path.join(DATA_DIR, 'transaction_data.csv'))

df = pd.read_csv(os.path.join(DATA_DIR, 'transaction_data.csv'))
# Try multiple common date formats
for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y %H:%M:%S', '%m/%d/%Y']:
    df['date'] = pd.to_datetime(df['date'], format=fmt, errors='coerce')
    if not df['date'].isna().all():
        print(f"Successfully parsed 'date' with format: {fmt}")
        break
if df['date'].isna().all():
    print("Sample of unparsed dates:", df['date'].head().tolist())
    raise ValueError("Failed to parse 'date' column. Check the date format. Sample dates printed above.")
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

end_date = df.index.max()  # Use the latest date from the data
start_date = end_date - timedelta(days=365)
lcr_historical = lcr_series[lcr_series.index >= start_date]
nsfr_historical = nsfr_series[nsfr_series.index >= start_date]

# Train models if not pre-trained
if not os.path.exists(os.path.join(MODEL_DIR, "best_lcr_model.pkl")) or not os.path.exists(os.path.join(MODEL_DIR, "best_nsfr_model.pkl")):
    lcr_calc.get_best_model()
    nsfr_calc.get_best_model()

lcr_forecast = lcr_calc.forecast_lcr(steps=30)  # Use 30 days as a default for visualization
nsfr_forecast = nsfr_calc.forecast_nsfr(steps=365)  # 1 year by default
lcr_full = pd.concat([lcr_historical, lcr_forecast])
nsfr_full = pd.concat([nsfr_historical, nsfr_forecast])

lcr_safe = lcr_historical[lcr_historical >= 100]
lcr_unsafe = lcr_historical[lcr_historical < 100]
nsfr_safe = nsfr_historical[nsfr_historical >= 100]
nsfr_unsafe = nsfr_historical[nsfr_historical < 100]

# Identify low liquidity accounts/divisions
low_liquidity_accounts = df[df['lcr'] < 100].groupby('account_id').size().reset_index(name='low_lcr_count') if 'account_id' in df.columns else pd.DataFrame()
low_liquidity_divisions = df[df['lcr'] < 100].groupby('division').size().reset_index(name='low_lcr_count') if 'division' in df.columns else pd.DataFrame()

lcr_fig = go.Figure(
    data=[
        go.Scatter(x=lcr_safe.index, y=lcr_safe.values, mode='lines', name='Safe Historical LCR', line=dict(color='green')),
        go.Scatter(x=lcr_unsafe.index, y=lcr_unsafe.values, mode='lines', name='Unsafe Historical LCR', line=dict(color='red')),
        go.Scatter(x=lcr_forecast.index, y=lcr_forecast.values, mode='lines+markers', name='Predicted/Forecasted LCR', line=dict(color='blue', width=1), marker=dict(size=4))
    ],
    layout=go.Layout(
        title='LCR Over Time (Last Year)',
        xaxis_title='Date',
        yaxis_title='LCR (%)',
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(range=[min(lcr_full.min(), 0), max(lcr_full.max(), 500)]),
        shapes=[dict(type='line', x0=lcr_full.index[0], x1=lcr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
        margin=dict(l=50, r=50, t=50, b=50)
    )
)

nsfr_fig = go.Figure(
    data=[
        go.Scatter(x=nsfr_safe.index, y=nsfr_safe.values, mode='lines', name='Safe Historical NSFR', line=dict(color='green')),
        go.Scatter(x=nsfr_unsafe.index, y=nsfr_unsafe.values, mode='lines', name='Unsafe Historical NSFR', line=dict(color='red')),
        go.Scatter(x=nsfr_forecast.index, y=nsfr_forecast.values, mode='lines', name='Forecasted NSFR', line=dict(dash='dash', color='blue'))
    ],
    layout=go.Layout(
        title='NSFR Over Time (Last Year)',
        xaxis_title='Date',
        yaxis_title='NSFR (%)',
        hovermode='x unified',
        template='plotly_white',
        yaxis=dict(range=[min(nsfr_full.min(), 0), max(nsfr_full.max(), 150)]),
        shapes=[dict(type='line', x0=nsfr_full.index[0], x1=nsfr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
        margin=dict(l=50, r=50, t=50, b=50)
    )
)

lcr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'LCR (%)': round(val, 2)} for idx, val in lcr_full.items()]
nsfr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'NSFR (%)': round(val, 2)} for idx, val in nsfr_full.items()]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], suppress_callback_exceptions=True)
server = app.server

app.layout = dbc.Container([
    html.H1("Liquidity & Stability Dashboard", className="text-center my-4", style={'color': '#2c3e50'}),
    dbc.Row([
        dbc.Col(html.H4("Last Updated: " + end_date.strftime('%Y-%m-%d %H:%M:%S'), className="text-muted"), width=12, className="text-center mb-3")
    ]),
    dbc.Tabs([
        dbc.Tab(label="LCR Analysis", tab_id="lcr-tab", children=[
            dbc.Card([
                dbc.CardHeader(html.H5("LCR Forecast & Historical Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='lcr-forecast-period',
                                type='number',
                                placeholder='Forecast Periods (e.g., 1)',
                                min=1,
                                value=1,
                                style={'width': '150px', 'margin-bottom': '10px'}
                            ),
                            dbc.Button('Update Forecast', id='update-lcr-button', color="primary", className="ms-2")
                        ], width=4),
                        dbc.Col([
                            html.Div(id='lcr-forecast-output', style={'font-weight': 'bold', 'margin-top': '5px'})
                        ], width=8)
                    ]),
                    dcc.Graph(id='lcr-plot', figure=lcr_fig, style={'height': '400px'}),
                    html.H5("Low Liquidity Accounts", className="mt-4"),
                    dash_table.DataTable(
                        id='low-lcr-accounts-table',
                        columns=[{'name': i, 'id': i} for i in low_liquidity_accounts.columns],
                        data=low_liquidity_accounts.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': 14},
                        style_header={'backgroundColor': '#e74c3c', 'color': 'white', 'fontWeight': 'bold'},
                        page_size=5
                    ),
                    html.H5("Low Liquidity Divisions", className="mt-4"),
                    dash_table.DataTable(
                        id='low-lcr-divisions-table',
                        columns=[{'name': i, 'id': i} for i in low_liquidity_divisions.columns],
                        data=low_liquidity_divisions.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': 14},
                        style_header={'backgroundColor': '#e74c3c', 'color': 'white', 'fontWeight': 'bold'},
                        page_size=5
                    ),
                    dash_table.DataTable(
                        id='lcr-table',
                        columns=[{'name': i, 'id': i} for i in ['Date', 'LCR (%)']],
                        data=lcr_table_data,
                        style_table={'overflowX': 'auto', 'margin-top': '20px'},
                        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': 14},
                        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{LCR (%)} < 100'}, 'backgroundColor': '#ffcccc', 'color': 'black'},
                            {'if': {'column_id': 'LCR (%)', 'filter_query': '{Date} >= ' + end_date.strftime('%Y-%m-%d')}, 'backgroundColor': '#d5f5d5', 'color': 'darkgreen'}
                        ],
                        page_size=10
                    ),
                    dbc.Button('Export Alerts', id='export-lcr-alerts', color="warning", className="mt-3"),
                    dcc.Download(id="download-lcr-alerts")
                ])
            ], className="mb-4")
        ]),
        dbc.Tab(label="NSFR Analysis", tab_id="nsfr-tab", children=[
            dbc.Card([
                dbc.CardHeader(html.H5("NSFR Forecast & Historical Data", className="mb-0")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='nsfr-forecast-period',
                                type='number',
                                placeholder='Forecast Periods (e.g., 1)',
                                min=1,
                                value=1,
                                style={'width': '150px', 'margin-bottom': '10px'}
                            ),
                            dbc.Button('Update Forecast', id='update-nsfr-button', color="primary", className="ms-2")
                        ], width=4),
                        dbc.Col([
                            html.Div(id='nsfr-forecast-output', style={'font-weight': 'bold', 'margin-top': '5px'})
                        ], width=8)
                    ]),
                    dcc.Graph(id='nsfr-plot', figure=nsfr_fig, style={'height': '400px'}),
                    dash_table.DataTable(
                        id='nsfr-table',
                        columns=[{'name': i, 'id': i} for i in ['Date', 'NSFR (%)']],
                        data=nsfr_table_data,
                        style_table={'overflowX': 'auto', 'margin-top': '20px'},
                        style_cell={'textAlign': 'center', 'padding': '8px', 'fontSize': 14},
                        style_header={'backgroundColor': '#3498db', 'color': 'white', 'fontWeight': 'bold'},
                        style_data_conditional=[
                            {'if': {'filter_query': '{NSFR (%)} < 100'}, 'backgroundColor': '#ffcccc', 'color': 'black'},
                            {'if': {'column_id': 'NSFR (%)', 'filter_query': '{Date} >= ' + end_date.strftime('%Y-%m-%d')}, 'backgroundColor': '#d5f5d5', 'color': 'darkgreen'}
                        ],
                        page_size=10
                    )
                ])
            ], className="mb-4")
        ])
    ], active_tab="lcr-tab"),
    html.H5("Stress Simulation", className="mt-4"),
    dbc.Form([
        dbc.Row([
            dbc.Col(dbc.Input(id='hqla-adjustment', type='number', placeholder='HQLA Adjustment', value=0)),
            dbc.Col(dbc.Input(id='outflow-adjustment', type='number', placeholder='Outflow Adjustment', value=0)),
            dbc.Col(dbc.Input(id='inflow-adjustment', type='number', placeholder='Inflow Adjustment', value=0)),
            dbc.Col(dbc.Input(id='stable-funding-adjustment', type='number', placeholder='Stable Funding Adjustment', value=0)),
            dbc.Col(dbc.Input(id='required-funding-adjustment', type='number', placeholder='Required Funding Adjustment', value=0)),
            dbc.Col(dbc.Button('Run Simulation', id='run-simulation', color="primary"))
        ]),
        dbc.Row([
            dbc.Col(html.Div(id='simulation-output'))
        ])
    ])
], fluid=True, style={'padding': '20px', 'background-color': '#f5f6fa'})

@app.callback(
    Output('lcr-plot', 'figure'),
    Output('lcr-table', 'data'),
    Output('lcr-forecast-output', 'children'),
    Input('update-lcr-button', 'n_clicks'),
    State('lcr-forecast-period', 'value')
)
def update_lcr(n_clicks, forecast_periods):
    if n_clicks is None or forecast_periods is None or forecast_periods <= 0:
        lcr_forecast = lcr_calc.forecast_lcr(steps=30)  # Use global lcr_calc
        lcr_full = pd.concat([lcr_historical, lcr_forecast])
        lcr_safe = lcr_historical[lcr_historical >= 100]
        lcr_unsafe = lcr_historical[lcr_historical < 100]
        lcr_fig = go.Figure(
            data=[
                go.Scatter(x=lcr_safe.index, y=lcr_safe.values, mode='lines', name='Safe Historical LCR', line=dict(color='green')),
                go.Scatter(x=lcr_unsafe.index, y=lcr_unsafe.values, mode='lines', name='Unsafe Historical LCR', line=dict(color='red')),
                go.Scatter(x=lcr_forecast.index, y=lcr_forecast.values, mode='lines+markers', name='Predicted/Forecasted LCR', line=dict(color='blue', width=1), marker=dict(size=4))
            ],
            layout=go.Layout(
                title='LCR Over Time (Last Year)',
                xaxis_title='Date',
                yaxis_title='LCR (%)',
                hovermode='x unified',
                template='plotly_white',
                yaxis=dict(range=[min(lcr_full.min(), 0), max(lcr_full.max(), 500)]),
                shapes=[dict(type='line', x0=lcr_full.index[0], x1=lcr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
                margin=dict(l=50, r=50, t=50, b=50)
            )
        )
        lcr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'LCR (%)': round(val, 2)} for idx, val in lcr_full.items()]
        return lcr_fig, lcr_table_data, f"Current Predicted Range: {round(lcr_forecast.min(), 2)}% to {round(lcr_forecast.max(), 2)}% on {lcr_forecast.index[0].strftime('%Y-%m-%d')} to {lcr_forecast.index[-1].strftime('%Y-%m-%d')}"
    lcr_forecast = lcr_calc.forecast_lcr(steps=30 * forecast_periods)  # Use global lcr_calc, 30 days per period
    lcr_full = pd.concat([lcr_historical, lcr_forecast])
    lcr_safe = lcr_historical[lcr_historical >= 100]
    lcr_unsafe = lcr_historical[lcr_historical < 100]
    lcr_fig = go.Figure(
        data=[
            go.Scatter(x=lcr_safe.index, y=lcr_safe.values, mode='lines', name='Safe Historical LCR', line=dict(color='green')),
            go.Scatter(x=lcr_unsafe.index, y=lcr_unsafe.values, mode='lines', name='Unsafe Historical LCR', line=dict(color='red')),
            go.Scatter(x=lcr_forecast.index, y=lcr_forecast.values, mode='lines+markers', name='Predicted/Forecasted LCR', line=dict(color='blue', width=1), marker=dict(size=4))
        ],
        layout=go.Layout(
            title='LCR Over Time (Last Year)',
            xaxis_title='Date',
            yaxis_title='LCR (%)',
            hovermode='x unified',
            template='plotly_white',
            yaxis=dict(range=[min(lcr_full.min(), 0), max(lcr_full.max(), 500)]),
            shapes=[dict(type='line', x0=lcr_full.index[0], x1=lcr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
            margin=dict(l=50, r=50, t=50, b=50)
        )
    )
    lcr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'LCR (%)': round(val, 2)} for idx, val in lcr_full.items()]
    print(f"Debug: Forecast period {forecast_periods}, steps {30 * forecast_periods}, dates {lcr_forecast.index[0]} to {lcr_forecast.index[-1]}")  # Debug print
    return lcr_fig, lcr_table_data, f"Updated Predicted Range ({forecast_periods} period(s)): {round(lcr_forecast.min(), 2)}% to {round(lcr_forecast.max(), 2)}% on {lcr_forecast.index[0].strftime('%Y-%m-%d')} to {lcr_forecast.index[-1].strftime('%Y-%m-%d')}"

@app.callback(
    Output('nsfr-plot', 'figure'),
    Output('nsfr-table', 'data'),
    Output('nsfr-forecast-output', 'children'),
    Input('update-nsfr-button', 'n_clicks'),
    State('nsfr-forecast-period', 'value')
)
def update_nsfr(n_clicks, forecast_periods):
    if n_clicks is None or forecast_periods is None or forecast_periods <= 0:
        nsfr_forecast = nsfr_calc.forecast_nsfr(steps=365)  # 1 year by default
        nsfr_full = pd.concat([nsfr_historical, nsfr_forecast])
        nsfr_safe = nsfr_historical[nsfr_historical >= 100]
        nsfr_unsafe = nsfr_historical[nsfr_historical < 100]
        nsfr_fig = go.Figure(
            data=[
                go.Scatter(x=nsfr_safe.index, y=nsfr_safe.values, mode='lines', name='Safe Historical NSFR', line=dict(color='green')),
                go.Scatter(x=nsfr_unsafe.index, y=nsfr_unsafe.values, mode='lines', name='Unsafe Historical NSFR', line=dict(color='red')),
                go.Scatter(x=nsfr_forecast.index, y=nsfr_forecast.values, mode='lines', name='Forecasted NSFR', line=dict(dash='dash', color='blue'))
            ],
            layout=go.Layout(
                title='NSFR Over Time (Last Year)',
                xaxis_title='Date',
                yaxis_title='NSFR (%)',
                hovermode='x unified',
                template='plotly_white',
                yaxis=dict(range=[min(nsfr_full.min(), 0), max(nsfr_full.max(), 150)]),
                shapes=[dict(type='line', x0=nsfr_full.index[0], x1=nsfr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
                margin=dict(l=50, r=50, t=50, b=50)
            )
        )
        nsfr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'NSFR (%)': round(val, 2)} for idx, val in nsfr_full.items()]
        return nsfr_fig, nsfr_table_data, f"Current Forecast (365 day(s)): {round(nsfr_forecast.values[0], 2)}% on {nsfr_forecast.index[0].strftime('%Y-%m-%d')}"
    nsfr_forecast = nsfr_calc.forecast_nsfr(steps=forecast_periods)  # Use days directly
    nsfr_full = pd.concat([nsfr_historical, nsfr_forecast])
    nsfr_safe = nsfr_historical[nsfr_historical >= 100]
    nsfr_unsafe = nsfr_historical[nsfr_historical < 100]
    nsfr_fig = go.Figure(
        data=[
            go.Scatter(x=nsfr_safe.index, y=nsfr_safe.values, mode='lines', name='Safe Historical NSFR', line=dict(color='green')),
            go.Scatter(x=nsfr_unsafe.index, y=nsfr_unsafe.values, mode='lines', name='Unsafe Historical NSFR', line=dict(color='red')),
            go.Scatter(x=nsfr_forecast.index, y=nsfr_forecast.values, mode='lines', name='Forecasted NSFR', line=dict(dash='dash', color='blue'))
        ],
        layout=go.Layout(
            title='NSFR Over Time (Last Year)',
            xaxis_title='Date',
            yaxis_title='NSFR (%)',
            hovermode='x unified',
            template='plotly_white',
            yaxis=dict(range=[min(nsfr_full.min(), 0), max(nsfr_full.max(), 150)]),
            shapes=[dict(type='line', x0=nsfr_full.index[0], x1=nsfr_full.index[-1], y0=100, y1=100, line=dict(color='black', dash='dash'))],
            margin=dict(l=50, r=50, t=50, b=50)
        )
    )
    nsfr_table_data = [{'Date': idx.strftime('%Y-%m-%d'), 'NSFR (%)': round(val, 2)} for idx, val in nsfr_full.items()]
    return nsfr_fig, nsfr_table_data, f"Updated Forecast ({forecast_periods} day(s)): {round(nsfr_forecast.values[0], 2)}% on {nsfr_forecast.index[0].strftime('%Y-%m-%d')}"

@app.callback(
    Output("download-lcr-alerts", "data"),
    Input("export-lcr-alerts", "n_clicks"),
    prevent_initial_call=True
)
def export_lcr_alerts(n_clicks):
    if n_clicks:
        low_lcr_data = df[df['lcr'] < 100]
        if not low_lcr_data.empty:
            output = StringIO()
            writer = csv.writer(output)
            writer.writerow(['Date', 'Account ID', 'Division', 'LCR (%)', 'Recommendation'])
            for index, row in low_lcr_data.iterrows():
                recommendation = f"Increase HQLA by {max(100 - row['lcr'], 10)}% or reduce outflows by {max(100 - row['lcr'], 10)}% for account {row['account_id']} in division {row['division']}."
                writer.writerow([index.strftime('%Y-%m-%d'), row['account_id'], row['division'], round(row['lcr'], 2), recommendation])
            return dcc.send_string(output.getvalue(), filename="lcr_alerts.csv")
        return dcc.send_string("No alerts to export.", filename="lcr_alerts.txt")



@app.callback(
    Output('simulation-output', 'children'),
    Input('run-simulation', 'n_clicks'),
    State('hqla-adjustment', 'value'),
    State('outflow-adjustment', 'value'),
    State('inflow-adjustment', 'value'),
    State('stable-funding-adjustment', 'value'),
    State('required-funding-adjustment', 'value')
)
def run_simulation(n_clicks, hqla_adj, outflow_adj, inflow_adj, stable_adj, req_adj):
    if n_clicks:
        simulations = sim_engine.run_stress_simulation(
            hqla_adjustment=hqla_adj or 0,
            outflow_adjustment=outflow_adj or 0,
            inflow_adjustment=inflow_adj or 0,
            stable_funding_adjustment=stable_adj or 0,
            required_funding_adjustment=req_adj or 0
        )
        output = []
        for days, series in simulations.items():
            min_val = round(series.min(), 2)
            max_val = round(series.max(), 2)
            avg_val = round(series.mean(), 2)
            output.append(html.P(f"Stress Simulation ({days} days): LCR/NSFR ≈ {min_val}% to {max_val}% (avg: {avg_val}%)"))
        return output
    return "Enter adjustments and click 'Run Simulation' to see results."


if __name__ == '__main__':
    print("Starting Dash server...")
    app.run(debug=True, host='localhost', port=8050)