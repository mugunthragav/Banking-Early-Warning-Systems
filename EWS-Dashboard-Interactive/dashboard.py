import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table
from flask import Flask
import numpy as np
from datetime import datetime

# Initialize Flask and Dash
flask_app = Flask(__name__)
dash_app = Dash(__name__, server=flask_app, url_base_pathname='/dashboard/')

# Load datasets from generate_data.py
try:
    credit_df = pd.read_csv('data/credit_risk_data.csv')
    liquidity_df = pd.read_csv('data/liquidity_risk_data.csv')
    market_df = pd.read_csv('data/market_risk_data.csv')
    capital_df = pd.read_csv('data/capital_compliance_data.csv')
except FileNotFoundError:
    print("Error: Dataset files not found. Please run generate_data.py first.")
    exit(1)

# Constants from generate_data.py
BANKS = ['Citigroup', 'HSBC', 'Deutsche Bank', 'Barclays']
CURRENCIES = ['USD', 'EUR', 'GBP', 'CAD']
RATINGS = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'D']

# Basel III/IV and IFRS 9 compliant metric calculations
def calculate_npl_ratio(df, threshold=90):
    """Calculate Non-Performing Loan Ratio per Basel III (90 days past due)"""
    npl_amount = df[df['Days_Overdue'] >= threshold]['Balance'].sum()
    total_amount = df['Balance'].sum()
    return (npl_amount / total_amount * 100) if total_amount > 0 else 0

def calculate_ifrs9_stage(df):
    """Assign IFRS 9 stages based on credit risk deterioration"""
    conditions = [
        (df['Days_Overdue'] >= 90) | (df['Default_Prob'] > 0.1),  # Stage 3: Credit-impaired
        (df['Days_Overdue'] > 30) | (df['Default_Prob'] > 0.05),  # Stage 2: Significant increase
        (df['Days_Overdue'] <= 30) & (df['Default_Prob'] <= 0.05)  # Stage 1: Performing
    ]
    choices = [3, 2, 1]
    return np.select(conditions, choices, default=1)

def calculate_lcr(df, hqla_threshold=100):
    """Calculate Liquidity Coverage Ratio per Basel III"""
    hqla = df[df['LCR_Compliance']]['Buffer_Amount'].sum()  # High-Quality Liquid Assets
    net_outflows = df['Asset_Amount'].sum() * 0.3  # 30% runoff rate
    lcr = (hqla / net_outflows * 100) if net_outflows > 0 else 0
    return lcr, lcr >= hqla_threshold

def calculate_var(df, confidence=0.95):
    """Calculate Value at Risk per Basel III market risk framework"""
    return df['VaR_95'].sum()

def calculate_capital_ratio(df, threshold=8):
    """Calculate Capital Adequacy Ratio per Basel III/IV"""
    capital = df['Capital_Amount'].sum()
    rwa = df['RWA'].sum()
    ratio = (capital / rwa * 100) if rwa > 0 else 0
    return ratio, ratio >= threshold

# Initial calculations
credit_df['IFRS9_Stage'] = calculate_ifrs9_stage(credit_df)

# Dash layout
dash_app.layout = html.Div([
    html.H1("Bank Risk Management Dashboard", style={
        'textAlign': 'center', 'color': 'white', 'backgroundColor': '#003087',
        'padding': '20px', 'fontFamily': 'Arial', 'fontSize': '30px', 'margin': '0'
    }),
    html.Div([
        html.H3("Configuration Panel", style={'fontWeight': 'bold', 'marginBottom': '10px'}),
        html.Label("Select Bank:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='bank-filter',
            options=[{'label': bank, 'value': bank} for bank in sorted(BANKS)] + [{'label': 'All Banks', 'value': 'All'}],
            value='All',
            style={'width': '50%', 'marginBottom': '10px'}
        ),
        html.Label("Select Currency:", style={'fontWeight': 'bold'}),
        dcc.Dropdown(
            id='currency-filter',
            options=[{'label': curr, 'value': curr} for curr in sorted(CURRENCIES)] + [{'label': 'All Currencies', 'value': 'All'}],
            value='All',
            style={'width': '50%', 'marginBottom': '10px'}
        ),
        html.Label("NPL Days Overdue Threshold:", style={'fontWeight': 'bold'}),
        dcc.Slider(id='npl-threshold', min=30, max=180, step=30, value=90,
                   marks={i: str(i) for i in range(30, 181, 30)},
                   tooltip={'placement': 'bottom', 'always_visible': True}),
        html.Label("LCR Minimum Threshold (%):", style={'fontWeight': 'bold'}),
        dcc.Slider(id='lcr-threshold', min=50, max=150, step=10, value=100,
                   marks={i: str(i) for i in range(50, 151, 10)},
                   tooltip={'placement': 'bottom', 'always_visible': True}),
        html.Label("Capital Ratio Threshold (%):", style={'fontWeight': 'bold'}),
        dcc.Slider(id='capital-threshold', min=4, max=12, step=1, value=8,
                   marks={i: str(i) for i in range(4, 13, 1)},
                   tooltip={'placement': 'bottom', 'always_visible': True}),
        html.Button("Apply Changes", id='apply-button', n_clicks=0,
                    style={'marginTop': '15px', 'backgroundColor': '#005f73', 'color': 'white',
                           'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer'})
    ], style={'padding': '20px', 'backgroundColor': '#e6ecef', 'margin': '20px', 'borderRadius': '5px'}),
    dcc.Tabs([
        dcc.Tab(label='Credit Risk', children=[
            html.H3("Key Metrics", style={'marginTop': '10px'}),
            html.Div([
                dcc.Graph(id='npl-gauge', style={'display': 'inline-block', 'width': '33%'}),
                dcc.Graph(id='ifrs9-pie', style={'display': 'inline-block', 'width': '33%'}),
                dcc.Graph(id='rating-bar', style={'display': 'inline-block', 'width': '33%'}),
            ]),
            html.H3("Loan Portfolio", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='credit-table',
                columns=[
                    {'name': 'Loan ID', 'id': 'Loan_ID'},
                    {'name': 'Bank', 'id': 'Bank'},
                    {'name': 'Balance ($)', 'id': 'Balance', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rating', 'id': 'Rating'},
                    {'name': 'Status', 'id': 'Status'},
                    {'name': 'IFRS 9 Stage', 'id': 'IFRS9_Stage'}
                ],
                style_table={'overflowX': 'auto', 'margin': '10px'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#003087', 'color': 'white', 'fontWeight': 'bold'},
                page_size=10
            )
        ]),
        dcc.Tab(label='Liquidity Risk', children=[
            html.H3("Key Metrics", style={'marginTop': '10px'}),
            html.Div([
                dcc.Graph(id='lcr-gauge', style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(id='asset-bar', style={'display': 'inline-block', 'width': '50%'}),
            ]),
            html.H3("Asset Portfolio", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='liquidity-table',
                columns=[
                    {'name': 'Asset ID', 'id': 'Asset_ID'},
                    {'name': 'Bank', 'id': 'Bank'},
                    {'name': 'Asset Amount ($)', 'id': 'Asset_Amount', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Rating', 'id': 'Rating'},
                    {'name': 'LCR Compliant', 'id': 'LCR_Compliance'}
                ],
                style_table={'overflowX': 'auto', 'margin': '10px'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#003087', 'color': 'white', 'fontWeight': 'bold'},
                page_size=10
            )
        ]),
        dcc.Tab(label='Market Risk', children=[
            html.H3("Key Metrics", style={'marginTop': '10px'}),
            html.Div([
                dcc.Graph(id='var-gauge', style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(id='instrument-bar', style={'display': 'inline-block', 'width': '50%'}),
            ]),
            html.H3("Trading Book", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='market-table',
                columns=[
                    {'name': 'Trade ID', 'id': 'Trade_ID'},
                    {'name': 'Bank', 'id': 'Bank'},
                    {'name': 'Value ($)', 'id': 'Value', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Instrument', 'id': 'Instrument'},
                    {'name': 'VaR 95% ($)', 'id': 'VaR_95', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                style_table={'overflowX': 'auto', 'margin': '10px'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#003087', 'color': 'white', 'fontWeight': 'bold'},
                page_size=10
            )
        ]),
        dcc.Tab(label='Capital & Compliance', children=[
            html.H3("Key Metrics", style={'marginTop': '10px'}),
            html.Div([
                dcc.Graph(id='capital-gauge', style={'display': 'inline-block', 'width': '50%'}),
                dcc.Graph(id='counterparty-bar', style={'display': 'inline-block', 'width': '50%'}),
            ]),
            html.H3("Capital Overview", style={'marginTop': '20px'}),
            dash_table.DataTable(
                id='capital-table',
                columns=[
                    {'name': 'Capital ID', 'id': 'Capital_ID'},
                    {'name': 'Bank', 'id': 'Bank'},
                    {'name': 'Capital Amount ($)', 'id': 'Capital_Amount', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'Capital Ratio (%)', 'id': 'Capital_Ratio', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    {'name': 'RWA ($)', 'id': 'RWA', 'type': 'numeric', 'format': {'specifier': '.2f'}}
                ],
                style_table={'overflowX': 'auto', 'margin': '10px'},
                style_cell={'textAlign': 'left', 'padding': '8px', 'fontSize': '14px'},
                style_header={'backgroundColor': '#003087', 'color': 'white', 'fontWeight': 'bold'},
                page_size=10
            )
        ])
    ]),
    html.Div(id='last-updated', style={'textAlign': 'right', 'padding': '10px', 'fontStyle': 'italic', 'fontSize': '14px'})
], style={'fontFamily': 'Arial', 'backgroundColor': '#f4f7fa', 'padding': '20px'})

# Dash callbacks
@dash_app.callback(
    [
        Output('npl-gauge', 'figure'),
        Output('ifrs9-pie', 'figure'),
        Output('rating-bar', 'figure'),
        Output('credit-table', 'data'),
        Output('lcr-gauge', 'figure'),
        Output('asset-bar', 'figure'),
        Output('liquidity-table', 'data'),
        Output('var-gauge', 'figure'),
        Output('instrument-bar', 'figure'),
        Output('market-table', 'data'),
        Output('capital-gauge', 'figure'),
        Output('counterparty-bar', 'figure'),
        Output('capital-table', 'data'),
        Output('last-updated', 'children')
    ],
    [Input('apply-button', 'n_clicks')],
    [
        State('npl-threshold', 'value'),
        State('lcr-threshold', 'value'),
        State('capital-threshold', 'value'),
        State('bank-filter', 'value'),
        State('currency-filter', 'value')
    ]
)
def update_dashboard(n_clicks, npl_threshold, lcr_threshold, capital_threshold, bank_filter, currency_filter):
    # Filter datasets
    filtered_credit = credit_df.copy()
    filtered_liquidity = liquidity_df.copy()
    filtered_market = market_df.copy()
    filtered_capital = capital_df.copy()

    if bank_filter != 'All':
        filtered_credit = filtered_credit[filtered_credit['Bank'] == bank_filter]
        filtered_liquidity = filtered_liquidity[filtered_liquidity['Bank'] == bank_filter]
        filtered_market = filtered_market[filtered_market['Bank'] == bank_filter]
        filtered_capital = filtered_capital[filtered_capital['Bank'] == bank_filter]

    if currency_filter != 'All':
        filtered_liquidity = filtered_liquidity[filtered_liquidity['Currency'] == currency_filter]
        filtered_market = filtered_market[filtered_market['Currency'] == currency_filter]

    # Recalculate metrics
    npl_ratio = calculate_npl_ratio(filtered_credit, npl_threshold)
    filtered_credit['IFRS9_Stage'] = calculate_ifrs9_stage(filtered_credit)
    lcr, lcr_compliant = calculate_lcr(filtered_liquidity, lcr_threshold)
    var_95 = calculate_var(filtered_market)
    capital_ratio, capital_compliant = calculate_capital_ratio(filtered_capital, capital_threshold)

    # NPL Gauge
    npl_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=npl_ratio,
        title={'text': "NPL Ratio (%) - Basel III"},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "#d90429" if npl_ratio > 5 else "#0a9396"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': 5},
            'steps': [
                {'range': [0, 5], 'color': "#e9f5db"},
                {'range': [5, 10], 'color': "#ffccd5"},
                {'range': [10, 20], 'color': "#ff4d6d"}
            ]
        }
    ))

    # IFRS 9 Stage Pie
    stage_counts = filtered_credit['IFRS9_Stage'].value_counts().reindex([1, 2, 3], fill_value=0)
    ifrs9_pie = px.pie(
        names=['Stage 1', 'Stage 2', 'Stage 3'],
        values=stage_counts,
        title="IFRS 9 Stage Distribution",
        color_discrete_sequence=['#0a9396', '#ffb703', '#d90429']
    )

    # Credit Rating Bar
    rating_counts = filtered_credit['Rating'].value_counts().reindex(RATINGS, fill_value=0)
    rating_bar = px.bar(
        x=rating_counts.index, y=rating_counts.values, title="Credit Rating Distribution",
        labels={'x': 'Rating', 'y': 'Count'}, color_discrete_sequence=['#003087']
    )

    # LCR Gauge
    lcr_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=lcr,
        title={'text': f"LCR (%) - {'Compliant' if lcr_compliant else 'Non-Compliant'}"},
        gauge={
            'axis': {'range': [0, 200]},
            'bar': {'color': "#d90429" if not lcr_compliant else "#0a9396"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': lcr_threshold},
            'steps': [
                {'range': [0, lcr_threshold], 'color': "#ffccd5"},
                {'range': [lcr_threshold, 200], 'color': "#e9f5db"}
            ]
        }
    ))

    # Asset Category Bar
    asset_counts = filtered_liquidity['Asset_Category'].value_counts()
    asset_bar = px.bar(
        x=asset_counts.index, y=asset_counts.values, title="Asset Category Distribution",
        labels={'x': 'Category', 'y': 'Count'}, color_discrete_sequence=['#003087']
    )

    # VaR Gauge
    var_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=var_95,
        title={'text': "VaR 95% ($) - Basel III"},
        gauge={
            'axis': {'range': [0, var_95 * 1.5]},
            'bar': {'color': "#0a9396"},
            'steps': [{'range': [0, var_95 * 1.5], 'color': "#e9f5db"}]
        }
    ))

    # Instrument Bar
    instrument_counts = filtered_market['Instrument'].value_counts()
    instrument_bar = px.bar(
        x=instrument_counts.index, y=instrument_counts.values, title="Instrument Distribution",
        labels={'x': 'Instrument', 'y': 'Count'}, color_discrete_sequence=['#003087']
    )

    # Capital Ratio Gauge
    capital_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=capital_ratio,
        title={'text': f"Capital Ratio (%) - {'Compliant' if capital_compliant else 'Non-Compliant'}"},
        gauge={
            'axis': {'range': [0, 20]},
            'bar': {'color': "#d90429" if not capital_compliant else "#0a9396"},
            'threshold': {'line': {'color': "red", 'width': 4}, 'value': capital_threshold},
            'steps': [
                {'range': [0, capital_threshold], 'color': "#ffccd5"},
                {'range': [capital_threshold, 20], 'color': "#e9f5db"}
            ]
        }
    ))

    # Counterparty Rating Bar
    counterparty_counts = filtered_capital['Rating'].value_counts().reindex(RATINGS, fill_value=0)
    counterparty_bar = px.bar(
        x=counterparty_counts.index, y=counterparty_counts.values, title="Counterparty Rating Distribution",
        labels={'x': 'Rating', 'y': 'Count'}, color_discrete_sequence=['#003087']
    )

    # Table data
    credit_table_data = filtered_credit[[
        'Loan_ID', 'Bank', 'Balance', 'Rating', 'Status', 'IFRS9_Stage'
    ]].to_dict('records')
    liquidity_table_data = filtered_liquidity[[
        'Asset_ID', 'Bank', 'Asset_Amount', 'Rating', 'LCR_Compliance'
    ]].to_dict('records')
    market_table_data = filtered_market[[
        'Trade_ID', 'Bank', 'Value', 'Instrument', 'VaR_95'
    ]].to_dict('records')
    capital_table_data = filtered_capital[[
        'Capital_ID', 'Bank', 'Capital_Amount', 'Capital_Ratio', 'RWA'
    ]].to_dict('records')

    # Last updated timestamp
    last_updated = f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    return (
        npl_fig, ifrs9_pie, rating_bar, credit_table_data,
        lcr_fig, asset_bar, liquidity_table_data,
        var_fig, instrument_bar, market_table_data,
        capital_fig, counterparty_bar, capital_table_data,
        last_updated
    )

# Flask API endpoints
@flask_app.route('/api/npl_ratio', methods=['POST'])
def npl_ratio_endpoint():
    from flask import request
    data = request.get_json()
    df = pd.DataFrame(data['data'])
    threshold = data.get('threshold', 90)
    npl_ratio = calculate_npl_ratio(df, threshold)
    return {'npl_ratio': npl_ratio, 'threshold': threshold}

@flask_app.route('/api/lcr', methods=['POST'])
def lcr_endpoint():
    from flask import request
    data = request.get_json()
    df = pd.DataFrame(data['data'])
    threshold = data.get('threshold', 100)
    lcr, compliant = calculate_lcr(df, threshold)
    return {'lcr': lcr, 'compliant': compliant, 'threshold': threshold}

@flask_app.route('/api/capital_ratio', methods=['POST'])
def capital_ratio_endpoint():
    from flask import request
    data = request.get_json()
    df = pd.DataFrame(data['data'])
    threshold = data.get('threshold', 8)
    ratio, compliant = calculate_capital_ratio(df, threshold)
    return {'capital_ratio': ratio, 'compliant': compliant, 'threshold': threshold}

# Run the server
if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=8050, debug=False)