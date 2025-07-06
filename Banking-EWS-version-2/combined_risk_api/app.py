import dash
from dash import html, dcc, Input, Output, State
import requests
import pandas as pd
import plotly.io as pio
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[
    'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'
])

# API base URL
API_BASE_URL = "http://localhost:8000"

# Layout
app.layout = html.Div(className="container mx-auto p-4", children=[
    html.H1("Combined Risk API Dashboard", className="text-3xl font-bold mb-4"),
    
    # Health Check Section
    html.Div(className="mb-6", children=[
        html.H2("API Health Check", className="text-xl font-semibold mb-2"),
        html.Button("Check Health", id="health-button", className="bg-blue-500 text-white px-4 py-2 rounded"),
        html.Pre(id="health-output", className="mt-2 p-2 bg-gray-100 rounded")
    ]),
    
    # Liquidity Prediction Section
    html.Div(className="mb-6", children=[
        html.H2("Liquidity Prediction", className="text-xl font-semibold mb-2"),
        html.Label("Cash (13_CASH):", className="block mb-1"),
        dcc.Input(id="input-cash", type="number", value=5000000, className="border p-2 mb-2 w-full"),
        html.Label("Treasury Bills:", className="block mb-1"),
        dcc.Input(id="input-treasury", type="number", value=7000000, className="border p-2 mb-2 w-full"),
        html.Label("Deposit Growth Rate:", className="block mb-1"),
        dcc.Input(id="input-deposit-growth", type="number", value=0.4021, step=0.0001, className="border p-2 mb-2 w-full"),
        html.Label("Funding Cost Change Proxy:", className="block mb-1"),
        dcc.Input(id="input-funding-cost", type="number", value=-0.021, step=0.0001, className="border p-2 mb-2 w-full"),
        html.Label("Exposed Banks (comma-separated):", className="block mb-1"),
        dcc.Input(id="input-banks", type="text", value="Safe Bank 1,SBI,Safe Bank 2", className="border p-2 mb-2 w-full"),
        html.Label("Exposure Amounts (comma-separated):", className="block mb-1"),
        dcc.Input(id="input-amounts", type="text", value="500000,250000,1000000", className="border p-2 mb-2 w-full"),
        html.Button("Predict Liquidity Risk", id="liquidity-button", className="bg-blue-500 text-white px-4 py-2 rounded"),
        html.Pre(id="liquidity-output", className="mt-2 p-2 bg-gray-100 rounded")
    ]),
    
    # Market Metrics Section
    html.Div(className="mb-6", children=[
        html.H2("Market Metrics (@CL#C)", className="text-xl font-semibold mb-2"),
        html.Button("Fetch Metrics", id="metrics-button", className="bg-blue-500 text-white px-4 py-2 rounded"),
        html.Pre(id="metrics-output", className="mt-2 p-2 bg-gray-100 rounded")
    ]),
    
    # Market Visualizations Section
    html.Div(className="mb-6", children=[
        html.H2("Market Visualizations (@CL#C)", className="text-xl font-semibold mb-2"),
        html.Button("Fetch Visualizations", id="visualizations-button", className="bg-blue-500 text-white px-4 py-2 rounded"),
        dcc.Graph(id="feature-importance-plot"),
        dcc.Graph(id="mc-var-plot")
    ]),
    
    # Market Report Section
    html.Div(className="mb-6", children=[
        html.H2("Market Report (@CL#C)", className="text-xl font-semibold mb-2"),
        html.A("Download Report", id="report-download", href="", download="@CL#C_report.pdf", className="bg-blue-500 text-white px-4 py-2 rounded"),
        html.Pre(id="report-output", className="mt-2 p-2 bg-gray-100 rounded")
    ]),
    
    # Credit Placeholder Section
    html.Div(className="mb-6", children=[
        html.H2("Credit Risk (Mocked)", className="text-xl font-semibold mb-2"),
        html.P("Credit Risk module is mocked. Provide initial_preprocessor.py for real functionality.", className="text-gray-600")
    ])
])

# Callbacks
@app.callback(
    Output("health-output", "children"),
    Input("health-button", "n_clicks")
)
def check_health(n_clicks):
    if n_clicks is None:
        return "Click button to check API health"
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Error: {str(e)}"

@app.callback(
    Output("liquidity-output", "children"),
    Input("liquidity-button", "n_clicks"),
    [
        State("input-cash", "value"),
        State("input-treasury", "value"),
        State("input-deposit-growth", "value"),
        State("input-funding-cost", "value"),
        State("input-banks", "value"),
        State("input-amounts", "value")
    ]
)
def predict_liquidity(n_clicks, cash, treasury, deposit_growth, funding_cost, banks, amounts):
    if n_clicks is None:
        return "Enter values and click button to predict"
    try:
        payload = {
            "13_CASH": cash,
            "Treasury_bills": treasury,
            "Labels_Liquid": 3000000,
            "Curr_Deposit": 10000000,
            "Fixed_Deposit": 8000000,
            "General_Savings": 9000000,
            "Borrowing_Borrow": 1000000,
            "Balance_Interbank": 2000000,
            "Warehouse_Flag": 1000000,
            "earnings_Capital": 10000000,
            "earnings_Gross_Loans": 7100000,
            "Commercial_Flag": 500000,
            "Liabilities_Income": 11500000,
            "Deposit_Growth_Rate": deposit_growth,
            "Funding_Cost_Change_Proxy": funding_cost,
            "Institution_Models": banks,
            "earnings_amounts": amounts
        }
        response = requests.post(f"{API_BASE_URL}/liquidity/predict", json=payload)
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Error: {str(e)}"

@app.callback(
    Output("metrics-output", "children"),
    Input("metrics-button", "n_clicks")
)
def fetch_metrics(n_clicks):
    if n_clicks is None:
        return "Click button to fetch metrics"
    try:
        response = requests.get(f"{API_BASE_URL}/market/metrics/@CL#C")
        response.raise_for_status()
        return str(response.json())
    except Exception as e:
        return f"Error: {str(e)}"

@app.callback(
    [Output("feature-importance-plot", "figure"),
     Output("mc-var-plot", "figure")],
    Input("visualizations-button", "n_clicks")
)
def fetch_visualizations(n_clicks):
    if n_clicks is None:
        return {}, {}
    try:
        response = requests.get(f"{API_BASE_URL}/market/visualizations/@CL#C")
        response.raise_for_status()
        data = response.json()
        return data["feature_importance"], data["mc_var_distribution"]
    except Exception as e:
        return {"data": [], "layout": {"title": {"text": f"Error: {str(e)}"}}}, \
               {"data": [], "layout": {"title": {"text": f"Error: {str(e)}"}}}

@app.callback(
    [Output("report-download", "href"),
     Output("report-output", "children")],
    Input("report-download", "n_clicks")
)
def download_report(n_clicks):
    if n_clicks is None:
        return "", "Click link to download report"
    try:
        response = requests.get(f"{API_BASE_URL}/market/report/@CL#C")
        response.raise_for_status()
        # Encode PDF content as base64 for download link
        encoded = base64.b64encode(response.content).decode()
        href = f"data:application/pdf;base64,{encoded}"
        return href, "Report downloaded successfully"
    except Exception as e:
        return "", f"Error: {str(e)}"

# Run the app
if __name__ == "__main__":
    app.run(debug=True, port=8050)