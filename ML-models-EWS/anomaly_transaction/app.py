import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Initialize Dash app
app = dash.Dash(__name__)

# Set paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Load models and scaler
with open(os.path.join(OUTPUT_DIR, 'iso_forest.pkl'), 'rb') as f:
    iso_forest = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, 'autoencoder.pkl'), 'rb') as f:
    autoencoder = pickle.load(f)
with open(os.path.join(OUTPUT_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

# Load data
data = pd.read_csv(os.path.join(OUTPUT_DIR, 'anomaly_results.csv'))
alert_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'alert_data.csv'))
common_anomalies = pd.read_csv(os.path.join(OUTPUT_DIR, 'common_anomalies.csv'))
anomaly_data = common_anomalies  # Use common anomalies as the default
print(f"Anomaly data shape: {anomaly_data.shape}")

# Layout
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard", style={'textAlign': 'center'}),
    html.Div([
        html.H3("Anomaly Overview"),
        html.P(f"Total High-Confidence Anomalies: {len(anomaly_data)}"),
    ]),
    html.Div([
        html.H3("Filters"),
        dcc.Dropdown(
            id='customer-dropdown',
            options=[{'label': cid, 'value': cid} for cid in anomaly_data['CustomerID'].unique()],
            value=anomaly_data['CustomerID'].iloc[0]
        ),
        dcc.Dropdown(
            id='filter-dropdown',
            options=[
                {'label': 'Total Spent', 'value': 'TotalSpent'},
                {'label': 'Transaction Frequency', 'value': 'HistFreq'},
                {'label': 'Unusual Destination', 'value': 'UnusualDestination'},
                {'label': 'Quantity Spike', 'value': 'QuantitySpike'}
            ],
            value='TotalSpent'
        ),
        html.Ul(id='anomaly-list', style={'listStyleType': 'none', 'padding': 0}),
    ]),
    dcc.Graph(id='anomaly-chart'),
    html.Div([
        html.H3("Drill-Down Table"),
        dash_table.DataTable(
            id='drill-down-table',
            columns=[{'name': i, 'id': i} for i in alert_data.columns],
            data=alert_data.to_dict('records'),
            filter_action='native',
            sort_action='native',
            page_size=10
        )
    ])
])

# Callbacks
@app.callback(
    [Output('anomaly-list', 'children'),
     Output('anomaly-chart', 'figure'),
     Output('drill-down-table', 'data')],
    [Input('customer-dropdown', 'value'),
     Input('filter-dropdown', 'value')]
)
def update_output(customer_value, metric_value):
    # Filter data by customer (already common anomalies)
    filtered_data = anomaly_data[anomaly_data['CustomerID'] == customer_value]

    # Anomaly list and chart
    sorted_anomalies = filtered_data.sort_values(by=metric_value, ascending=False).head(10)
    list_items = [html.Li(f"Invoice: {row['InvoiceNo']}, {metric_value}: {row[metric_value]:.2f}, Risk: {row['RiskScore']:.2f}") for _, row in sorted_anomalies.iterrows()]
    fig = px.bar(sorted_anomalies, x='InvoiceNo', y=metric_value, color='RiskScore', title=f'Top 10 Anomalies by {metric_value} for Customer {customer_value}')

    # Drill-down table
    drill_down_data = alert_data[alert_data['CustomerID'] == customer_value].to_dict('records')

    return list_items, fig, drill_down_data

# Run app
if __name__ == '__main__':
    app.run(debug=True)