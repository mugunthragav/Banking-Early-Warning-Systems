# src/data/load_data.py
import os
import pandas as pd

def load_contracts(symbol, freq='minutely'):
    if freq not in ['minutely', 'daily']:
        raise ValueError("freq must be 'minutely' or 'daily'")
    data_path = os.path.join(os.path.dirname(__file__), '../../data', freq, f'{symbol}.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No data found for {symbol} in {freq}")
    return pd.read_csv(data_path)