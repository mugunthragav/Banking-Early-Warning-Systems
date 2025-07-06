import pandas as pd
import numpy as np


class LiquidityCalculator:
    def __init__(self, df):
        self.df = df
        # Basel III weights (simplified)
        self.lcr_outflow_weights = {
            'stable_deposits': 0.05,  # 5% outflow rate
            'wholesale_funding': 1.0,  # 100% outflow rate
            'other_outflows': 0.5  # 50% outflow rate
        }
        self.nsfr_asf_weights = {
            'stable_funding': 0.9,  # 90% ASF
            'capital': 1.0  # 100% ASF
        }
        self.nsfr_rsf_weights = {
            'required_funding': 0.5,  # 50% RSF
            'rwa': 0.85  # 85% RSF
        }

    def calculate_lcr(self, date, account_id=None, division=None):
        # Point-in-time calculation for the given date over 30-day horizon
        start_date = pd.to_datetime(date)
        end_date = start_date + pd.Timedelta(days=30)
        df_filtered = self.df.loc[start_date:end_date]
        if df_filtered.empty:
            return np.nan

        # Filter by account_id or division if specified
        if account_id is not None:
            df_filtered = df_filtered[df_filtered['account_id'] == account_id]
        elif division is not None:
            df_filtered = df_filtered[df_filtered['division'] == division]

        if df_filtered.empty:
            return np.nan

        latest_date = df_filtered.index.max()
        df_latest = df_filtered.loc[latest_date:latest_date]

        # Debug: Print available columns
        print(f"Available columns in df_latest: {df_latest.columns.tolist()}")

        # Calculate outflows with error handling for missing columns
        outflows = 0
        for col, weight in self.lcr_outflow_weights.items():
            value = df_latest.get(col, pd.Series([0])).sum()  # Default to Series with 0 if column missing
            outflows += value * weight

        # Calculate inflows
        inflows = df_latest.get('inflows', pd.Series([0])).sum() * 0.75  # Cap inflows at 75%
        net_cash_outflows = max(outflows - inflows, outflows * 0.05)  # Minimum 5% of outflows
        lcr = (df_latest.get('hqla_value', pd.Series([0])).sum() / net_cash_outflows if net_cash_outflows > 0 else 999)
        return min(lcr, 999)

    def calculate_nsfr(self, date, account_id=None, division=None):
        # Point-in-time calculation for the given date over 1-year horizon
        start_date = pd.to_datetime(date)
        end_date = start_date + pd.Timedelta(days=365)
        df_filtered = self.df.loc[start_date:end_date]
        if df_filtered.empty:
            return np.nan

        # Filter by account_id or division if specified
        if account_id is not None:
            df_filtered = df_filtered[df_filtered['account_id'] == account_id]
        elif division is not None:
            df_filtered = df_filtered[df_filtered['division'] == division]

        if df_filtered.empty:
            return np.nan

        latest_date = df_filtered.index.max()
        df_latest = df_filtered.loc[latest_date:latest_date]

        # Calculate ASF with error handling for missing columns
        asf = sum(df_latest.get(col, pd.Series([0])).sum() * weight for col, weight in self.nsfr_asf_weights.items())
        # Calculate RSF with error handling for missing columns
        rsf = sum(df_latest.get(col, pd.Series([0])).sum() * weight for col, weight in self.nsfr_rsf_weights.items())
        nsfr = (asf / rsf if rsf > 0 else 999)
        return min(nsfr, 999)