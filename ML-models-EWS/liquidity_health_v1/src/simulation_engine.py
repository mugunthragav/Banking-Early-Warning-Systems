import sys
import os
import pandas as pd
from src.core.lcr_models import LCRModels
from src.core.nsfr_models import NSFRModels
import numpy as np
from datetime import timedelta

class SimulationEngine:
    def __init__(self, data_path):
        self.lcr_model = LCRModels(data_path)
        self.nsfr_model = NSFRModels(data_path)
        self.df = pd.read_csv(data_path, parse_dates=['date'])
        self.df.set_index('date', inplace=True)
        print(f"Data index type: {type(self.df.index)}, sample index: {self.df.index[:5]}")

    def simulate_lcr(self, hqla_adjustment=0, outflow_adjustment=0, inflow_adjustment=0, days=30):
        df_sim = self.df.copy()
        df_sim['hqla_value'] += hqla_adjustment
        df_sim['outflows'] += outflow_adjustment
        df_sim['inflows'] += inflow_adjustment
        # Update LCR calculation with adjusted data
        avg_outflow = df_sim['outflows'].mean()
        df_sim['net_outflow'] = df_sim['outflows'].rolling(window=30, min_periods=30).sum() - df_sim['inflows'].rolling(window=30, min_periods=30).sum()
        df_sim['net_outflow'] = np.where(df_sim['net_outflow'] <= 0, avg_outflow * 0.05, df_sim['net_outflow'])
        df_sim['lcr'] = df_sim['hqla_value'] / df_sim['net_outflow'] * 100
        df_sim['lcr'] = np.where(df_sim['lcr'] > 1000, 1000, df_sim['lcr'])
        df_sim = df_sim.dropna(subset=['lcr'])
        # Use LCRModels to forecast the adjusted LCR
        last_date = pd.to_datetime(self.df.index[-1])
        # Update LCRModels with adjusted data
        self.lcr_model.df = df_sim  # Temporarily update the model's data
        lcr_forecast = self.lcr_model.forecast_lcr(steps=days)
        print(f"LCR Forecast for {days} days: {lcr_forecast.values}")  # Debug print
        return lcr_forecast

    def simulate_nsfr(self, stable_funding_adjustment=0, required_funding_adjustment=0, days=365):
        df_sim = self.df.copy()
        df_sim['stable_funding'] += stable_funding_adjustment
        df_sim['required_funding'] += required_funding_adjustment
        # Update NSFR calculation with adjusted data
        df_sim['nsfr'] = np.where(df_sim['required_funding'] != 0,
                                 df_sim['stable_funding'].rolling(window=365, min_periods=365).mean() /
                                 df_sim['required_funding'].rolling(window=365, min_periods=365).mean() * 100,
                                 np.nan)
        df_sim = df_sim.dropna(subset=['nsfr'])
        # Use NSFRModels to forecast the adjusted NSFR
        last_date = pd.to_datetime(self.df.index[-1])
        # Update NSFRModels with adjusted data
        self.nsfr_model.df = df_sim  # Temporarily update the model's data
        nsfr_forecast = self.nsfr_model.forecast_nsfr(steps=days)
        print(f"NSFR Forecast for {days} days: {nsfr_forecast.values}")  # Debug print
        return nsfr_forecast

    def ad_hoc_scenario(self, scenario_type, adjustment, days=30):
        if scenario_type == "merger":
            return self.simulate_nsfr(stable_funding_adjustment=adjustment, required_funding_adjustment=adjustment, days=days)
        elif scenario_type == "sale":
            return self.simulate_lcr(hqla_adjustment=adjustment, days=days)
        return None

    def run_stress_simulation(self, hqla_adjustment=0, outflow_adjustment=0, inflow_adjustment=0, stable_funding_adjustment=0, required_funding_adjustment=0):
        simulations = {
            30: self.simulate_lcr(hqla_adjustment, outflow_adjustment, inflow_adjustment, days=30),
            90: self.simulate_lcr(hqla_adjustment, outflow_adjustment, inflow_adjustment, days=90),
            180: self.simulate_lcr(hqla_adjustment, outflow_adjustment, inflow_adjustment, days=180),
            365: self.simulate_nsfr(stable_funding_adjustment, required_funding_adjustment, days=365)
        }
        return simulations