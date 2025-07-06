# src/data/synthetic_data.py
import numpy as np
import pandas as pd
import os


def generate_synthetic_data(symbols, n_rows_min=600000, n_rows_day=2520, start_date='2015-01-01'):
    sector_params = {
        'Equity Index': (0.08, 0.15, 4000),  # (annual_ret, annual_vol, start_price)
        'Interest Rate': (0.03, 0.08, 120),
        'Agriculture': (0.05, 0.20, 500),
        'Energy': (0.06, 0.25, 70),
        'Metals': (0.04, 0.18, 1800),
        'Currencies': (0.02, 0.10, 1.10)
    }

    # Create directories if they don't exist
    os.makedirs('data/minutely', exist_ok=True)
    os.makedirs('data/daily', exist_ok=True)

    for symbol in symbols:
        sector = 'Equity Index' if symbol.split('#')[0] in ['@ES', '@NQ'] else \
            'Interest Rate' if symbol.split('#')[0] in ['@TY', '@GC'] else \
                'Agriculture' if symbol.split('#')[0] in ['@C', '@S'] else \
                    'Energy' if symbol.split('#')[0] in ['@CL', '@NG'] else \
                        'Metals' if symbol.split('#')[0] in ['@SI', '@US'] else \
                            'Currencies' if symbol.split('#')[0] in ['@AD', '@EC'] else 'Equity Index'

        annual_ret, annual_vol, start_price = sector_params[sector]
        minute_ret = annual_ret / (252 * 24 * 60)
        minute_vol = annual_vol / np.sqrt(252 * 24 * 60)
        daily_ret = annual_ret / 252
        daily_vol = annual_vol / np.sqrt(252)

        # Minute data (≈416 days)
        dates_min = pd.date_range(start_date, periods=n_rows_min, freq='min')
        returns_min = np.random.randn(n_rows_min) * minute_vol + minute_ret
        prices_min = start_price * np.exp(np.cumsum(returns_min))
        data_min = pd.DataFrame({
            'date': dates_min,
            'open_p': prices_min * (1 + np.random.randn(n_rows_min) * 0.0001),
            'close_p': prices_min,
            'volume': np.random.randint(100, 1000, n_rows_min)
        })
        data_min.to_csv(f'data/minutely/{symbol}.csv', index=False)

        # Derive daily data from minute data for the available period, then extend
        daily_data = data_min.set_index('date').resample('B').agg({
            'open_p': 'first',
            'close_p': 'last',
            'volume': 'sum'
        }).reset_index()
        # Use all available resampled days (no strict 347 check)
        base_days = len(daily_data)

        # Extend to 10 years (2,520 business days) with synthetic data
        remaining_days = n_rows_day - base_days
        if remaining_days > 0:
            dates_ext = pd.date_range(daily_data['date'].iloc[-1] + pd.offsets.BDay(1), periods=remaining_days,
                                      freq='B')
            last_price = daily_data['close_p'].iloc[-1]
            returns_ext = np.random.randn(remaining_days) * daily_vol + daily_ret
            prices_ext = last_price * np.exp(np.cumsum(returns_ext))
            data_ext = pd.DataFrame({
                'date': dates_ext,
                'open_p': prices_ext * (1 + np.random.randn(remaining_days) * 0.001),
                'close_p': prices_ext,
                'volume': np.random.randint(10000, 100000, remaining_days)
            })
            daily_data = pd.concat([daily_data, data_ext])

        daily_data.to_csv(f'data/daily/{symbol}.csv', index=False)

    return {
        s: {'minutely': pd.read_csv(f'data/minutely/{s}.csv'), 'daily': pd.read_csv(f'data/daily/{s}.csv')}
        for s in symbols
    }