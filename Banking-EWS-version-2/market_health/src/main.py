import argparse
import pandas as pd
import numpy as np
import os
import sys
import yaml
import logging
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from scipy import stats
import time
import matplotlib
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import json
from numpy.linalg import cholesky, LinAlgError

matplotlib.use('Agg')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_risk.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import modules
from src.data.load_data import load_contracts
from src.data.synthetic_data import generate_synthetic_data
from src.features.util import getDailyVol
from src.labeling.filters import cusum
from src.labeling.labelling import getEvents, getBins
from src.models.active_signals import avgActiveSignals
from src.utils.stats import psr, dsr
from src.visualization.plots import plot_returns
from report import generate_report

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open(os.path.join(project_root, 'config.yaml'), 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise

def ewma_covariance(returns_df, lambda_factor=0.94):
     """
     Calculates EWMA covariance matrix from a DataFrame of returns.
     Assumes returns_df has shape (time_steps, n_assets).
     Returns the latest covariance matrix.
     """
     T, n_assets = returns_df.shape
     if T == 0:
         return np.zeros((n_assets, n_assets))

     covariance_matrix = np.zeros((T, n_assets, n_assets))

     initial_window = min(50, T)
     if initial_window < n_assets + 1:
          initial_window = T
          if initial_window < n_assets + 1 and T > 0:
               logger.warning(f"Not enough data ({T}) to reliably estimate initial covariance matrix for {n_assets} assets. Matrix might be singular.")

     if initial_window > 0:
         # Ensure there's data to calculate initial covariance
         if T >= initial_window:
             initial_cov = np.cov(returns_df.iloc[:initial_window], rowvar=False)
             covariance_matrix[initial_window - 1] = initial_cov
         else: # Should not happen if initial_window <= T, but as safeguard
              return np.zeros((n_assets, n_assets))
     else:
         return np.zeros((n_assets, n_assets))


     returns_values = returns_df.values
     returns_outer_product = np.array([np.outer(returns_values[t], returns_values[t]) for t in range(T)])

     for t in range(initial_window, T):
         covariance_matrix[t] = lambda_factor * covariance_matrix[t-1] + (1 - lambda_factor) * returns_outer_product[t]

     return covariance_matrix[-1]

def calculate_var_monte_carlo(prices_day, confidence_levels, horizons, simulations):
    """Calculate Monte Carlo VaR with revenue and capital impact for a single asset."""
    try:
        historical_daily_returns = prices_day.pct_change().dropna()

        if historical_daily_returns.empty:
             logger.warning("Historical daily returns are empty for MC VaR.")
             return {}

        latest_price = prices_day.iloc[-1]

        mean_daily_return = historical_daily_returns.mean()
        std_daily_return = historical_daily_returns.std()

        # Avoid issues if std_daily_return is zero or NaN
        if np.isnan(std_daily_return) or std_daily_return <= 0:
            logger.warning("Standard deviation of historical daily returns is zero or NaN for MC VaR.")
            results = {}
            for conf in confidence_levels:
                 for horizon in horizons:
                      results[f'mc_var_{conf}_{horizon}d'] = {
                          'var': 0.0, 'revenue_impact': 0.0,
                          'capital_impact': 0.0, 'liquidity_impact': 0.0
                      }
            return results


        df = 4 # Degrees of freedom for t-distribution

        results = {}
        capital_reserve = latest_price * 0.1
        liquidity_buffer = latest_price * 0.05

        for conf in confidence_levels:
            for horizon in horizons:
                simulated_final_prices = np.zeros(simulations)

                # Simulate returns for each day in the horizon, then compound
                # Use the estimated mean and std dev for the t-distribution
                daily_shocks = stats.t.rvs(df, loc=mean_daily_return, scale=std_daily_return, size=(simulations, horizon))

                # Compound returns for each simulation path
                cumulative_returns = np.prod(1 + daily_shocks, axis=1) - 1

                # Calculate the final price for each path
                simulated_final_prices = latest_price * (1 + cumulative_returns)

                losses = latest_price - simulated_final_prices
                losses = np.maximum(0, losses)

                # Ensure there are losses to calculate percentile on
                if len(losses) == 0:
                     var = 0.0
                else:
                     var = np.percentile(losses, conf * 100)

                revenue_impact = var * 0.2
                capital_impact = max(0, var - capital_reserve)
                liquidity_impact = max(0, var - liquidity_buffer)

                results[f'mc_var_{conf}_{horizon}d'] = {
                    'var': float(var),
                    'revenue_impact': float(revenue_impact),
                    'capital_impact': float(capital_impact),
                    'liquidity_impact': float(liquidity_impact)
                }
        return results
    except Exception as e:
        logger.error(f"Monte Carlo VaR calculation failed for price data with shape {prices_day.shape}: {e}")
        return {}

def calculate_historical_var(returns, prices, confidence_levels, horizons):
    """Calculate Historical VaR with revenue and capital impact."""
    try:
        results = {}
        capital_reserve = prices.iloc[-1] * 0.1
        liquidity_buffer = prices.iloc[-1] * 0.05
        for conf in confidence_levels:
            for horizon in horizons:
                # Need enough data for the rolling window
                if len(returns) < horizon:
                     logger.warning(f"Not enough return data ({len(returns)}) for Historical VaR with horizon {horizon}.")
                     results[f'hist_var_{conf}_{horizon}d'] = {}
                     continue

                rolling_returns = returns.rolling(horizon).apply(lambda x: np.prod(1 + x) - 1, raw=True).dropna() # Use raw=True for efficiency

                if rolling_returns.empty:
                     results[f'hist_var_{conf}_{horizon}d'] = {}
                     continue

                # VaR percentile for Historical simulation is typically at 1-conf quantile of returns
                # e.g., for 95% VaR, 5th percentile of returns
                var_return = np.percentile(rolling_returns, (1 - conf) * 100)

                # VaR value as a positive loss
                var_value = -var_return * prices.iloc[-1] # Loss is positive when return is negative

                revenue_impact = var_value * 0.2
                capital_impact = max(0, var_value - capital_reserve)
                liquidity_impact = max(0, var_value - liquidity_buffer)
                results[f'hist_var_{conf}_{horizon}d'] = {
                    'var': float(var_value),
                    'revenue_impact': float(revenue_impact),
                    'capital_impact': float(capital_impact),
                    'liquidity_impact': float(liquidity_impact)
                }
        return results
    except Exception as e:
        logger.error(f"Historical VaR calculation failed: {e}")
        return {}

def apply_stress_test(prices_day, scenarios):
    """Apply stress tests with revenue, capital, and liquidity impacts by shocking latest price."""
    try:
        stress_results = {}
        if prices_day.empty:
            logger.warning("Price data is empty for stress test.")
            return {}

        latest_price = prices_day.iloc[-1]
        capital_reserve = latest_price * 0.1
        liquidity_buffer = latest_price * 0.05

        for scenario, params in scenarios.items():
            factor = params.get('price_factor', 1.0)

            stressed_price = latest_price * factor

            drawdown = (1 - stressed_price / latest_price) * 100 if latest_price != 0 else 0
            loss = max(0, latest_price - stressed_price)

            revenue_impact = loss * 0.2
            capital_impact = max(0, loss - capital_reserve)
            liquidity_impact = max(0, loss - liquidity_buffer)

            margin_shortfall = max(0, drawdown * latest_price * 0.1 / 100)


            stress_results[scenario] = {
                'final_value': float(stressed_price),
                'loss': float(loss),
                'drawdown': float(drawdown),
                'margin_shortfall': float(margin_shortfall),
                'revenue_impact': float(revenue_impact),
                'capital_impact': float(capital_impact),
                'liquidity_impact': float(liquidity_impact)
            }
        return stress_results
    except Exception as e:
        logger.error(f"Stress test failed for prices_day with shape {prices_day.shape}: {e}")
        return {}

def calculate_portfolio_var(symbols, weights, prices_dict, confidence_level, horizon, simulations, lambda_factor=0.94):
    """Calculate portfolio VaR using Monte Carlo with correlation."""
    try:
        n_assets = len(symbols)
        if n_assets == 0 or n_assets != len(weights):
            logger.error("Invalid number of symbols or weights for portfolio VaR.")
            return np.nan

        latest_prices = np.array([prices_dict[s].iloc[-1] if s in prices_dict and not prices_dict[s].empty else np.nan for s in symbols])
        if np.isnan(latest_prices).any():
             logger.warning("Missing or empty price data for some assets for portfolio VaR.")
             return np.nan

        initial_portfolio_value = np.sum(latest_prices * weights)

        historical_returns_list = []
        for s in symbols:
             if s in prices_dict and not prices_dict[s].empty:
                 historical_returns_list.append(prices_dict[s].pct_change().dropna().rename(s))
             else:
                  # If data for an asset is missing, create an empty Series to keep the column slot
                  logger.warning(f"Historical price data for {s} not available for portfolio returns.")
                  historical_returns_list.append(pd.Series(dtype=float).rename(s))


        if not historical_returns_list:
             logger.warning("No asset return data collected for portfolio VaR.")
             return np.nan

        historical_returns_df = pd.concat(historical_returns_list, axis=1) # Do not dropna here yet

        # Drop rows with any NaN after concatenation (only keep dates where all assets have returns)
        historical_returns_df = historical_returns_df.dropna()

        if historical_returns_df.empty or len(historical_returns_df) < 2: # Need at least 2 rows for cov
            logger.warning("Not enough common historical return data after dropping NaNs for portfolio VaR calculation.")
            return np.nan

        cov_matrix = ewma_covariance(historical_returns_df, lambda_factor)

        jitter = 0
        max_jitter_attempts = 10
        for attempt in range(max_jitter_attempts):
            try:
                if jitter > 0:
                     cov_matrix_jittered = cov_matrix + np.eye(n_assets) * jitter * np.mean(np.diag(cov_matrix))
                else:
                     cov_matrix_jittered = cov_matrix

                chol_matrix = cholesky(cov_matrix_jittered)
                break
            except LinAlgError:
                if attempt == max_jitter_attempts - 1:
                    logger.error(f"Cholesky decomposition failed after {max_jitter_attempts} attempts with jitter.")
                    return np.nan
                jitter = max(jitter * 2, 1e-6)
                logger.warning(f"Covariance matrix not positive definite, attempting with jitter: {jitter}")


        mean_daily_returns = historical_returns_df.mean().values

        final_portfolio_values = np.zeros(simulations)

        dt = 1

        for i in range(simulations):
            current_prices = latest_prices.copy()

            for d in range(horizon):
                z = np.random.normal(size=n_assets)
                correlated_shocks = chol_matrix @ z

                # Log return step: ln(P_t/P_{t-1}) = (mu - 0.5*sigma^2)*dt + sqrt(Cov) @ epsilon_vector * sqrt(dt)
                # correlated_shocks is sqrt(Cov) @ epsilon_vector
                log_returns_step = (mean_daily_returns - 0.5 * np.diag(cov_matrix_jittered)) * dt + correlated_shocks * np.sqrt(dt)

                step_returns = np.exp(log_returns_step) - 1
                current_prices = current_prices * (1 + step_returns)

            final_portfolio_values[i] = np.sum(current_prices * weights)

        losses = initial_portfolio_value - final_portfolio_values
        losses = np.maximum(0, losses)

        if len(losses) == 0:
             var = 0.0
        else:
             var = np.percentile(losses, confidence_level * 100)


        return float(var)

    except Exception as e:
        logger.error(f"Portfolio VaR calculation failed: {e}")
        return np.nan

def backtest_var(prices_day, var_dict, config_horizons):
    """
    Backtest VaR calculations comparing historical forecasts to actual P&L.
    Uses config_horizons to get the list of horizons to test.
    Compares the *current* VaR value to historical rolling losses.
    NOTE: This is NOT a standard backtest of historical forecast accuracy.
    """
    try:
        backtest_results = {}

        horizons_to_test = config_horizons

        if prices_day.empty:
            logger.warning("Price data is empty for backtesting.")
            for var_key in var_dict.keys():
                 backtest_results[var_key] = {'exceedances': 0, 'coverage': np.nan}
            return backtest_results

        max_horizon_needed = max(horizons_to_test) if horizons_to_test else 1
        min_data_points = max_horizon_needed + 1

        if len(prices_day) < min_data_points:
             logger.warning(f"Not enough data ({len(prices_day)}) for backtesting with max horizon ({max_horizon_needed}). Need at least {min_data_points}.")
             for var_key in var_dict.keys():
                  # Assign NaN coverage for insufficient data
                  backtest_results[var_key] = {'exceedances': 0, 'coverage': np.nan}
             return backtest_results

        for var_key, var_info in var_dict.items():
            try:
                parts = var_key.split('_')
                if len(parts) < 3: continue
                try:
                    horizon_str = parts[-1]
                    horizon = int(horizon_str[:-1])
                except ValueError:
                    continue

                if horizon not in horizons_to_test:
                    continue

                var_value = var_info.get('var', 0)

                # Calculate actual rolling losses over the historical period
                if len(prices_day) >= horizon:
                     rolling_losses = pd.Series(index=prices_day.index[horizon:])
                     # Ensure rolling loss calculation has data
                     if not prices_day.iloc[horizon:].empty:
                          for i in range(horizon, len(prices_day)):
                               rolling_losses.iloc[i - horizon] = prices_day.iloc[i - horizon] - prices_day.iloc[i]
                     else:
                          logger.warning(f"Insufficient data for rolling loss calculation for horizon {horizon}.")
                          exceedances = 0
                          num_periods_tested = 0
                          coverage = np.nan

                else:
                     logger.warning(f"Not enough data ({len(prices_day)}) for rolling loss calculation for horizon {horizon}.")
                     exceedances = 0
                     num_periods_tested = 0
                     coverage = np.nan


                if 'rolling_losses' in locals() and not rolling_losses.empty:
                    rolling_losses = rolling_losses.dropna()
                    exceedances = (rolling_losses > var_value).sum()
                    num_periods_tested = len(rolling_losses)
                    coverage = 1 - exceedances / num_periods_tested if num_periods_tested > 0 else np.nan
                else:
                    # Case where rolling_losses couldn't be calculated or is empty
                    exceedances = 0
                    num_periods_tested = 0
                    coverage = np.nan


                backtest_results[var_key] = {
                    'exceedances': int(exceedances),
                    'coverage': float(coverage) if not np.isnan(coverage) else None
                }
            except Exception as e:
                 logger.error(f"Backtest calculation failed for {var_key}: {e}")
                 backtest_results[var_key] = {'exceedances': 0, 'coverage': np.nan}


        return backtest_results
    except Exception as e:
        logger.error(f"VaR backtesting failed: {e}")
        return {}


def run_strategy(symbols, config, use_synthetic=False, all_results=None):
    """Run trading strategy for given symbols."""
    if all_results is None:
        all_results = {}
    try:
        # Load data for the current batch of symbols
        batch_data_dict = {}
        for s in symbols:
            try:
                if use_synthetic:
                     # Use n_rows_min from config if needed for synthetic data length
                    batch_data_dict[s] = {'minutely': generate_synthetic_data([s], n_rows_min=config['var']['simulations'], n_rows_day=2520)[s]['minutely'],
                                           'daily': generate_synthetic_data([s], n_rows_min=config['var']['simulations'], n_rows_day=2520)[s]['daily']}
                else:
                    batch_data_dict[s] = {'minutely': load_contracts(s, freq='minutely'),
                                           'daily': load_contracts(s, freq='daily')}
            except (FileNotFoundError, Exception) as e:
                logger.error(f"Failed to load/generate data for {s}: {e}. Skipping symbol.")
                # Store empty data for the symbol to avoid breaking subsequent loops
                batch_data_dict[s] = {'minutely': pd.DataFrame(), 'daily': pd.DataFrame()}


        # Process each symbol in the batch
        for symbol, data in batch_data_dict.items():
            logger.info(f"Processing symbol: {symbol}")

            # --- Initialize ML/Strategy related variables to default skipped values ---
            accuracy = np.nan
            sharpe = np.nan
            psr_val = np.nan
            dsr_val = np.nan
            mdi_imp = pd.Series().to_dict()
            returns_for_plot = pd.Series() # Initialize returns for plotting

            minutely_processed_successfully = False # Flag to track if minutely processing was successful

            # --- ML/Strategy Calculations (using minutely data) ---
            if 'minutely' in data and not data['minutely'].empty:
                try:
                    minutely_data = data['minutely'].copy() # Work on a copy
                    minutely_data['date'] = pd.to_datetime(minutely_data['date'], format='%Y-%m-%d %H:%M:%S')
                    bars = minutely_data.groupby(pd.Grouper(key='date', freq='1min')).agg(
                        {'open_p': 'first', 'close_p': 'last', 'volume': 'sum'}).dropna()
                    prices_min = bars['close_p']

                    if not prices_min.empty:
                         # Apply ffill directly, no need for fillna(method='ffill')
                         prices_min_filled = prices_min.ffill().fillna(prices_min.mean() if not prices_min.dropna().empty else 0)

                         # Ensure volatility calculation handles potential NaNs or empty series
                         vol_min = getDailyVol(prices_min.dropna()) # Calculate vol on non-NaN prices
                         # Reindex vol_min to prices_min_filled index, then ffill, then fill remaining NaNs
                         vol_min = vol_min.reindex(prices_min_filled.index).ffill()
                         # Fill any remaining NaNs (e.g., at the very start) with the mean or a default
                         vol_min = vol_min.fillna(vol_min.mean() if not vol_min.dropna().empty else 0)


                         tEvents = cusum(prices_min_filled, h=0.02 * (1 + vol_min.mean()))
                         if not tEvents.empty:
                              events = getEvents(close=prices_min_filled, tEvents=tEvents, ptSl=[2, 4], trgt=vol_min * 2.0, minRet=0.01)
                              if not events.empty:
                                   labels = getBins(events, prices_min_filled)

                                   # Ensure X and y are aligned and have data/classes
                                   X = pd.DataFrame({
                                       'Returns': prices_min_filled.pct_change().fillna(0),
                                       'Lagged_Returns': prices_min_filled.pct_change().shift(1).fillna(0),
                                       'Volatility': vol_min, # Already aligned and filled
                                       'Momentum': prices_min_filled.pct_change(periods=5).rolling(5).mean().fillna(0),
                                       'Range': (prices_min_filled.rolling(10).max() - prices_min_filled.rolling(10).min()) / prices_min_filled.replace(0, np.nan).ffill().fillna(0),
                                       'Volume_Change': minutely_data.set_index('date').reindex(prices_min_filled.index)['volume'].pct_change().fillna(0).fillna(0) # Fill remaining NaNs after reindex
                                   })
                                   X = X.dropna() # Drop any rows that still have NaNs after filling attempts

                                   # Ensure y is reindexed to X index and handle potential missing labels
                                   y = labels['bin'].reindex(X.index)
                                   # Drop rows from X where y is NaN (corresponds to dates with no labels)
                                   X = X.loc[y.dropna().index]
                                   y = y.dropna()

                                   # --- ML Training and Signal Generation ---
                                   if not X.empty and len(y) == len(X) and len(np.unique(y)) >= 2:
                                        try:
                                            smote = SMOTE(random_state=42, sampling_strategy=0.5)
                                            X_res, y_res = smote.fit_resample(X, y)
                                            logger.info(f"After SMOTE: X rows = {len(X_res)}, y classes = {np.unique(y_res, return_counts=True)}")

                                            # Ensure enough data/classes after SMOTE for splitting
                                            if len(X_res) >= 2 and len(np.unique(y_res)) >= 2:
                                                 X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

                                                 # Ensure train/test sets have enough classes
                                                 if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                                                     param_grid = {'max_iter': [50], 'max_depth': [10], 'min_samples_leaf': [20]}
                                                     clf = GridSearchCV(HistGradientBoostingClassifier(random_state=42), param_grid, cv=min(3, len(X_train)//2), n_jobs=2) # Adjust cv if data is small
                                                     clf.fit(X_train, y_train)
                                                     perm_importance = permutation_importance(clf.best_estimator_, X_test, y_test, n_repeats=5, random_state=42, n_jobs=2)
                                                     mdi_imp = pd.Series(perm_importance.importances_mean, index=X.columns).sort_values(ascending=False)
                                                     accuracy = clf.score(X_test, y_test)

                                                     signals = pd.DataFrame({'signal': clf.predict_proba(X)[:, 1]}, index=X.index)
                                                     active_signals = avgActiveSignals(signals)

                                                     # Calculate returns based on signals
                                                     aligned_returns = X['Returns'].reindex(active_signals.index).fillna(0)
                                                     returns_for_plot = active_signals * aligned_returns # Assign to returns_for_plot

                                                     # Avoid division by zero if returns std is 0
                                                     sharpe = returns_for_plot.mean() / returns_for_plot.std() * np.sqrt(252 * 24 * 60 / 10080) if returns_for_plot.std() != 0 else np.nan
                                                     psr_val = psr(sharpe, len(returns_for_plot), returns_for_plot.skew(), returns_for_plot.kurt()) if not np.isnan(sharpe) else np.nan
                                                     dsr_val = dsr(sharpe, 0.1, 100, len(returns_for_plot), returns_for_plot.skew(), returns_for_plot.kurt()) if not np.isnan(sharpe) else np.nan

                                                     minutely_processed_successful = True # Mark as successful if we reached here

                                                 else:
                                                      logger.warning(f"Not enough classes in train or test set after split for {symbol}. Skipping ML training.")
                                            else:
                                                 logger.warning(f"Not enough data or classes after SMOTE for {symbol}. Skipping ML training.")
                                        except Exception as e:
                                             logger.error(f"ML training or signal generation failed for {symbol}: {e}. Skipping ML parts.")
                                   else:
                                        logger.warning(f"Not enough valid features/labels or classes for {symbol}. Skipping ML training.")
                              else:
                                   logger.warning(f"No events generated for {symbol}. Skipping ML parts.")
                         else:
                              logger.warning(f"No tEvents generated for {symbol}. Skipping ML parts.")
                    else:
                         logger.warning(f"Minutely prices for {symbol} are empty after processing. Skipping ML parts.")
                except Exception as e:
                    logger.error(f"Minutely data processing or ML feature/labeling failed for {symbol}: {e}. Skipping ML parts.")


            # --- Risk Calculations (using daily data) ---
            # Initialize risk related variables to default skipped values
            monte_carlo_var = {}
            historical_var = {}
            # portfolio_var is calculated per batch later, initialize per symbol result here
            symbol_portfolio_var = np.nan # Use a temporary name to avoid overwriting batch portfolio_var
            backtest_results = {}
            stress_results = {}

            daily_processed_successfully = False # Flag to track if daily processing was successful

            if 'daily' in data and not data['daily'].empty:
                 try:
                      daily_data = data['daily'].copy() # Work on a copy
                      daily_data['date'] = pd.to_datetime(daily_data['date'], format='%Y-%m-%d')
                      # Ensure daily prices are not empty or all NaN
                      prices_day = pd.Series(daily_data['close_p'].values, index=daily_data['date']).dropna()

                      # Ensure prices_day has enough data points after dropping NaNs for min horizon
                      min_horizon_needed = min(config['var']['horizons']) if config['var']['horizons'] else 1
                      if len(prices_day) < min_horizon_needed + 1:
                           logger.warning(f"Not enough daily price data ({len(prices_day)}) for basic risk calculations for {symbol}. Need at least {min_horizon_needed + 1} points.")
                      else:
                           # getDailyVol is intended for daily prices here, calculate on non-NaN prices
                           vol_day = getDailyVol(prices_day.dropna())
                           vol_day = vol_day.reindex(prices_day.index).ffill().fillna(vol_day.mean() if not vol_day.dropna().empty else 0) # Ensure aligned and filled


                           monte_carlo_var = calculate_var_monte_carlo(prices_day, config['var']['confidence_levels'], config['var']['horizons'], config['var']['simulations'])
                           historical_var = calculate_historical_var(prices_day.pct_change().dropna(), prices_day, config['var']['confidence_levels'], config['var']['horizons'])

                           # Pass horizons from config to backtest_var
                           backtest_results = backtest_var(prices_day, {**monte_carlo_var, **historical_var}, config['var']['horizons'])
                           # Remove vol_day argument from apply_stress_test call
                           stress_results = apply_stress_test(prices_day, config['stress_scenarios'])

                           daily_processed_successful = True # Mark as successful if we reached here

                 except Exception as e:
                      logger.error(f"Daily data processing or risk calculation failed for {symbol}: {e}. Skipping risk calculations.")
            else:
                 logger.warning(f"Daily data for {symbol} not found or empty. Skipping risk calculations.")

            # --- Populate results for the current symbol ---
            all_results[symbol] = {
                'accuracy': accuracy,
                'sharpe': sharpe,
                'psr': psr_val,
                'dsr': dsr_val,
                'monte_carlo_var': monte_carlo_var,
                'historical_var': historical_var,
                'portfolio_var': np.nan, # Initialize here, will be updated per batch later
                'backtest_results': backtest_results,
                'stress_results': stress_results,
                'mdi_imp': mdi_imp if isinstance(mdi_imp, dict) else (mdi_imp.to_dict() if hasattr(mdi_imp, 'to_dict') else {}) # Ensure it's a dict
            }

            # Plot returns if ML processing was successful
            if minutely_processed_successful and not returns_for_plot.empty:
                 try:
                      plot_returns(returns_for_plot, symbol)
                 except Exception as e:
                      logger.error(f"Plotting returns failed for {symbol}: {e}")


        # --- Calculate Portfolio VaR for the batch AFTER processing all symbols in the batch ---
        # This requires data for all symbols in the current batch
        batch_symbols_with_daily_data = [s for s in symbols if s in batch_data_dict and 'daily' in batch_data_dict[s] and not batch_data_dict[s]['daily'].empty and not batch_data_dict[s]['daily']['close_p'].dropna().empty]

        if len(batch_symbols_with_daily_data) >= 2: # Need at least 2 assets for portfolio correlation
             try:
                 # Collect prices for assets with valid daily data in the batch
                 portfolio_prices_dict = {
                     s: pd.Series(batch_data_dict[s]['daily']['close_p'].values, index=pd.to_datetime(batch_data_dict[s]['daily']['date'], format='%Y-%m-%d')).dropna()
                     for s in batch_symbols_with_daily_data
                 }
                 portfolio_weights = [1 / len(batch_symbols_with_daily_data)] * len(batch_symbols_with_daily_data)

                 # Ensure minimum data for portfolio VaR calculation across all assets
                 # This min data check is handled inside calculate_portfolio_var

                 batch_portfolio_var = calculate_portfolio_var(batch_symbols_with_daily_data, portfolio_weights, portfolio_prices_dict,
                                                                config['var']['confidence_levels'][0], config['var']['horizons'][0], config['var']['simulations'])

                 # Update the portfolio_var for each symbol in the batch
                 for s in batch_symbols_with_daily_data:
                      if s in all_results: # Ensure symbol exists in results dictionary
                           all_results[s]['portfolio_var'] = batch_portfolio_var

             except Exception as e:
                 logger.error(f"Batch portfolio VaR calculation failed for batch {symbols}: {e}")
        elif len(batch_symbols_with_daily_data) == 1:
             logger.warning(f"Only one asset ({batch_symbols_with_daily_data[0]}) has valid daily data in batch. Portfolio VaR calculation skipped.")
        else:
             logger.warning(f"No assets have valid daily data in batch. Portfolio VaR calculation skipped.")


        # Return results for the symbols in this batch
        return {s: all_results.get(s, {}) for s in symbols} # Return only results for the requested symbols


    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        return all_results # Return partial results if the outer loop fails


def run_batch_strategy(config):
    """Run strategy in batches and save results."""
    try:
        symbols = config['symbols']
        all_results = {}
        batch_size = 5
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            logger.info(f"Processing batch: {batch}")
            # run_strategy processes the batch and updates all_results directly (due to passing all_results)
            # and also returns the results for the symbols in the batch.
            # The update inside run_strategy is enough, but also using the return value is fine.
            run_strategy(batch, config, use_synthetic=False, all_results=all_results)


        os.makedirs('models', exist_ok=True)
        def convert_nan_to_none(obj):
             if isinstance(obj, float) and np.isnan(obj):
                  return None
             if isinstance(obj, dict):
                  return {k: convert_nan_to_none(v) for k, v in obj.items()}
             if isinstance(obj, list):
                  return [convert_nan_to_none(elem) for elem in obj]
             # Handle pandas specific types if necessary, though to_dict usually handles them
             if isinstance(obj, pd.Series):
                 return convert_nan_to_none(obj.to_dict())
             if isinstance(obj, pd.DataFrame):
                 return convert_nan_to_none(obj.to_dict('records')) # Or 'list' or 'dict'
             return obj

        all_results_cleaned = convert_nan_to_none(all_results)


        with open('models/all_commodities_model.json', 'w') as f:
            json.dump(all_results_cleaned, f, indent=4)
        logger.info("Results saved to models/all_commodities_model.json")

        # generate_report expects the raw results dictionary (before NaN conversion for JSON)
        generate_report(all_results, config)
        logger.info("Report generation initiated.")


        return all_results
    except Exception as e:
        logger.error(f"Batch strategy failed: {e}")
        return {}

def schedule_simulations(config):
    """Schedule periodic simulations."""
    try:
        scheduler = BackgroundScheduler()
        interval = config['scheduler']['interval']
        time_str = config['scheduler']['time']
        if interval == 'monthly':
            scheduler.add_job(run_batch_strategy, 'cron', args=[config], day='1', hour=int(time_str.split(':')[0]), minute=int(time_str.split(':')[1]))
        elif interval == 'quarterly':
            scheduler.add_job(run_batch_strategy, 'cron', args=[config], month='1,4,7,10', day='1', hour=int(time_str.split(':')[0]), minute=int(time_str.split(':')[1]))
        scheduler.start()
        logger.info(f"Scheduled {interval} simulations at {time_str}")
        # Keep the main thread alive for the scheduler to run
        try:
             # This is just to keep the script running for the scheduler
             while True:
                  time.sleep(60)
        except (KeyboardInterrupt, SystemExit):
             # Allow graceful exit on Ctrl+C or system exit
             scheduler.shutdown()
             logger.info("Scheduler shut down.")

    except Exception as e:
        logger.error(f"Scheduling failed: {e}")

def main():
    """Main entry point for the application."""
    try:
        config = load_config()
        parser = argparse.ArgumentParser(description='Financial Machine Learning Application')
        parser.add_argument('--mode', choices=['strategy', 'batch_strategy', 'schedule'], required=True, help='Run mode')
        parser.add_argument('--synthetic', action='store_true', help='Use synthetic data')
        args = parser.parse_args()

        if args.mode == 'strategy':
            logger.info("Running in 'strategy' mode for @CL#C.")
            # Note: Portfolio VaR will likely fail or be NaN in this mode as it expects a batch >= 2
            run_strategy(['@CL#C'], config, args.synthetic)
        elif args.mode == 'batch_strategy':
            logger.info("Running in 'batch_strategy' mode.")
            run_batch_strategy(config)
        elif args.mode == 'schedule':
            logger.info("Running in 'schedule' mode.")
            schedule_simulations(config)
    except Exception as e:
        logger.error(f"Application failed: {e}")


if __name__ == '__main__':
    main()