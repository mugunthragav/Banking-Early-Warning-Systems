import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_expected_loss(data, pd, lgd_dt, lgd_svr, ead):
    """Calculate Expected Loss (EL = PD * LGD * EAD)."""
    try:
        # Combine predictions
        result = data.copy()
        result['PD'] = pd
        result['LGD_DT'] = lgd_dt
        result['LGD_SVR'] = lgd_svr
        result['EAD'] = ead

        # Calculate EL for both LGD models
        result['EL_DT'] = result['PD'] * result['LGD_DT'] * result['EAD']
        result['EL_SVR'] = result['PD'] * result['LGD_SVR'] * result['EAD']

        # Summarize
        summary = result[['funded_amnt', 'PD', 'LGD_DT', 'LGD_SVR', 'EAD', 'EL_DT', 'EL_SVR']].describe()
        portfolio_el_dt = result['EL_DT'].sum()
        portfolio_el_svr = result['EL_SVR'].sum()
        portfolio_funded = result['funded_amnt'].sum()
        el_ratio_dt = portfolio_el_dt / portfolio_funded
        el_ratio_svr = portfolio_el_svr / portfolio_funded

        logging.info(f"Portfolio EL (Decision Tree): {portfolio_el_dt:.2f}")
        logging.info(f"Portfolio EL (SVR): {portfolio_el_svr:.2f}")
        logging.info(f"EL/Funded Amount Ratio (Decision Tree): {el_ratio_dt:.4f}")
        logging.info(f"EL/Funded Amount Ratio (SVR): {el_ratio_svr:.4f}")

        return result, summary
    except Exception as e:
        logging.error(f"Error calculating Expected Loss: {e}")
        raise