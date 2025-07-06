import os
import logging
from pathlib import Path

# Logger setup
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    try:
        handler.stream.reconfigure(encoding='utf-8')
    except AttributeError:
        pass
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def generate_report(data, config):
    """
    Generate a PDF report for the given symbol data.
    
    Args:
        data (dict): Data containing symbol information (e.g., {'@CL#C': {...}})
        config (dict): Configuration with output directory (e.g., {'report': {'output_dir': 'path'}})
    
    Returns:
        str: Path to the generated PDF file
    """
    try:
        output_dir = config['report']['output_dir']
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists
        symbol = list(data.keys())[0]
        report_path = os.path.join(output_dir, f"{symbol}_report.pdf")
        
        # Mock PDF generation (replace with actual implementation if available)
        with open(report_path, 'wb') as f:
            f.write(b"%PDF-1.4\n% Mock PDF\n% Symbol: " + symbol.encode('utf-8') + b"\n")
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise