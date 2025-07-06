from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import pandas as pd
import os
import logging

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

def generate_report(results, config):
    """Generate a PDF report for market risk results."""
    try:
        output_dir = config['report']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        styles = getSampleStyleSheet()
        elements = []

        for symbol, data in results.items():
            report_path = os.path.join(output_dir, f"{symbol}_report.pdf")
            doc = SimpleDocTemplate(report_path, pagesize=letter)
            elements.append(Paragraph(f"Market Risk Report: {symbol}", styles['Title']))
            elements.append(Spacer(1, 12))

            # VaR Results
            elements.append(Paragraph("Value at Risk (VaR)", styles['Heading2']))
            var_data = pd.DataFrame({
                k: [v['var'], v['revenue_impact'], v['capital_impact'], v['liquidity_impact']]
                for k, v in data['monte_carlo_var'].items()
            }, index=['VaR ($)', 'Revenue Impact ($)', 'Capital Impact ($)', 'Liquidity Impact ($)']).T
            table = Table([['Metric'] + list(var_data.index)] + [[k] + list(v) for k, v in var_data.items()])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

            # Stress Test Results
            elements.append(Paragraph("Stress Test Results", styles['Heading2']))
            stress_data = pd.DataFrame({
                k: [v['loss'], v['drawdown'], v['revenue_impact'], v['capital_impact'], v['liquidity_impact']]
                for k, v in data['stress_results'].items()
            }, index=['Loss ($)', 'Drawdown (%)', 'Revenue Impact ($)', 'Capital Impact ($)', 'Liquidity Impact ($)']).T
            table = Table([['Scenario'] + list(stress_data.index)] + [[k] + list(v) for k, v in stress_data.items()])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)
            elements.append(Spacer(1, 12))

            # Backtest Results
            elements.append(Paragraph("Backtest Results", styles['Heading2']))
            backtest_data = pd.DataFrame(data['backtest_results']).T
            table = Table([['Metric', 'Exceedances', 'Coverage (%)']] + [[k] + list(v) for k, v in backtest_data.iterrows()])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            elements.append(table)

            doc.build(elements)
            logger.info(f"Report generated: {report_path}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")