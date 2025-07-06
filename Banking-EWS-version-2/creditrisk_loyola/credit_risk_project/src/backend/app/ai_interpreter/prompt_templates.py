from typing import Dict, Any

def create_risk_interpretation_prompt(
    scores: Dict[str, Any],
    application_features: Dict[str, Any]
) -> str:
    """
    Creates a detailed prompt for an LLM to interpret credit risk scores.
    """
    pd_prob_ml = scores.get('pd_ml_probability', "N/A")
    lgd_ann_ml = scores.get('lgd_ml_ann', "N/A")  # This is LGD value
    ead_meta_ml = scores.get('ead_ml_meta', "N/A")
    el_ml = scores.get('expected_loss_ml', "N/A")

    pd_prob_ml_str = f"{pd_prob_ml * 100:.1f}%" if isinstance(pd_prob_ml, (float, int)) else "N/A"
    lgd_ann_ml_str = f"{lgd_ann_ml * 100:.1f}%" if isinstance(lgd_ann_ml, (float, int)) else "N/A"
    
    recovery_rate_ml = 1.0 - lgd_ann_ml if isinstance(lgd_ann_ml, (float, int)) else "N/A"
    recovery_rate_ml_str = f"{recovery_rate_ml * 100:.1f}%" if isinstance(recovery_rate_ml, (float, int)) else "N/A"

    loan_amnt_str = application_features.get('loan_amnt', "N/A")
    purpose_str = application_features.get('purpose', "N/A")
    grade_str = application_features.get('grade', "N/A")
    annual_inc_str = application_features.get('annual_inc', "N/A")
    dti_str = application_features.get('dti', "N/A")

    prompt = f"""
Analyze the following credit risk assessment for a loan application.
Provide a concise (3-5 sentences), easy-to-understand interpretation for a non-expert user.
Do not give financial advice or make a loan approval/rejection decision.
Focus only on interpreting the provided scores and features.

Application Details:
Loan Amount: {loan_amnt_str}
Purpose: {purpose_str}
Grade: {grade_str}
Annual Income: {annual_inc_str}
Debt-to-Income (DTI): {dti_str}

Risk Scores:
Probability of Default (PD): {pd_prob_ml_str} (Likelihood of not repaying the loan)
Loss Given Default (LGD): {lgd_ann_ml_str} (Estimated percentage of loss if default occurs)
Exposure at Default (EAD): {ead_meta_ml} (Estimated outstanding amount if default occurs)
Expected Loss (EL): {el_ml} (Overall anticipated monetary loss: PD * LGD * EAD)

Based on these figures, please explain what this risk profile generally means.
For example, what does a PD of {pd_prob_ml_str} indicate?
How do LGD and EAD contribute to the Expected Loss?
Conclude with a single sentence summarizing the risk.
"""
    return prompt.strip()


# --- NEW FUNCTION for Aggregate Metrics Summary Prompt ---
def create_aggregate_metrics_summary_prompt(
    cumulative_expected_loss: float,
    credit_risk_percentage: float,
    defaulters_percentage: float
) -> str:
    """
    Creates a prompt for an LLM to summarize aggregate credit risk metrics for a portfolio.
    """
    # Formatting the metrics for clear presentation in the prompt
    cel_str = f"{cumulative_expected_loss:,.2f}" # Format with commas and 2 decimal places
    crp_str = f"{credit_risk_percentage:.2%}"    # Format as percentage
    dp_str = f"{defaulters_percentage:.2%}"       # Format as percentage

    prompt = f"""
You are an AI assistant tasked with explaining the overall credit risk profile of a batch of loan applications
to a user who may not be an expert in finance. Use clear, concise language.
Do not provide financial advice or make investment decisions. Your goal is to interpret these summary metrics.

Here are the aggregate risk metrics for the batch:

1.  **Cumulative Expected Loss (CEL):** {cel_str}
    *Definition: This is the total monetary loss anticipated from this batch of loans due to potential defaults.*

2.  **Overall Credit Risk Percentage (CRP):** {crp_str}
    *Definition: This represents the Cumulative Expected Loss as a percentage of the total loan amount in the batch (CEL / Total Loan Amount). It indicates the portfolio's overall riskiness relative to its size.*

3.  **Defaulters Percentage (DP):** {dp_str}
    *Definition: This is the percentage of loan applications in the batch that are predicted to default (i.e., borrowers are likely to fail to repay their loans).*

Based on these aggregate metrics:
- Provide a brief (2-4 sentences) summary of what these figures imply for the overall risk profile of this batch of loans.
- Explain how these three metrics interrelate to give a picture of the portfolio's risk. For example, how does a high Defaulters Percentage potentially impact the Cumulative Expected Loss?
- Conclude with a single sentence that gives a general sense of the risk level suggested by these combined metrics (e.g., "Overall, these metrics suggest a [low/moderate/high/etc.] level of credit risk for this batch.").
"""
    return prompt.strip()