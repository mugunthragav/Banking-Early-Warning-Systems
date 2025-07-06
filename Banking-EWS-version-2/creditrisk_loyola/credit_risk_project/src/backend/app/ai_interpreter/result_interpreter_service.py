from typing import Dict, Any
from openai import OpenAI # Import the OpenAI library

# MODIFIED: Import the new prompt creation function for aggregate metrics
from .prompt_templates import create_risk_interpretation_prompt, create_aggregate_metrics_summary_prompt # NEW import

# Global OpenAI client instance
openai_client = None

def _get_openai_client(api_key: str) -> OpenAI | None:
    """Initializes and returns the OpenAI client if not already done."""
    global openai_client
    if openai_client is None or not hasattr(openai_client, 'api_key') or openai_client.api_key != api_key:
        if api_key and api_key != "YOUR_LLM_API_KEY_HERE" and api_key.startswith("sk-"):
            try:
                openai_client = OpenAI(api_key=api_key)
                print("[AI Interpreter] OpenAI client initialized/updated.")
            except Exception as e:
                print(f"[AI Interpreter] Failed to initialize OpenAI client: {e}")
                openai_client = None
        else:
            if openai_client is not None or (api_key and api_key != "YOUR_LLM_API_KEY_HERE" and not api_key.startswith("sk-")):
                print("[AI Interpreter] OpenAI API key not configured or invalid for client initialization.")
            openai_client = None  # Ensure it's None if key is bad
    return openai_client

def generate_interpretation(
    scores: Dict[str, Any],
    application_features: Dict[str, Any],
    llm_config: Dict[str, Any] # Expects {'api_key': '...', 'model_name': '...'}
) -> str:
    """
    Generates a human-readable interpretation of individual credit risk scores using OpenAI LLM.
    """
    prompt = create_risk_interpretation_prompt(scores, application_features)
    api_key = llm_config.get('api_key')
    model_name = llm_config.get('model_name', "gpt-3.5-turbo")

    # print(f"[AI Interpreter] Attempting to use model: {model_name} for individual interpretation.")
    # print(f"[AI Interpreter] Generated Prompt for LLM (Individual):\n{prompt}")

    client = _get_openai_client(api_key)

    if not client:
        print("[AI Interpreter] OpenAI client not initialized for individual interpretation. Returning placeholder.")
        pd_prob = scores.get('pd_ml_probability')
        el_ml = scores.get('expected_loss_ml')
        if pd_prob is not None:
            risk_level = "low" if pd_prob < 0.1 else "moderate" if pd_prob < 0.3 else "high"
            return (f"AI Interpretation (LLM Client Error/No Key): "
                    f"The application's ML Probability of Default (PD) is {pd_prob*100:.1f}%, "
                    f"indicating a {risk_level} risk. "
                    f"The ML Expected Loss is {el_ml if el_ml is not None else 'N/A'}.")
        else:
            return "AI Interpretation (LLM Client Error/No Key): Key ML scores not available for a detailed interpretation."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        interpretation = response.choices[0].message.content.strip()
        # print(f"[AI Interpreter] Successfully received individual interpretation from OpenAI LLM.")
        return interpretation
    except Exception as e:
        print(f"[AI Interpreter] Error calling OpenAI LLM API for individual interpretation: {e}")
        return f"Error: Could not generate AI interpretation due to an API issue. ({str(e)[:100]}...)"


# --- NEW FUNCTION for Aggregate Metrics Summary ---
def generate_aggregate_metrics_summary(
    cumulative_expected_loss: float,
    credit_risk_percentage: float,
    defaulters_percentage: float,
    llm_config: Dict[str, Any]  # Expects {'api_key': '...', 'model_name': '...'}
) -> str:
    """
    Generates a human-readable summary of aggregate credit risk metrics using OpenAI LLM.
    """
    prompt = create_aggregate_metrics_summary_prompt(
        cumulative_expected_loss,
        credit_risk_percentage,
        defaulters_percentage
    )
    api_key = llm_config.get('api_key')
    model_name = llm_config.get('model_name', "gpt-3.5-turbo") # Default model

    print(f"[AI Interpreter] Attempting to use model: {model_name} for aggregate summary.")
    # print(f"[AI Interpreter] Generated Prompt for LLM (Aggregate):\n{prompt}") # Uncomment for debugging

    client = _get_openai_client(api_key)

    if not client:
        print("[AI Interpreter] OpenAI client not initialized for aggregate summary. Returning placeholder summary.")
        return (f"Aggregate Metrics Summary (LLM Client Error/No Key): "
                f"Cumulative Expected Loss: {cumulative_expected_loss:.2f}, "
                f"Credit Risk Percentage: {credit_risk_percentage:.2%}, "
                f"Defaulters Percentage: {defaulters_percentage:.2%}. "
                f"Automated AI summary could not be generated.")

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # You might add a system message here if the prompt template doesn't include one
                # {"role": "system", "content": "You are an AI assistant summarizing portfolio credit risk metrics."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # Slightly higher for more narrative summary
            max_tokens=300    # Adjust as needed for desired length
        )
        summary = response.choices[0].message.content.strip()
        print(f"[AI Interpreter] Successfully received aggregate summary from OpenAI LLM.")
        return summary
    except Exception as e:
        print(f"[AI Interpreter] Error calling OpenAI LLM API for aggregate summary: {e}")
        return f"Error: Could not generate aggregate AI summary due to an API issue. ({str(e)[:100]}...)"