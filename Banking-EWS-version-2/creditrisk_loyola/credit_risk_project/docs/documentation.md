# API Documentation

The application exposes a REST API for predicting credit risk from a batch of loan applications.

### **Base URL**
`http://localhost:8000`

### **Prediction Endpoint**

`POST /api/v1/predict/batch_json`

This is the primary endpoint for the application. It accepts a list of loan applications and returns a detailed risk analysis.

#### **Request Body**

The request must be a JSON object containing a key `applications`, which is a list of individual loan application objects.

**Example Request Structure:**
```json
{
  "applications": [
    {
      "loan_amnt": 5000,
      "term": "36 months",
      "grade": "B",
      "home_ownership": "RENT",
      "annual_inc": 24000,
      "purpose": "debt_consolidation"
    },
    {
      "loan_amnt": 12000,
      "term": "60 months",
      "grade": "C",
      "home_ownership": "MORTGAGE",
      "annual_inc": 75000,
      "purpose": "credit_card"
    }
  ]
}
```

#### **Response Body**

The API returns a JSON object with aggregated metrics for the entire batch and a list of results for each individual application.

**Example Response Structure:**
```json
{
  "cumulative_expected_loss": 1250.75,
  "credit_risk_percentage": 0.0735,
  "defaulters_percentage": 0.15,
  "aggregate_metrics_ai_summary": "The overall risk for this batch is moderate...",
  "results": [
    {
      "application_db_id": 1,
      "pd_ml_probability": 0.10,
      "expected_loss_ml": 450.25,
      ...
    },
    {
      "application_db_id": 2,
      "pd_ml_probability": 0.20,
      "expected_loss_ml": 800.50,
      ...
    }
  ]
}
```