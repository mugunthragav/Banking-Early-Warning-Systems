# Liquidity Risk Predictor

This project is a full-stack web application designed to predict the liquidity risk of a financial institution using a machine learning model. It features a modern frontend built with Angular and a robust backend API powered by FastAPI and Python.

The application not only predicts the risk status but also provides explanations for the prediction using SHAP (SHapley Additive exPlanations), offering users actionable insights into the key drivers of risk.

## ✨ Features

-   **Intuitive Web Interface**: Easy-to-use form for inputting financial metrics.
-   **CSV Data Upload**: Populate the form quickly by uploading a CSV file.
-   **ML-Powered Predictions**: Utilizes a pre-trained XGBoost model to classify risk as "RISKY" or "NOT RISKY".
-   **Explainable AI (XAI)**: Integrates SHAP to explain which factors contributed most to each prediction.
-   **Actionable Advice**: Generates targeted recommendations based on the most impactful risk factors.
-   **Decoupled Architecture**: A standalone Angular frontend that communicates with a FastAPI backend API.

## 🛠️ Technology Stack

-   **Backend**: Python, FastAPI, Uvicorn, XGBoost, SHAP, Pandas, Scikit-learn
-   **Frontend**: Angular, TypeScript, HTML5, CSS3
-   **Core Libraries**: Joblib, Pydantic, NumPy

## 📂 Project Structure

The project is organized into two main parts: the backend server and the frontend client.


## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

-   [Node.js](https://nodejs.org/) (which includes npm) for the frontend.
-   [Python 3.8+](https://www.python.org/) and `pip` for the backend.

### ⚙️ Installation & Setup

**1. Backend (FastAPI)**

First, set up the Python backend server.

```bash
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
# On Windows:
python -m venv .venv
.\.venv\Scripts\activate
# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Navigate to the frontend directory from the root
cd frontend

# Install the required Node.js packages
npm install

# In the backend/ directory (with virtual environment activated)
uvicorn main:app --reload

# In the frontend/ directory
ng serve