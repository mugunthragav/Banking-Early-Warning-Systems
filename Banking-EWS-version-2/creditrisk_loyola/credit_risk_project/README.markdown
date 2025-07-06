# Credit Risk Project

This project is a full-stack application designed to perform credit risk analysis. It consists of a backend built with FastAPI and a frontend developed using Angular. The backend processes loan application data, calculates risk metrics (PD, LGD, EAD, Expected Loss), and attempts to log results to a MySQL database. The frontend provides a user interface for displaying liquidity risk analysis results.

## Project Structure
- **Backend**: Located in `src/backend`, implemented using FastAPI and Python.
- **Frontend**: Located in `src/frontend/Liquidity-risk analysis`, built with Angular.

## Prerequisites
- **Python 3.11**: For running the backend.
- **Node.js and Angular CLI**: For running the frontend.
- **MySQL**: For database connectivity (ensure the MySQL server is running).
- **Virtual Environment**: Recommended for managing Python dependencies.
- **Required Python Packages**: Install via `requirements.txt` (e.g., `fastapi`, `uvicorn`, `sqlalchemy`, `mysql-connector-python`, `scikit-learn`).
- **Required Node Modules**: Install via `npm install` in the frontend directory.

## Setup Instructions

### Backend Setup

1. **Set up a virtual environment** (if not already set up):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt


   ```
---To add : Steps to generate 7 pkl files and store in artifacts folder which is required to run the backend api
3. **Navigate to the backend directory**:
   ```bash
   cd src/backend
   ```
4. **Configure environment variables**:(optional if you need database for logging results)
   - Set the MySQL database URL in your environment or configuration file:
     ```
     DATABASE_URL=mysql+mysqlconnector://USER:PASSWORD@HOST:PORT/DB_NAME
     ```
   - Ensure the OpenAI API key is configured if aggregate summary functionality is needed.

5. **Run the backend server**:
   ```bash
   uvicorn app.main:app --reload
   ```
   - The server will run on `http://127.0.0.1:8000`.
   - Access the API documentation at `http://127.0.0.1:8000/docs`.

### Frontend Setup
1. **Navigate to the frontend directory**:
   ```bash
   cd src/frontend/Liquidity-risk analysis
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Run the frontend server**:
   ```bash
   ng serve
   ```
   - The application will be available at `http://localhost:4200/Market-health'.

## Known Issues and Troubleshooting

### Backend Issues
1. **ModuleNotFoundError: No module named 'app'**:
   - Ensure you are running `uvicorn` from the correct directory (`src/backend`) where `app/main.py` exists.
   - Example command: `uvicorn app.main:app --reload`.

2. **MySQL Connection Error**:
   - Error: `(mysql.connector.errors.DatabaseError) 2003 (HY000): Can't connect to MySQL server on 'localhost:3306' (10061)`.
   - **Solution**: Verify that the MySQL server is running and accessible. Check the `DATABASE_URL` configuration for correct credentials and host/port.

3. **InconsistentVersionWarning from scikit-learn**:
   - Warning: Models were pickled with scikit-learn version 1.6.1, but the current version is 1.7.0.
   - **Solution**: Re-pickle models using the current scikit-learn version or suppress warnings if results are unaffected. Refer to [scikit-learn documentation](https://scikit-learn.org/stable/model_persistence.html#security-maintainability-limitations).

4. **OpenAI API Key Issue**:
   - Error: `OpenAI API key or model name is not configured or invalid`.
   - **Solution**: Configure the OpenAI API key in your environment or skip the aggregate summary feature.

### Frontend Issues
1. **NG1: Object is possibly 'null'**:
   - Errors in `Liquidity-risk.component.html` related to `response` object properties.
   - **Solution**: Update the component to ensure `response` is properly initialized or use safe navigation operators (`?.`) correctly.

2. **NG8107: Unnecessary optional chain operator**:
   - Warnings suggest replacing `?.` with `.` in `Liquidity-risk.component.html`.
   - **Solution**: Modify the template to use `.` for properties that are guaranteed to be defined, or ensure proper type checking in the component.

3. **Component Name Mismatch**:
   - Errors reference `MarketRiskComponent` but occur in `Liquidity-risk.component.ts`.
   - **Solution**: Verify the component name in `Liquidity-risk.component.ts` and ensure the template file is correctly linked.

## Usage
- **Backend**:
  - Upload a CSV file (e.g., `merlin checking.csv`) via the `/api/v1/predict/batch_csv` endpoint to process loan applications.
  - The API calculates risk metrics (PD, LGD, EAD, Expected Loss) and attempts to log results to the database.
  - Check logs for processing details and errors.

- **Frontend**:
  - Access the application at `http://localhost:4200` to view liquidity risk analysis.
  - The interface displays metrics such as `aggregate_metrics_ai_summary`, `cumulative_expected_loss`, `credit_risk_percentage`, and `defaulters_percentage`.

## Notes
- Ensure the MySQL server is running before starting the backend.
- Address scikit-learn version warnings to prevent potential issues with model predictions.
- Fix frontend template errors to ensure proper rendering of data.
- The backend artifacts are stored in `src/backend/artifacts`.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.