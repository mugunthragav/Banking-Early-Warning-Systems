from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import pandas as pd
import io
import re # Kept for now, though its primary use in old snippet logic is removed.
from typing import List, Dict, Any # List, Dict, Any may not be strictly needed if using Pydantic models directly for type hints of orchestrator output

from app.database.db_session import get_db
from app.services.prediction_orchestration_service import orchestrate_predictions_for_batch
# Import your Pydantic schemas; ensure AggregatedPredictionResponse is defined there
from app.schemas import loan_application_schemas

router = APIRouter(
    prefix="/predict",
    tags=["Predictions"]
)

# MODIFIED: response_model changed to AggregatedPredictionResponse
@router.post("/batch_json", response_model=loan_application_schemas.AggregatedPredictionResponse)
async def predict_batch_from_json(
    request_batch: loan_application_schemas.LoanApplicationRequestBatch,
    db: Session = Depends(get_db)
):
    """
    Receives a batch of loan applications in JSON format, processes them,
    and returns an aggregated prediction response including overall metrics,
    AI summary of aggregates, and a list of lean individual results.
    """
    if not request_batch.applications:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No applications provided in the batch.")

    try:
        raw_apps_list_of_dicts = [app.model_dump(exclude_unset=True) for app in request_batch.applications]
        raw_applications_df = pd.DataFrame(raw_apps_list_of_dicts)
        print(f"[API Router] Received {len(raw_applications_df)} applications for JSON batch processing.")
    except Exception as e:
        print(f"[API Router] Error converting input JSON batch to DataFrame: {e}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input data format: {e}")

    try:
        # orchestrate_predictions_for_batch now returns the AggregatedPredictionResponse object.
        # The type hint for the variable can also be updated to AggregatedPredictionResponse.
        aggregated_response: loan_application_schemas.AggregatedPredictionResponse = orchestrate_predictions_for_batch(
            raw_applications_df=raw_applications_df,
            db=db
        )
        # Directly return the response from the service.
        # The old loop for creating api_response_items and PredictionResultItem is removed.
        return aggregated_response

    except ValueError as ve:
        print(f"[API Router] Value error during orchestration: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        print(f"[API Router] Unexpected error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {str(e)}")

# MODIFIED: response_model changed to AggregatedPredictionResponse
@router.post("/batch_csv", response_model=loan_application_schemas.AggregatedPredictionResponse)
async def predict_batch_from_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Receives a batch of loan applications as a CSV file, processes them,
    and returns an aggregated prediction response.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only CSV files are accepted.")

    contents = await file.read()
    # Try decoding with utf-8, then latin-1 as a fallback.
    try:
        decoded_contents = contents.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_contents = contents.decode('latin-1')
            print("[API Router] Decoded CSV with latin-1 after UTF-8 failed.")
        except UnicodeDecodeError as ude:
            print(f"[API Router] Failed to decode CSV with UTF-8 and Latin-1: {ude}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not decode CSV file. Ensure it's UTF-8 or Latin-1. Error: {ude}")
            
    buffer = io.StringIO(decoded_contents)
    try:
        raw_applications_df = pd.read_csv(buffer)
        print(f"[API Router] Received {len(raw_applications_df)} applications from CSV file '{file.filename}'.")
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV file is empty.")
    except Exception as e: # Catch other potential pandas parsing errors
        print(f"[API Router] Error parsing CSV file: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error parsing CSV file: {e}")
    finally:
        buffer.close()
        await file.close() # Ensure file is closed

    if raw_applications_df.empty:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No data found in CSV file after parsing.")

    try:
        # orchestrate_predictions_for_batch now returns the AggregatedPredictionResponse object.
        aggregated_response: loan_application_schemas.AggregatedPredictionResponse = orchestrate_predictions_for_batch(
            raw_applications_df=raw_applications_df,
            db=db
        )
        # Directly return the response from the service.
        # The old loop for creating api_response_items and PredictionResultItem is removed.
        return aggregated_response

    except ValueError as ve:
        print(f"[API Router] Value error during orchestration for CSV: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        print(f"[API Router] Unexpected error during CSV batch processing: {e}")
        import traceback
        traceback.print_exc()
        # It's good practice to ensure the error detail is a string
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred during CSV processing: {str(e)}")

@router.post("/batch_export_csv", response_class=StreamingResponse)
async def predict_batch_and_export_csv(
    request_batch: loan_application_schemas.LoanApplicationRequestBatch,
    db: Session = Depends(get_db)
):
    """
    Receives a batch of loan applications in JSON format, processes them,
    and returns the results as a CSV file.
    """
    if not request_batch.applications:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No applications provided in the batch.")

    try:
        raw_apps_list_of_dicts = [app.model_dump(exclude_unset=True) for app in request_batch.applications]
        raw_applications_df = pd.DataFrame(raw_apps_list_of_dicts)
        print(f"[API Router] Received {len(raw_applications_df)} applications for CSV export.")

        # Get predictions
        aggregated_response = orchestrate_predictions_for_batch(
            raw_applications_df=raw_applications_df,
            db=db
        )

        # Convert results to DataFrame
        results_df = pd.DataFrame([result.model_dump() for result in aggregated_response.results])
        
        # Convert DataFrame to CSV
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        # Return CSV file
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={
                'Content-Disposition': 'attachment; filename="prediction_results.csv"'
            }
        )
        
        return response

    except ValueError as ve:
        print(f"[API Router] Value error during CSV export: {ve}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve))
    except Exception as e:
        print(f"[API Router] Unexpected error during CSV export: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"An internal error occurred: {str(e)}")