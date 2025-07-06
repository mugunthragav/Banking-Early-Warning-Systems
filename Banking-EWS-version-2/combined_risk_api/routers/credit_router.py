from fastapi import APIRouter, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
import io
from pydantic import BaseModel
from typing import List, Dict, Any

router = APIRouter(
    prefix="/predict",
    tags=["Credit Predictions"]
)

class LoanApplicationRequestBatch(BaseModel):
    applications: List[Dict[str, Any]]

class PredictionResult(BaseModel):
    status: str
    input_data: Dict[str, Any]

class AggregatedPredictionResponse(BaseModel):
    results: List[PredictionResult]

@router.post("/batch_json", response_model=AggregatedPredictionResponse)
async def predict_batch_from_json(
    request_batch: LoanApplicationRequestBatch
):
    """
    Mock endpoint for JSON batch processing of loan applications.
    """
    if not request_batch.applications:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No applications provided in the batch.")
    
    try:
        results = [
            PredictionResult(status="Mock prediction processed", input_data=app)
            for app in request_batch.applications
        ]
        return AggregatedPredictionResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input data format: {str(e)}")

@router.post("/batch_csv", response_model=AggregatedPredictionResponse)
async def predict_batch_from_csv(
    file: UploadFile = File(...)
):
    """
    Mock endpoint for CSV batch processing of loan applications.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file type. Only CSV files are accepted.")

    contents = await file.read()
    try:
        decoded_contents = contents.decode('utf-8')
    except UnicodeDecodeError:
        try:
            decoded_contents = contents.decode('latin-1')
        except UnicodeDecodeError as ude:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Could not decode CSV file: {str(ude)}")
            
    buffer = io.StringIO(decoded_contents)
    try:
        df = pd.read_csv(buffer)
        if df.empty:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV file is empty.")
        results = [
            PredictionResult(status="Mock prediction processed", input_data=row.to_dict())
            for _, row in df.iterrows()
        ]
        return AggregatedPredictionResponse(results=results)
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV file is empty.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error parsing CSV file: {str(e)}")
    finally:
        buffer.close()
        await file.close()

@router.post("/batch_export_csv", response_class=StreamingResponse)
async def predict_batch_and_export_csv(
    request_batch: LoanApplicationRequestBatch
):
    """
    Mock endpoint for processing loan applications and exporting results as CSV.
    """
    if not request_batch.applications:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No applications provided in the batch.")

    try:
        results = [
            {"status": "Mock prediction processed", **app}
            for app in request_batch.applications
        ]
        results_df = pd.DataFrame(results)
        output = io.StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={'Content-Disposition': 'attachment; filename="credit_prediction_results.csv"'}
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid input data format: {str(e)}")