# app.py (Optional FastAPI wrapper)
from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from services.tf_logistic_regression import ContractClassifierService

router = APIRouter(prefix="/tf_logistic_regression", tags=["tf_logistic_regression"])

# Initialize the classifier service
classifier_service = ContractClassifierService()

class ClassificationRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.3
    top_n: Optional[int] = 3

class BatchClassificationRequest(BaseModel):
    texts: List[str]
    confidence_threshold: Optional[float] = 0.3
    top_n: Optional[int] = 3

class ClassificationResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    all_scores: dict
    top_predictions: list
    text_preview: str
    error: Optional[str] = None


@router.post("/classify", response_model=ClassificationResponse)
async def classify_contract(request: ClassificationRequest):
    """Classify a single contract document"""
    result = classifier_service.classify_contract(
        request.text,
        request.confidence_threshold,
        request.top_n
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result

@router.post("/classify/batch")
async def classify_batch(request: BatchClassificationRequest):
    """Classify multiple contract documents"""
    results = classifier_service.classify_batch(
        request.texts,
        request.confidence_threshold,
        request.top_n
    )
    return {"results": results}
