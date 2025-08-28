from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import your existing modules
from src.utils.text_conversion import pdf_to_text, docx_to_text
from src.services import ContractClassifier, ClassificationResponse, BatchClassificationResponse, ContractRequest

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router (instead of app)
router = APIRouter(prefix="/zero_shot", tags=["zero_shot"])

# Initialize the classifier
classifier = ContractClassifier()

def extract_text_from_request(request: ContractRequest) -> str:
    """Extract text from either direct text input or file path"""
    if request.text:
        return request.text
    
    if request.file_path:
        if request.file_path.lower().endswith('.pdf'):
            return pdf_to_text(request.file_path)
        elif request.file_path.lower().endswith(('.doc', '.docx')):
            return docx_to_text(request.file_path)
        else:
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file format. Only PDF and DOCX are supported."
            )
    
    raise HTTPException(
        status_code=400, 
        detail="Either 'text' or 'file_path' must be provided"
    )

@router.post("/classify", response_model=ClassificationResponse)
async def classify_contract(request: ContractRequest):
    """
    Classify a single contract document using zero-shot learning
    """
    try:
        # Extract text from request
        text_content = extract_text_from_request(request)
        
        if not text_content.strip():
            raise HTTPException(
                status_code=400, 
                detail="No text content could be extracted from the input"
            )
        
        # Perform classification
        result = classifier.classify_contract(
            text_content, 
            confidence_threshold=request.confidence_threshold
        )
        
        return ClassificationResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing classification request: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_contract_batch(requests: List[ContractRequest]):
    """
    Classify multiple contract documents in batch
    """
    results = []
    successful = 0
    total_confidence = 0.0
    
    for request in requests:
        try:
            # Extract text from request
            text_content = extract_text_from_request(request)
            
            if not text_content.strip():
                result = ClassificationResponse(
                    prediction="Error",
                    confidence=0.0,
                    all_scores={},
                    text_preview="",
                    error="No text content extracted"
                )
            else:
                # Perform classification
                classification_result = classifier.classify_contract(
                    text_content, 
                    confidence_threshold=request.confidence_threshold
                )
                result = ClassificationResponse(**classification_result)
                
                if result.error is None:
                    successful += 1
                    total_confidence += result.confidence
            
        except Exception as e:
            result = ClassificationResponse(
                prediction="Error",
                confidence=0.0,
                all_scores={},
                text_preview="",
                error=str(e)
            )
        
        results.append(result)
    
    # Calculate summary
    avg_confidence = total_confidence / successful if successful else 0
    
    return BatchClassificationResponse(
        results=results,
        summary={
            "total_requests": len(requests),
            "successful": successful,
            "failed": len(requests) - successful,
            "average_confidence": round(avg_confidence, 4)
        }
    )

@router.get("/categories")
async def get_categories():
    """Get supported contract categories"""
    return {
        "categories": classifier.candidate_labels
    }
