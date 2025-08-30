# services/few_shot_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from PyPDF2 import PdfReader
import io
import os
import tempfile

from utils.ocr_pdf import extract_text_from_scanned
from models.schema import ContractRequest, ClassificationResponse, BatchClassificationResponse
from models.schema import FewShotAddExampleResponse as AddExampleResponse, FewShotAddExampleRequest as AddExampleRequest
from services.few_shot_classifier import FewShotContractClassifier

router = APIRouter(prefix="/few_shot", tags=["few_shot"])

# Initialize the classifier service
classifier_service = FewShotContractClassifier()


@router.post("/classify", response_model=ClassificationResponse)
async def classify_contract(request: ContractRequest):
    """Classify a single contract document using few-shot learning"""
    try:
        if len(request.contract_text.strip()) < 10:
            return ClassificationResponse(
                prediction="Unknown",
                confidence=0.0,
                all_scores={},
                text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                error="Text too short for classification (minimum 10 characters required)"
            )

        result = classifier_service.classify_contract(
            request.contract_text,
            confidence_threshold=0.3
        )

        if "error" in result:
            return ClassificationResponse(
                prediction="Unknown",
                confidence=0.0,
                all_scores={},
                text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                error=result["error"]
            )

        return ClassificationResponse(
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            all_scores=result["all_scores"],
            text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
            error=None
        )

    except Exception as e:
        return ClassificationResponse(
            prediction="Unknown",
            confidence=0.0,
            all_scores={},
            text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
            error=f"Classification failed: {str(e)}"
        )


@router.post("/classify-pdf", response_model=ClassificationResponse)
async def classify_contract_pdf(file: UploadFile = File(...)):
    """
    Upload PDF file and classify the contract.
    Falls back to OCR if the PDF is scanned.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_content = await file.read()

        # ✅ First try PyPDF2
        reader = PdfReader(io.BytesIO(pdf_content))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        # ✅ Fallback to OCRmyPDF if no text extracted
        if not text.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            try:
                text = extract_text_from_scanned(tmp_path, language="eng")
            finally:
                os.unlink(tmp_path)  # cleanup no matter what

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF (even with OCR)"
            )

        # ✅ Run classification
        result = classifier_service.classify_contract(text)
        return ClassificationResponse(
            prediction=result["predicted_class"],
            confidence=result["confidence"],
            all_scores=result["all_scores"],
            text_preview=text[:100] + "..." if len(text) > 100 else text,
            error=None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(requests: List[ContractRequest]):
    """Classify multiple contract documents in batch"""
    results = []
    successful = 0
    failed = 0
    confidence_scores = []
    predictions_count = {}

    for request in requests:
        try:
            if len(request.contract_text.strip()) < 10:
                result = ClassificationResponse(
                    prediction="Unknown",
                    confidence=0.0,
                    all_scores={},
                    text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                    error="Text too short for classification"
                )
                failed += 1
            else:
                classification_result = classifier_service.classify_contract(
                    request.contract_text,
                    confidence_threshold=0.3
                )

                if "error" in classification_result:
                    result = ClassificationResponse(
                        prediction="Unknown",
                        confidence=0.0,
                        all_scores={},
                        text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                        error=classification_result["error"]
                    )
                    failed += 1
                else:
                    result = ClassificationResponse(
                        prediction=classification_result["predicted_class"],
                        confidence=classification_result["confidence"],
                        all_scores=classification_result["all_scores"],
                        text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                        error=None
                    )
                    successful += 1
                    confidence_scores.append(classification_result["confidence"])

                    # Count predictions
                    if classification_result["predicted_class"] in predictions_count:
                        predictions_count[classification_result["predicted_class"]] += 1
                    else:
                        predictions_count[classification_result["predicted_class"]] = 1

            results.append(result)

        except Exception as e:
            result = ClassificationResponse(
                prediction="Unknown",
                confidence=0.0,
                all_scores={},
                text_preview=request.contract_text[:100] + "..." if len(request.contract_text) > 100 else request.contract_text,
                error=f"Processing error: {str(e)}"
            )
            results.append(result)
            failed += 1

    # Calculate summary statistics
    summary = {
        "total_processed": len(requests),
        "successful_classifications": successful,
        "failed_classifications": failed,
        "success_rate": round(successful / len(requests) * 100, 2) if len(requests) > 0 else 0,
        "average_confidence": round(sum(confidence_scores) / len(confidence_scores), 3) if confidence_scores else 0,
        "predictions_distribution": predictions_count,
        "confidence_range": {
            "min": round(min(confidence_scores), 3) if confidence_scores else 0,
            "max": round(max(confidence_scores), 3) if confidence_scores else 0
        }
    }

    return BatchClassificationResponse(results=results, summary=summary)
