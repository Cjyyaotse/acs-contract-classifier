# app.py (Optional FastAPI wrapper)
from fastapi import APIRouter, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import io
import os
import tempfile

from src.utils.ocr_pdf import extract_text_from_scanned
from src.models.schema import (
    TfLRClassificationRequest,
    TfLRBatchClassificationRequest,
    TfLRClassificationResponse,
)
from src.services.tf_logistic_regression import ContractClassifierService

router = APIRouter(prefix="/tf_logistic_regression", tags=["tf_logistic_regression"])

# Initialize classifier
classifier_service = ContractClassifierService()


@router.post("/classify", response_model=TfLRClassificationResponse)
async def classify_contract(request: TfLRClassificationRequest):
    """Classify a single contract text."""
    result = classifier_service.classify_contract(
        request.text,
        request.confidence_threshold,
        request.top_n,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/classify-pdf", response_model=TfLRClassificationResponse)
async def classify_contract_pdf(file: UploadFile = File(...)):
    """Upload and classify a PDF (OCR fallback if scanned)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_content = await file.read()

        # Extract text with PyPDF2
        reader = PdfReader(io.BytesIO(pdf_content))
        text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)

        # Fallback: OCR if no text found
        if not text.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name
            try:
                text = extract_text_from_scanned(tmp_path, language="eng")
            finally:
                os.unlink(tmp_path)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF (even with OCR)")

        result = classifier_service.classify_contract(text)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@router.post("/classify/batch")
async def classify_batch(request: TfLRBatchClassificationRequest):
    """Classify multiple contract texts in batch."""
    results = classifier_service.classify_batch(
        request.texts,
        request.confidence_threshold,
        request.top_n,
    )
    return {"results": results}
