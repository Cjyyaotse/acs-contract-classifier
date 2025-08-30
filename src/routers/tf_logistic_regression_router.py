# app.py (Optional FastAPI wrapper)
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict
from PyPDF2 import PdfReader
import io
import os
import tempfile

from utils.ocr_pdf import extract_text_from_scanned
from models.schema import (
    TfLRClassificationRequest,
    TfLRBatchClassificationRequest,
    TfLRClassificationResponse,
)
from services.tf_logistic_regression import ContractClassifierService

router = APIRouter(prefix="/tf_logistic_regression", tags=["tf_logistic_regression"])

# Initialize the classifier service
classifier_service = ContractClassifierService()


@router.post("/classify", response_model=TfLRClassificationResponse)
async def classify_contract(request: TfLRClassificationRequest):
    """Classify a single contract document"""
    result = classifier_service.classify_contract(
        request.text,
        request.confidence_threshold,
        request.top_n,
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/classify-pdf", response_model=TfLRClassificationResponse)
async def classify_contract_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload PDF file and classify the contract.
    Falls back to OCR if no text can be extracted.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # ✅ Read PDF content
        pdf_content = await file.read()
        reader = PdfReader(io.BytesIO(pdf_content))

        # ✅ Extract text from all pages
        text = ""
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"

        # ✅ Fallback to OCR if no text extracted
        if not text.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            try:
                text = extract_text_from_scanned(tmp_path, language="eng")
            finally:
                os.unlink(tmp_path)  # cleanup temp file

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF (even with OCR)",
            )

        # ✅ Classify
        result = classifier_service.classify_contract(text)

        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")


@router.post("/classify/batch")
async def classify_batch(request: TfLRBatchClassificationRequest):
    """Classify multiple contract documents"""
    results = classifier_service.classify_batch(
        request.texts,
        request.confidence_threshold,
        request.top_n,
    )
    return {"results": results}
