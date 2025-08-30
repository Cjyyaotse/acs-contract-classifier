from fastapi import APIRouter, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import io
import os
import tempfile

from src.utils.ocr_pdf import extract_text_from_scanned
from src.models.schema import LlmContractText, LlmContractResponse
from src.services.llm_classifier import ContractClassifier

router = APIRouter(prefix="/llm", tags=["llm"])

# Initialize classifier service
classifier = ContractClassifier()


@router.post("/classify-text", response_model=LlmContractResponse)
async def classify_contract_text(contract: LlmContractText) -> LlmContractResponse:
    """Classify contract text directly."""
    try:
        result = classifier.predict_contract_category(contract.text)
        return LlmContractResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify-pdf", response_model=LlmContractResponse)
async def classify_contract_pdf(file: UploadFile = File(...)) -> LlmContractResponse:
    """Upload and classify a PDF (OCR fallback if scanned)."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        pdf_content = await file.read()

        # Try extracting text with PyPDF2
        reader = PdfReader(io.BytesIO(pdf_content))
        text = "".join((page.extract_text() or "") + "\n" for page in reader.pages)

        # Fallback to OCR if no text found
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

        # Run classification
        result = classifier.predict_contract_category(text)
        return LlmContractResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
