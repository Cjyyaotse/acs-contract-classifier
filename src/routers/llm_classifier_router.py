from fastapi import APIRouter, UploadFile, File, HTTPException
from PyPDF2 import PdfReader
import io
import os
import tempfile

# Import your existing classifier and OCR helper
from utils.ocr_pdf import extract_text_from_scanned
from models.schema import LlmContractText, LlmContractResponse
from services.llm_classifier import ContractClassifier

router = APIRouter(prefix="/llm", tags=["llm"])

# Initialize your classifier
classifier = ContractClassifier()


@router.post("/classify-text", response_model=LlmContractResponse)
async def classify_contract_text(contract: LlmContractText) -> LlmContractResponse:
    """
    Classify contract text directly
    """
    try:
        result = classifier.predict_contract_category(contract.text)
        return LlmContractResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify-pdf", response_model=LlmContractResponse)
async def classify_contract_pdf(file: UploadFile = File(...)) -> LlmContractResponse:
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

        # ✅ Fallback to OCRmyPDF if text empty
        if not text.strip():
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_content)
                tmp_path = tmp.name

            text = extract_text_from_scanned(tmp_path, language="eng")
            os.unlink(tmp_path)  # cleanup

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from PDF (even with OCR)"
            )

        # ✅ Run classification
        result = classifier.predict_contract_category(text)
        return LlmContractResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")
