from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import PyPDF2
import io
from typing import Dict

# Import your existing classifier
from services.llm_classifier import ContractClassifier

router = APIRouter(prefix="/contracts", tags=["contracts"])

# Initialize your classifier (make sure to import it)
classifier = ContractClassifier()

class ContractText(BaseModel):
    text: str

class ContractResponse(BaseModel):
    category: str
    reason: str

@router.post("/classify-text", response_model=ContractResponse)
async def classify_contract_text(contract: ContractText) -> Dict[str, str]:
    """
    Classify contract text directly
    """
    try:
        result = classifier.predict_contract_category(contract.text)
        return ContractResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@router.post("/classify-pdf", response_model=ContractResponse)
async def classify_contract_pdf(file: UploadFile = File(...)) -> Dict[str, str]:
    """
    Upload PDF file and classify the contract
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    try:
        # Read PDF content
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        # Classify the extracted text
        result = classifier.predict_contract_category(text)
        return ContractResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing failed: {str(e)}")

@router.get("/categories")
async def get_contract_categories():
    """
    Get list of available contract categories
    """
    return {"categories": classifier.categories}
