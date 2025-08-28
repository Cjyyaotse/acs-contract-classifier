from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Contract Processing API",
    description="API for processing raw contract text",
    version="1.0.0"
)

class ContractRequest(BaseModel):
    contract_text: str = Field(..., min_length=1, description="Raw contract text or PDF extract")
    metadata: dict = Field(default_factory=dict, description="Optional metadata about the contract")

# Your service import (replace with your actual service)
# from your_service import process_document

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Contract Processing API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/process-contract")
async def process_contract_endpoint(
    request: ContractRequest
):
    """
    Process raw contract text

    Args:
        request: ContractRequest containing contract text and optional metadata

    Returns:
        JSON response with processing results
    """
    try:
        contract_text = request.contract_text.strip()

        # Validate contract text
        if not contract_text:
            raise HTTPException(status_code=400, detail="Contract text cannot be empty")

        # Optional: Check text length
        max_length = 1000000  # 1MB of text
        if len(contract_text) > max_length:
            raise HTTPException(status_code=413, detail="Contract text too long")

        logger.info(f"Processing contract with {len(contract_text)} characters")

        # Here you would call your service
        # Replace this with your actual service call
        # result = your_service.process_contract(contract_text, request.metadata)

        # Mock response - replace with your service output
        result = {
            "text_length": len(contract_text),
            "word_count": len(contract_text.split()),
            "metadata": request.metadata,
            "status": "processed",
            "message": "Contract processed successfully",
            "processed_data": {
                "summary": "Contract processing completed",
                "key_terms": [],
                "analysis": {}
            }
        }

        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": result
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing contract: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
