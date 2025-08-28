from pydantic import BaseModel, Field
from typing import Optional , Any, Dict, List

class ContractRequest(BaseModel):
    contract_text: str = Field(..., min_length=1, description="Raw contract text or PDF extract")
    metadata: dict = Field(default_factory=dict, description="Optional metadata about the contract")

class ClassificationResponse(BaseModel):
    prediction: str
    confidence: float
    all_scores: Dict[str, float]
    text_preview: str
    error: Optional[str] = None

class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]
    summary: Dict[str, Any]