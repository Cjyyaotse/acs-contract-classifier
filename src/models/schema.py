from pydantic import BaseModel, Field
from typing import Optional , Any, Dict, List

#Pydantic models for few shot classification
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

class FewShotAddExampleRequest(BaseModel):
    contract_type: str
    example_text: str = Field(..., min_length=10, description="Training example text")
    metadata: dict = Field(default_factory=dict, description="Optional metadata about the example")

class FewShotAddExampleResponse(BaseModel):
    success: bool
    message: str
    contract_type: str
    examples_count: int

#Pydantic models for LLM classification
class LlmContractText(BaseModel):
    text: str

class LlmContractResponse(BaseModel):
    category: str
    reason: str

#Pydantic models for LLM classification
class TfLRClassificationRequest(BaseModel):
    text: str
    confidence_threshold: Optional[float] = 0.3
    top_n: Optional[int] = 3

class TfLRBatchClassificationRequest(BaseModel):
    texts: List[str]
    confidence_threshold: Optional[float] = 0.3
    top_n: Optional[int] = 3

class TfLRClassificationResponse(BaseModel):
    success: bool
    prediction: str
    confidence: float
    all_scores: dict
    top_predictions: list
    text_preview: str
    error: Optional[str] = None
