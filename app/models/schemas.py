from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum

class ChatMode(str, Enum):
    FAST = "fast"
    BALANCED = "balanced"
    RESEARCH = "research"

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: str | None = None
    mode: ChatMode = ChatMode.RESEARCH

class ChunkResult(BaseModel):
    rank: int
    score: float
    text: str
    metadata: Dict

class ChatResponse(BaseModel):
    query: str
    answer: str
    chunks: List[ChunkResult]
    llm_model: str
    council_opinions: List[Dict] = []

# --- Document Analyzer Models ---

class DocumentAnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class AnalysisSectionItem(BaseModel):
    text: str
    significance: Optional[str] = None
    severity: Optional[str] = None  # high, medium, low (for risks)
    party: Optional[str] = None  # for obligations
    deadline: Optional[str] = None  # for obligations
    term: Optional[str] = None  # for jargon
    simplified: Optional[str] = None  # for jargon

class AnalysisSection(BaseModel):
    title: str
    content: Optional[str] = None  # for summary
    items: Optional[List[AnalysisSectionItem]] = None  # for lists

class DocumentAnalysisResponse(BaseModel):
    id: str
    file_name: str
    file_type: str
    status: DocumentAnalysisStatus
    sections: Optional[List[AnalysisSection]] = None
    related_chunks: Optional[List[Dict]] = None
    created_at: Optional[str] = None

class DocumentFollowUpRequest(BaseModel):
    analysis_id: str
    question: str
