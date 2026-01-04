from pydantic import BaseModel
from typing import List, Dict

class ChatRequest(BaseModel):
    query: str
    top_k: int = 5
    conversation_id: str | None = None

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
