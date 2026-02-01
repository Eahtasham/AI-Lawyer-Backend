from pydantic import BaseModel
from typing import List, Dict
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
