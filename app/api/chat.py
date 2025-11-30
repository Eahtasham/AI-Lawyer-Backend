from fastapi import APIRouter, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import rag_service
from app.logger import logger

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f" New chat request: {request.query[:100]}...")
        response = rag_service.process_query(request.query, request.top_k)
        logger.info(" Chat response generated")
        return response
    except Exception as e:
        logger.error(f" Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))