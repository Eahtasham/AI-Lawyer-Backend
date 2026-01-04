from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import rag_service
from app.services.council import council_service
from app.services.qdrant import qdrant_service
from app.logger import logger
import json
import asyncio

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f" New chat request: {request.query[:100]}...")
        response = await rag_service.process_query(request.query, request.top_k)
        logger.info(" Chat response generated")
        return response
    except Exception as e:
        logger.error(f" Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def stream_chat(query: str):
    """Streaming chat endpoint for live deliberations"""
    async def event_generator():
        try:
            logger.info(f"[Stream] Request: {query}")
            
            # 1. Retrieve chunks first (Sync or Async)
            # We can replicate a bit of rag_service here or add a stream method there too.
            # For simplicity, we'll do the retrieval here then stream council.
            
            yield "log: Searching Indian Kanoon Database...\n"
            chunks = qdrant_service.search(query, top_k=5)
            
            if not chunks:
                yield "data: {\"answer\": \"No relevant documents found.\", \"chunks\": []}\n"
                return

            yield f"log: Found {len(chunks)} relevant legal documents.\n"
            
            # Send chunks to frontend immediately
            chunks_json = json.dumps([c for c in chunks])
            yield f"chunks: {chunks_json}\n"

            # Build context
            context = "\n\n".join([
                f"[Chunk {c['rank']}]\n{c['text']}\nMetadata: {c['metadata']}"
                for c in chunks
            ])
            
            # 2. Hand over to Council Stream
            yield "log: Convening AI Council...\n"
            async for event in council_service.deliberate_stream(query, context):
                yield event

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")