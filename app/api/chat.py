from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.rag import rag_service
from app.services.council import council_service
from app.services.qdrant import qdrant_service
from app.services.db import db_service
from app.api.deps import get_current_user
from app.logger import logger
import json
import asyncio

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    user_id: str = Depends(get_current_user)
):
    """Main chat endpoint"""
    try:
        logger.info(f" New chat request from {user_id}: {request.query[:100]}...")
        
        # 1. Create/Get Conversation
        conversation_id = request.conversation_id
        if not conversation_id:
            conversation_id = db_service.create_conversation(user_id, title=request.query[:50])
            
        # 2. Log User Message
        db_service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=request.query
        )

        # 3. Process Query
        response = await rag_service.process_query(request.query, request.top_k)
        
        # 4. Log Assistant Response
        db_service.add_message(
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=response.answer,
            metadata={"chunks": [c.dict() for c in response.chunks], "model": response.llm_model}
        )
        
        logger.info(" Chat response generated and logged")
        return response
    except Exception as e:
        logger.error(f" Chat endpoint error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chat/{conversation_id}")
async def delete_conversation_endpoint(
    conversation_id: str,
    user_id: str = Depends(get_current_user)
):
    try:
        success = db_service.delete_conversation(conversation_id, user_id)
        if not success:
             raise HTTPException(status_code=404, detail="Conversation not found or access denied")
        return {"status": "success", "message": "Conversation deleted"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/chat/{conversation_id}")
async def update_conversation_endpoint(
    conversation_id: str,
    payload: dict,
    user_id: str = Depends(get_current_user)
):
    """
    Update conversation details (title, is_pinned).
    Payload example: {"title": "New Name", "is_pinned": true}
    """
    try:
        title = payload.get("title")
        is_pinned = payload.get("is_pinned")
        
        success = db_service.update_conversation(conversation_id, user_id, title=title, is_pinned=is_pinned)
        if not success:
             raise HTTPException(status_code=404, detail="Conversation not found or update failed")
        
        return {"status": "success", "message": "Conversation updated"}
    except Exception as e:
        logger.error(f"Update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream")
async def stream_chat(
    query: str, 
    conversation_id: str = None,
    user_id: str = Depends(get_current_user)
):
    """
    Streaming chat endpoint.
    Uses standard Bearer token authentication via Depends.
    """
    # user_id is now guaranteed by Depends(get_current_user)

    async def event_generator():
        try:
            logger.info(f"[Stream] Request from {user_id}: {query}")
            
            # Lists to capture full metadata (Initialize EARLY)
            stream_logs = []
            stream_chunks = []
            stream_opinions = []

            # Use provided conversation_id or create new one
            conv_id = conversation_id
            
            # Log User Message (with recovery for missing/deleted conversations)
            try:
                if not conv_id:
                     raise ValueError("No ID provided, force creation")
                db_service.add_message(conv_id, user_id, "user", query)
            except Exception as e:
                is_fk_error = "foreign key constraint" in str(e) or "23503" in str(e) 
                if is_fk_error or not conv_id:
                    logger.warning(f"[Stream] Conversation {conv_id} missing or invalid. Creating new conversation.")
                    title = query[:50] + "..." if len(query) > 50 else query
                    # Force creation with the requested ID to keep Frontend in sync
                    conv_id = db_service.create_conversation(user_id, title=title, id=conv_id)
                    # Retry logging user message to new conversation
                    db_service.add_message(conv_id, user_id, "user", query)
                else:
                    raise e
            


            yield "log: Searching SamVidhaan Legal Corpus...\n"
            stream_logs.append("Searching SamVidhaan Legal Corpus...")
            
            chunks = qdrant_service.search(query, top_k=5)
            
            if not chunks:
                stream_logs.append("No relevant documents found.")
                yield "data: {\"answer\": \"No relevant documents found.\", \"chunks\": []}\n"
                
                # Save with metadata
                db_service.add_message(
                    conv_id, 
                    user_id, 
                    "assistant", 
                    "No relevant documents found.",
                    metadata={"logs": stream_logs}
                )
                return
            
            # ... (rest of logic) ...
            # We need to capture the full answer to log it.
            full_answer = []

            yield f"log: Found {len(chunks)} relevant legal documents.\n"
            stream_logs.append(f"Found {len(chunks)} relevant legal documents.")
            
            # Send chunks to frontend immediately
            # Ensure chunks are serializable
            chunks_list = [c for c in chunks]
            stream_chunks = chunks_list 
            chunks_json = json.dumps(chunks_list)
            
            yield f"chunks: {chunks_json}\n"

            # Build context
            context = "\n\n".join([
                f"[Chunk {c['rank']}]\n{c['text']}\nMetadata: {c['metadata']}"
                for c in chunks
            ])
            
            # 2. Hand over to Council Stream
            yield "log: Convening AI Council...\n"
            stream_logs.append("Convening AI Council...")
            
            async for event in council_service.deliberate_stream(query, context):
                # event is like "data: ... \n" or "log: ... \n" or "opinion: ... \n"
                clean_event = event.strip()
                # logger.info(f"Stream Event: {clean_event[:100]}") # Verbose logging

                if clean_event.startswith("log:"):
                    log_msg = clean_event[4:].strip()
                    stream_logs.append(log_msg)
                
                elif clean_event.startswith("opinion:"):
                    try:
                        op_data = json.loads(clean_event[8:].strip())
                        stream_opinions.append(op_data)
                    except Exception as e:
                        logger.error(f"Failed to parse opinion: {e}, Event: {clean_event}")

                # We try to parse data events to capture answer
                elif clean_event.startswith("data:"):
                    try:
                        data_payload = clean_event[5:].strip()
                        data_content = json.loads(data_payload)
                        
                        # If it's a token (for streaming simulation)
                        if "token" in data_content:
                             full_answer.append(data_content["token"])
                        # If it's the final answer (atomic chunk)
                        elif "answer" in data_content:
                             logger.info(f"Captured Final Answer from Stream: {data_content['answer'][:50]}...")
                             full_answer.append(data_content["answer"])
                        elif "error" in data_content:
                             logger.error(f"Stream Error Event: {data_content['error']}")
                    except Exception as e:
                        logger.error(f"Failed to parse data event: {e}, Payload: {clean_event[:50]}")
                
                yield event
            
            # Log full answer with metadata
            final_content = "".join(full_answer)
            
            metadata = {
                "logs": stream_logs,
                "chunks": stream_chunks,
                "council_opinions": stream_opinions
            }
            logger.info(f"[Stream] Saving Message Metadata: {json.dumps(metadata)[:200]}...") # Log summary
            
            db_service.add_message(
                conv_id, 
                user_id, 
                "assistant", 
                final_content or "[Streamed Response]",
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {{\"error\": \"{str(e)}\"}}\n"
            
            # Try to save partial state on error
            try:
                metadata = {
                    "logs": stream_logs,
                    "chunks": stream_chunks,
                    "council_opinions": stream_opinions,
                    "error": str(e)
                }
                db_service.add_message(
                    conv_id, 
                    user_id, 
                    "assistant", 
                    "".join(full_answer) + f"\n[Error: {str(e)}]",
                    metadata=metadata
                )
            except Exception as save_err:
                 logger.error(f"Failed to save error state: {save_err}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")