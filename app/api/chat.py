from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
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
    context_window: int = 5,
    web_search: str = "false", 
    mode: str = "research",
    user_id: str = Depends(get_current_user)
):
    """
    Streaming chat endpoint v3.0 (Clerk + Council)
    """
    async def event_generator():
        try:
            logger.info(f"[Stream] Request from {user_id}: {query}")
            
            # State Containers
            stream_logs = []
            stream_chunks = []
            stream_opinions = []
            full_answer = []
            
            # 1. Conversation Management
            conv_id = conversation_id
            
            # Resolve Web Search Boolean
            is_web_search_enabled = str(web_search).lower() == "true"
            actual_window_size = (max(1, min(50, int(context_window))) * 2) + 1

            # Check history for Retry/Regen logic
            history_for_check = db_service.get_conversation_history(conv_id, user_id, limit=2)
            should_add_user_msg = True
            
            if history_for_check:
                last_msg = history_for_check[-1]
                # Regeneration Logic
                if last_msg.get('role') == 'assistant' and len(history_for_check) >= 2:
                    prev_msg = history_for_check[-2]
                    if prev_msg.get('role') == 'user' and prev_msg.get('content') == query:
                        logger.info(f"[Stream] Regeneration detected. Deleting last response...")
                        db_service.delete_message(last_msg['id'])
                        should_add_user_msg = False
                # Retry Logic
                elif last_msg.get('role') == 'user' and last_msg.get('content') == query:
                     logger.info(f"[Stream] Retry detected. Reusing user message.")
                     should_add_user_msg = False

            # Add User Message to DB
            try:
                if should_add_user_msg:
                    if not conv_id:
                         # Create new conversation if needed
                         title = query[:50] + "..." if len(query) > 50 else query
                         conv_id = db_service.create_conversation(user_id, title=title)
                    
                    db_service.add_message(conv_id, user_id, "user", query)
            except Exception as e:
                # Fallback for FK errors
                # Fallback for FK errors
                 if "foreign key" in str(e) or "23503" in str(e) or not conv_id:
                    # Create with the REQUESTED ID if it exists, otherwise it will generate one
                    conv_id = db_service.create_conversation(user_id, title=query[:50], id=conv_id)
                    db_service.add_message(conv_id, user_id, "user", query)
                 else:
                    raise e
            
            # Fetch Context History
            history = db_service.get_conversation_history(conv_id, user_id, limit=actual_window_size)
            
            # 2. Delegate to Council Service
            # The service now handles Clerk, Retrieval, and Deliberation internally
            
            async for event in council_service.deliberate_stream(
                query=query, 
                chat_history=history, 
                enable_web_search=is_web_search_enabled,
                conv_id=conv_id,
                context_window_size=int(context_window),
                mode=mode  # User's slider value
            ):
                clean_event = event.strip()
                
                # --- Event Handling & Logging ---
                if clean_event.startswith("log:"):
                    log_msg = clean_event[4:].strip()
                    stream_logs.append(log_msg)
                
                elif clean_event.startswith("opinion:"):
                    try:
                        op_data = json.loads(clean_event[8:].strip())
                        stream_opinions.append(op_data)
                    except: pass

                elif clean_event.startswith("chunks:"):
                    try:
                        chunk_data = json.loads(clean_event[7:].strip())
                        stream_chunks = chunk_data # Replace or extend? Usually replace for unique set
                    except: pass
                
                elif clean_event.startswith("data:"):
                    try:
                        data_payload = json.loads(clean_event[5:].strip())
                        if "answer" in data_payload:
                             full_answer.append(data_payload["answer"])
                        if "error" in data_payload:
                             logger.error(f"Stream Error: {data_payload['error']}")
                    except: pass
                
                # Pass through to client
                yield event

            # 3. Save Final State to DB
            final_content = "".join(full_answer)
            metadata = {
                "logs": stream_logs,
                "chunks": stream_chunks,
                "council_opinions": stream_opinions
            }
            
            db_service.add_message(
                conv_id, 
                user_id, 
                "assistant", 
                final_content or "[No Response]",
                metadata=metadata
            )

        except asyncio.CancelledError:
            logger.warning(f"[Stream] Client cancelled conversation {conversation_id}")
            raise
        except Exception as e:
            logger.error(f"[Stream] Error: {e}", exc_info=True)
            yield f"data: {{\"error\": \"{str(e)}\"}}\n"
        finally:
            logger.info("[Stream] Closed.")

    return StreamingResponse(
        event_generator(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )