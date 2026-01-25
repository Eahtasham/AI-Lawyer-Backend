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
    context_window: int = 5,
    web_search: str = "false", # Boolean passed as string query param often
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
            
            # Resolve Web Search Boolean
            is_web_search_enabled = str(web_search).lower() == "true"
            # Ensure context_window is within bounds (Treat as turns/pairs, plus current msg)
            actual_window_size = (max(1, min(50, int(context_window))) * 2) + 1

            # Fetch History FIRST (before adding new message) to check for Regeneration/Retry
            history_for_check = db_service.get_conversation_history(conv_id, user_id, limit=2) # Get last 2 messages
            
            should_add_user_msg = True
            
            if history_for_check:
                last_msg = history_for_check[-1] # List is Chronological (Old -> New), so -1 is Last
                
                # REGENERATION DETECTED: Last messsage was Assistant, Prev was User matching current query
                if last_msg.get('role') == 'assistant' and len(history_for_check) >= 2:
                    prev_msg = history_for_check[-2]
                    if prev_msg.get('role') == 'user' and prev_msg.get('content') == query:
                        logger.info(f"[Stream] Regeneration detected for conv {conv_id}. Deleting last assistant response...")
                        db_service.delete_message(last_msg['id'])
                        should_add_user_msg = False # Reuse existing user message
                
                # RETRY DETECTED: Last message was User matching current query (AI failed to reply?)
                elif last_msg.get('role') == 'user' and last_msg.get('content') == query:
                     logger.info(f"[Stream] Retry detected for conv {conv_id}. Reusing existing user message.")
                     should_add_user_msg = False

            # Log User Message (only if not deduplicated)
            try:
                if should_add_user_msg:
                    if not conv_id:
                         raise ValueError("No ID provided, force creation")
                    db_service.add_message(conv_id, user_id, "user", query)
            except Exception as e:
                # Error recovery same as before
                is_fk_error = "foreign key constraint" in str(e) or "23503" in str(e) 
                if is_fk_error or not conv_id:
                    logger.warning(f"[Stream] Conversation {conv_id} missing or invalid. Creating new conversation.")
                    title = query[:50] + "..." if len(query) > 50 else query
                    conv_id = db_service.create_conversation(user_id, title=title, id=conv_id)
                    db_service.add_message(conv_id, user_id, "user", query)
                else:
                    raise e
            
            # Fetch History AGAIN (or refresh) for Context/CQR
            # We must fetch fresh history because we might have deleted something or added something.
            history_for_cqr = db_service.get_conversation_history(conv_id, user_id, limit=actual_window_size)
            
            # Debug Log for Context verification (Moved Up)
            logger.info(f"[Context] Fetching last {actual_window_size} messages for conversation {conv_id}")
            for idx, msg in enumerate(history_for_cqr):
                 # Sanitize for logging (Windows cp1252 safety)
                 raw_content = msg.get('content', '')[:50].replace('\n', ' ')
                 content_preview = raw_content.encode('ascii', 'replace').decode('ascii')
                 logger.info(f"   {idx+1}. [{msg.get('role')}] {content_preview}...")
            
            # --- CONTEXTUAL QUERY REWRITING (CQR) ---
            rewritten_query = query
            if history_for_cqr:
                 yield "log: Analyzing conversation context...\n"
                 # Exclude the current message we just added from history to avoid confusion
                 # (Though get_conversation_history might return it if it's already in DB. 
                 #  Let's trust the rewriter to handle it, or filter it out if needed.
                 #  Usually history implies "past messages". Logic check: db.add_message was called above.)
                 # Let's pass the fetched history. The rewriter prompt handles "Latest User Query" separately.
                 
                 # Optimization: Filter out the 'user' message we JUST added if it appears in history
                 # But actually we just want the PAST history. 
                 # get_conversation_history returns latest Last. 
                 # If we just added the message, it might be in the list.
                 # Let's rely on the LLM to understand or just pass it all.
                 
                 rewritten_query = await council_service.rewrite_query(query, history_for_cqr)
                 
                 if rewritten_query != query:
                     yield f"log: Contextualized Query: '{rewritten_query}'\n"
                     stream_logs.append(f"Contextualized Query: '{rewritten_query}'")
            # ----------------------------------------

            yield "log: Searching SamVidhaan Legal Corpus...\n"
            stream_logs.append("Searching SamVidhaan Legal Corpus...")
            
            # Search using REWRITTEN query
            chunks = qdrant_service.search(rewritten_query, top_k=5)
            
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

            context = "\n\n".join([
                f"[Chunk {c['rank']}]\n{c['text']}\nMetadata: {c['metadata']}"
                for c in chunks
            ])
            
            # Reuse the history fetched earlier (history_for_cqr)
            # This avoids double-fetching and ensures consistency.
            history_msgs = history_for_cqr
            
            # Use "Context Window Size" terminology as requested
            yield f"log: Restored conversation context (Window Size: {context_window}).\n"
            stream_logs.append(f"Restored conversation context (Window Size: {context_window}).")
            
            # 2. Hand over to Council Stream
            yield "log: Convening AI Council...\n"
            stream_logs.append("Convening AI Council...")
            
            async for event in council_service.deliberate_stream(query, context, history_msgs, enable_web_search=is_web_search_enabled):
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

        except asyncio.CancelledError:
            logger.info(f"[Stream] Client disconnected (cancelled) for {user_id}")
            # We can optionally save partial state here if desired
            # For now, just exit gracefully so the background tasks (in council.py) get cancelled via GeneratorExit
            raise

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
        finally:
            logger.info("[Stream] Generator closed.")

    return StreamingResponse(event_generator(), media_type="text/event-stream")