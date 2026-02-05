import asyncio
import httpx
import json
from typing import List, Dict, Any
from app.config import settings
from app.logger import logger
from app.services.clerk import clerk_service, IntentType
from app.services.qdrant import qdrant_service
from app.models.schemas import ChatMode

class CouncilService:
    """
    AI Council Service v3.0
    - Integrated Clerk (Router)
    - Dual Retrieval (Statutes & Cases)
    - User-Controlled Web Search (Global)
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        # Load Model Configs
        self.MODEL_CONSTITUTIONAL = settings.MODEL_STATUTORY # Reuse flash for now or add config
        self.MODEL_STATUTORY = settings.MODEL_STATUTORY
        self.MODEL_CASE_LAW = settings.MODEL_CASE_LAW
        self.MODEL_DEVIL = settings.MODEL_DEVIL
        self.MODEL_CHAIRMAN = settings.MODEL_CHAIRMAN
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Council service will fail.")

    async def _call_gemini(self, model: str, system_prompt: str, user_query: str, enable_search: bool = False) -> str:
        """Helper to call Gemini REST API"""
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        
        # Base Prompt
        contents = [{"parts": [{"text": f"SYSTEM INSTRUCTION: {system_prompt}\n\n{user_query}"}]}]
        
        # Tools Config
        tools = []
        if enable_search:
             tools.append({"googleSearch": {}})
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": settings.TEMPERATURE_COUNCIL}
        }
        
        if tools:
            payload["tools"] = tools
        
        async with httpx.AsyncClient(timeout=40.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Gemini API Error ({response.status_code}): {response.text}")
                response.raise_for_status()
            
            data = response.json()
            try:
                # Handle cases where Search tool returns 'groundingMetadata' but content is in a different structure
                # Usually standard candidates[0].content.parts[0].text works
                return data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                logger.error(f"Gemini Malformed Response: {data}")
                return "Error generating response."

    async def _get_member_opinion(self, role: str, model: str, system_prompt: str, user_query: str, context: str, enable_search: bool) -> Dict[str, str]:
        """Async worker to get a single council member's opinion"""
        try:
            logger.info(f"[{role}] Deliberating (Search={enable_search})...")
            
            # Context Injection
            full_prompt = f"""
            LEGAL CONTEXT:
            {context}

            USER QUERY:
            {user_query}
            
            Keep your opinion focused on your specific role. 
            If Web Search is enabled, verify facts if the context is insufficient.
            """
            
            opinion = await self._call_gemini(model, system_prompt, full_prompt, enable_search=enable_search)

            return {
                "role": role,
                "model": model,
                "opinion": opinion,
                "web_search_enabled": enable_search
            }

        except Exception as e:
            logger.warning(f"[{role}] Absented due to error: {str(e)}")
            return None

    async def _get_chairman_ruling(self, query: str, context: str, opinions: List[Dict[str, str]], enable_search: bool) -> str:
        """The Chairman synthesizes all opinions into a final answer"""
        valid_opinions = [op for op in opinions if op is not None]
        
        opinions_text = "\n\n".join([
            f"=== OPINION FROM {op['role']} ===\n{op['opinion']}"
            for op in valid_opinions
        ])

        system_prompt = settings.PROMPT_CHAIRMAN

        user_prompt = f"""
        QUERY: {query}
        
        RETRIEVED CONTEXT:
        {context[:settings.CONTEXT_MAX_CHARS]} 
        
        COUNCIL OPINIONS:
        {opinions_text}
        
        FINAL RULING:
        """
        
        try:
            return await self._call_gemini(self.MODEL_CHAIRMAN, system_prompt, user_prompt, enable_search=enable_search)
        except Exception as e:
            logger.error(f"[Chairman] Failed: {e}")
            return "The Chairman could not issue a ruling due to technical difficulties."

    async def deliberate_stream(self, query: str, chat_history: List[Dict] = [], enable_web_search: bool = False, conv_id:str = None, context_window_size: int = 5, mode: str = "research"):
        """
        Main Streaming Pipeline v3.0
        1. Clerk (Classify/Route)
        2. Retrieval (Parallel)
        3. Council (Parallel)
        4. Chairman (Synthesis)
        """
        
        # --- STEP 1: THE CLERK ---
        loop = asyncio.get_running_loop()
        
        logger.info(f"[Council] ========== PIPELINE START ==========")
        logger.info(f"[Council] Query: {query}")
        logger.info(f"[Council] Conversation ID: {conv_id}")
        logger.info(f"[Council] Context Window Size (User Setting): {context_window_size}")
        logger.info(f"[Council] History Messages Retrieved: {len(chat_history)}")
        logger.info(f"[Council] Web Search: {enable_web_search}")
        
        yield f"log: [STEP 1/4] Clerk analyzing query intent...\n"
        yield f"log: Context window: {context_window_size} (retrieved {len(chat_history)} messages)\n"
        
        clerk_resp = await clerk_service.classify_and_route(query, chat_history, enable_web_search=enable_web_search, mode=mode)
        
        if not clerk_resp.is_legal:
            # NON-LEGAL BYPASS
            # Backend: Detailed logging
            logger.info(f"[Council] NON-LEGAL Query Detected")
            logger.info(f"[Council] Query: {query}")
            logger.info(f"[Council] Context Window: {len(chat_history)} messages")
            logger.info(f"[Council] Web Search Enabled: {enable_web_search}")
            if chat_history:
                logger.info(f"[Council] Recent History Glimpse:")
                for msg in chat_history[-3:]:  # Last 3 messages
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:100]  # First 100 chars
                    logger.info(f"  - {role.upper()}: {content}...")
            
            # Frontend: Concise stream logs
            yield f"log: ✓ Clerk classified as NON-LEGAL query\n"
            yield f"log: Context window: {context_window_size} (retrieved {len(chat_history)} messages)\n"
            
            if enable_web_search:
                yield f"log: Web Search: ENABLED (using Gemini's grounding)\n"
            
            yield f"log: Generating direct response (bypassing legal council)...\n"
            answer = clerk_resp.direct_answer or "I cannot answer this legal query."
            
            
            # Backend: Log the response
            logger.info(f"[Council] Direct Answer Length: {len(answer)} chars")
            logger.info(f"[Council] Direct Answer Preview: {answer[:200]}...")
            
            # Send Final Answer Event
            import json
            yield f"data: {json.dumps({'answer': answer})}\n"
            return

        # LEGAL QUERY HANDLING
        rewritten_query = clerk_resp.rewritten_query
        intents = clerk_resp.search_intents
        
        # --- MODE-BASED OPTIMIZATION ---
        # NOTE: Intent filtering removed to support full retrieval in all modes as per redesign.
        # Decisions are now handled by Clerk natively or via model complexity differences.
        
        # if mode == ChatMode.FAST or mode == ChatMode.BALANCED:
        #     ... (Removed)


        # elif mode == ChatMode.RESEARCH:
        #      # Logic handled by Clerk natively now
        #      pass

        # Backend: Detailed legal path logging
        logger.info(f"[Council] ========== LEGAL PATH INITIATED ==========")
        logger.info(f"[Council] Mode: {mode}")
        logger.info(f"[Council] Original Query: {query}")
        logger.info(f"[Council] Rewritten Query: {rewritten_query}")
        logger.info(f"[Council] Search Intents: {[i.value for i in intents]}")
        logger.info(f"[Council] Context Window: {len(chat_history)} messages")
        logger.info(f"[Council] Web Search: {enable_web_search}")
        
        # Frontend: User-friendly logging
        yield f"log: ✓ Clerk classified as LEGAL query\n"
        yield f"log: Intent: {', '.join([i.value.replace('search_', '').title() for i in intents])}\n"
        
        if enable_web_search:
            yield f"log: Web Search: ENABLED (Gemini grounding active)\n"
        else:
            yield f"log: Web Search: DISABLED (using local database only)\n"
        
        if rewritten_query != query:
            yield f"log: Query optimized for better search results\n"
            logger.info(f"[Council] Query Optimization Applied")

        # --- STEP 2: PARALLEL RETRIEVAL ---
        logger.info(f"[Council] ========== STEP 2: DATABASE RETRIEVAL ==========")
        yield f"log: \n"
        yield f"log: [STEP 2/4] Searching Legal Databases...\n"
        
        # Define tasks
        search_tasks = []
        search_types = []
        
        if IntentType.SEARCH_STATUTES in intents or IntentType.SEARCH_BOTH in intents:
            logger.info(f"[Council] Scheduling Statutes Search (Qdrant Collection: indian_legal_docs)")
            logger.info(f"[Council] Search Query: {rewritten_query}")
            logger.info(f"[Council] Top-K: {settings.RAG_TOP_K}")
            search_tasks.append(loop.run_in_executor(None, qdrant_service.search_statutes, rewritten_query))
            search_types.append("Statutes")
            yield f"log: → Querying Statutes database (Indian Penal Code, CrPC, etc.)\n"
        else:
            search_tasks.append(asyncio.sleep(0)) # Dummy

        if IntentType.SEARCH_CASES in intents or IntentType.SEARCH_BOTH in intents:
            logger.info(f"[Council] Scheduling Cases Search (Qdrant Collection: supreme_court_cases)")
            logger.info(f"[Council] Search Query: {rewritten_query}")
            logger.info(f"[Council] Top-K: {settings.RAG_TOP_K}")
            search_tasks.append(loop.run_in_executor(None, qdrant_service.search_cases, rewritten_query))
            search_types.append("Cases")
            yield f"log: → Querying Case Law database (Supreme Court precedents)\n"
        else:
            search_tasks.append(asyncio.sleep(0)) # Dummy

        # Execute Search
        logger.info(f"[Council] Executing {len([t for t in search_types])} parallel search tasks...")
        yield f"log: Executing parallel vector similarity search...\n"
        
        raw_results = await asyncio.gather(*search_tasks)
        
        statute_chunks = raw_results[0] if isinstance(raw_results[0], list) else []
        case_chunks = raw_results[1] if isinstance(raw_results[1], list) else []
        
        # Backend: Detailed retrieval results
        logger.info(f"[Council] Retrieval Results:")
        logger.info(f"  - Statutes: {len(statute_chunks)} documents")
        if statute_chunks:
            for i, chunk in enumerate(statute_chunks[:3]):
                logger.info(f"    [{i+1}] {chunk['metadata'].get('law', 'Unknown')} (Score: {chunk.get('score', 0):.3f})")
        
        logger.info(f"  - Cases: {len(case_chunks)} documents")
        if case_chunks:
            for i, chunk in enumerate(case_chunks[:3]):
                logger.info(f"    [{i+1}] {chunk['metadata'].get('case_name', 'Unknown')} (Score: {chunk.get('score', 0):.3f})")
        
        total_docs = len(statute_chunks) + len(case_chunks)
        logger.info(f"[Council] Total Retrieved: {total_docs} documents")
        
        # Frontend: Concise results
        yield f"log: ✓ Found {len(statute_chunks)} Statutes, {len(case_chunks)} Case Precedents\n"
        if total_docs > 0:
            avg_score = sum([c.get('score', 0) for c in statute_chunks + case_chunks]) / total_docs
            yield f"log: Average relevance score: {avg_score:.1%}\n"
        
        # Send Chunks to UI (Merged list)
        all_chunks = statute_chunks + case_chunks
        # Re-rank based on scores broadly? Or just concat. Concat is fine for display.
        # Assign unified ranks for display
        for i, c in enumerate(all_chunks):
            c['rank'] = i + 1
            
        import json
        yield f"chunks: {json.dumps(all_chunks)}\n"

        # Context Formatting
        def format_context(chunks):
            return "\n\n".join([f"[Source: {c['metadata'].get('title')}]\n{c['text']}" for c in chunks])
            
        statute_ctx = format_context(statute_chunks)
        case_ctx = format_context(case_chunks)
        full_ctx = statute_ctx + "\n\n" + case_ctx

        # --- STEP 3: COUNCIL DELIBERATION (OR FAST BYPASS) ---
        logger.info(f"[Council] ========== STEP 3: COUNCIL DELIBERATION ==========")
        yield f"log: \n"
        
        # === FAST MODE: SINGLE SHOT DIRECT ===
        if mode == ChatMode.FAST:
            yield f"log: [STEP 3/3] Generating Fast Answer (Direct Mode)...\n"
            
            # Simple Prompt
            system_prompt = "You are a legal assistant. Answer the user query based on the provided context. Be concise and accurate."
            user_prompt = f"QUERY: {rewritten_query}\n\nCONTEXT:\n{full_ctx}" # Use full_ctx (Statutes + Cases)
            
            yield f"log: Generating direct response from retrieved documents...\n"
            
            # Use unified helper
            async for event in self._generate_and_stream_response(settings.MODEL_CLERK, system_prompt, user_prompt, enable_web_search):
                yield event
            return

        # === BALANCED MODE: REASONING ONE-SHOT ===
        if mode == ChatMode.BALANCED:
            yield f"log: [STEP 3/3] Generating Balanced Analysis (Reasoning Mode)...\n"
            
            # Chain of Thought Prompt
            system_prompt = settings.PROMPT_CHAIRMAN + "\n\nINSTRUCTION: You are acting as the sole legal authority. Analyze the User Query against the Retrieved Context (Statutes and Case Law). Considerations: 1. Constitutional validity. 2. Statutory interpretation. 3. Precedents. Formulate a balanced and legally sound answer."
            user_prompt = f"QUERY: {rewritten_query}\n\nRETRIEVED CONTEXT:\n{full_ctx}"
            
            yield f"log: Analyzing context with Chairman model (One-Shot)...\n"
            
            # Use unified helper
            async for event in self._generate_and_stream_response(self.MODEL_CHAIRMAN, system_prompt, user_prompt, enable_web_search):
                yield event
            return

        yield f"log: [STEP 3/4] Convening AI Legal Council (Deep Mode)...\n"
        
        # Define Member Tasks with Specialized Context
        tasks = []
        council_members = []
        
        # Backend: Log context preparation
        logger.info(f"[Council] Preparing context for council members:")
        logger.info(f"  - Full Context Length: {len(full_ctx)} chars")
        logger.info(f"  - Statute Context Length: {len(statute_ctx)} chars")
        logger.info(f"  - Case Context Length: {len(case_ctx)} chars")
        
        # 1. Constitutional Expert (Needs Broad Context)
        logger.info(f"[Council] Assigning Constitutional Expert (Model: {self.MODEL_CONSTITUTIONAL})")
        tasks.append(self._get_member_opinion(
            "Constitutional Expert", self.MODEL_CONSTITUTIONAL,
            settings.PROMPT_CONSTITUTIONAL,
            rewritten_query, full_ctx, enable_web_search
        ))
        council_members.append("Constitutional Expert")
        yield f"log: → Constitutional Expert analyzing fundamental rights & validity\n"
        
        # 2. Statutory Analyst
        if statute_chunks:
            logger.info(f"[Council] Assigning Statutory Analyst (Model: {self.MODEL_STATUTORY})")
            logger.info(f"  - Statute Context: {len(statute_chunks)} documents")
            tasks.append(self._get_member_opinion(
                "Statutory Analyst", self.MODEL_STATUTORY,
                settings.PROMPT_STATUTORY,
                rewritten_query, statute_ctx, enable_web_search
            ))
            council_members.append("Statutory Analyst")
            yield f"log: → Statutory Analyst examining legal provisions & penalties\n"
        else:
            logger.info(f"[Council] Skipping Statutory Analyst (no statute chunks)")
            
        # 3. Case Law Researcher
        # NOTE: Always active in Deep/Research Mode
        if (case_chunks or enable_web_search):
            logger.info(f"[Council] Assigning Case Law Researcher (Model: {self.MODEL_CASE_LAW})")
            logger.info(f"  - Case Context: {len(case_chunks)} documents")
            tasks.append(self._get_member_opinion(
                "Case Law Researcher", self.MODEL_CASE_LAW,
                settings.PROMPT_CASE_LAW,
                rewritten_query, case_ctx, enable_web_search
            ))
            council_members.append("Case Law Researcher")
            yield f"log: → Case Law Researcher reviewing precedents & judgments\n"
        else:
            logger.info(f"[Council] Skipping Case Law Researcher (no cases/search disabled)")
             
        # 4. Devil's Advocate
        logger.info(f"[Council] Assigning Devil's Advocate (Model: {self.MODEL_DEVIL})")
        tasks.append(self._get_member_opinion(
            "Devil's Advocate", self.MODEL_DEVIL,
            settings.PROMPT_DEVIL,
            rewritten_query, full_ctx, enable_web_search
        ))
        council_members.append("Devil's Advocate")
        yield f"log: → Devil's Advocate identifying counterarguments & loopholes\n"
        
        logger.info(f"[Council] Total Council Members: {len(council_members)}")
        logger.info(f"[Council] Members: {', '.join(council_members)}")
        yield f"log: Council size: {len(council_members)} expert members\n"
        yield f"log: Awaiting parallel deliberations...\n"
        
        # Execute Council
        completed_opinions = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                completed_opinions.append(result)
                opinion_length = len(result.get('opinion', ''))
                logger.info(f"[Council] {result['role']} completed (Opinion: {opinion_length} chars)")
                yield f"opinion: {json.dumps(result)}\n"
                yield f"log: ✓ {result['role']} submitted opinion ({opinion_length} chars)\n"

        # --- STEP 4: CHAIRMAN RULING ---
        logger.info(f"[Council] ========== STEP 4: CHAIRMAN SYNTHESIS ==========")
        
        if not completed_opinions:
            logger.error(f"[Council] CRITICAL: No opinions received from council")
            yield f"log: ✗ Error: Council failed to deliberate\n"
            yield f"data: {json.dumps({'error': 'Council failed to deliberate.'})}\n"
            return

        logger.info(f"[Council] Received {len(completed_opinions)} opinions")
        logger.info(f"[Council] Opinion Summary:")
        for op in completed_opinions:
            logger.info(f"  - {op['role']}: {len(op.get('opinion', ''))} chars")
        
        yield f"log: \n"
        yield f"log: [STEP 4/4] Chairman synthesizing final ruling...\n"
        yield f"log: Analyzing {len(completed_opinions)} expert opinions\n"
        
        
        # Calculate total context being sent to Chairman
        total_context_chars = len(full_ctx) + sum(len(op.get('opinion', '')) for op in completed_opinions)
        logger.info(f"[Council] Total Context for Chairman: {total_context_chars} chars")
        logger.info(f"  - Retrieved Documents: {len(full_ctx)} chars")
        logger.info(f"  - Council Opinions: {sum(len(op.get('opinion', '')) for op in completed_opinions)} chars")
        logger.info(f"[Council] Calling Chairman (Model: {self.MODEL_CHAIRMAN})")
        
        yield f"log: Context size: {total_context_chars:,} characters\n"
        yield f"log: Generating comprehensive legal analysis (Final Ruling)...\n"
        
        # Prepare Prompt for Chairman (reconstruct prompt logic from _get_chairman_ruling)
        valid_opinions = [op for op in completed_opinions]
        opinions_text = "\n\n".join([
            f"=== OPINION FROM {op['role']} ===\n{op['opinion']}"
            for op in valid_opinions
        ])
        
        system_prompt = settings.PROMPT_CHAIRMAN
        user_prompt = f"""
        QUERY: {rewritten_query}
        
        RETRIEVED CONTEXT:
        {full_ctx[:settings.CONTEXT_MAX_CHARS]} 
        
        COUNCIL OPINIONS:
        {opinions_text}
        
        FINAL RULING:
        """
        
        # Use unified helper
        async for event in self._generate_and_stream_response(self.MODEL_CHAIRMAN, system_prompt, user_prompt, enable_web_search):
            yield event

        logger.info(f"[Council] ========== PIPELINE COMPLETE ==========")
        
        yield f"log: ✓ Complete.\n"


    async def _stream_call_gemini(self, model: str, system_prompt: str, user_query: str, enable_search: bool = False):
        """Helper to call Gemini REST API with Streaming"""
        url = f"{self.base_url}/{model}:streamGenerateContent?key={self.api_key}&alt=sse"
        
        # Base Prompt
        contents = [{"parts": [{"text": f"SYSTEM INSTRUCTION: {system_prompt}\n\n{user_query}"}]}]
        
        # Tools Config
        tools = []
        if enable_search:
             tools.append({"googleSearch": {}})
        
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": settings.TEMPERATURE_CHAIRMAN}
        }
        
        if tools:
            payload["tools"] = tools
            
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(f"Gemini Streaming Error ({response.status_code}): {error_body.decode()}")
                    yield f"[Error: {response.status_code}]"
                    return

                try:
                    # Parse SSE-like JSON stream
                    async for line in response.aiter_lines():
                        if not line: continue
                        
                        if line.startswith("data:"):
                            try:
                                json_str = line[5:].strip()
                                if not json_str: continue
                                
                                data = json.loads(json_str)
                                if "candidates" in data:
                                    candidate = data["candidates"][0]
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        text_chunk = candidate["content"]["parts"][0].get("text", "")
                                        if text_chunk:
                                            yield text_chunk
                            except Exception as e:
                                pass
                except Exception as e:
                     logger.error(f"Stream Consumption Error: {e}")
                     pass
            
            
    # async def _generate_followup_questions(self, context_text: str) -> List[str]:
    #     """
    #     REMOVED: Integrated into Single Pass Stream.
    #     """
    #     return [] 
        
    # --- END OF REPLACEMENT CHUNK PLANNING ---
    
    # Let's do it in smaller chunks. 
    # FIRST: Add the helper methods to the CLASS.
    # SECOND: Replace Step 4 in `deliberate_stream`.


        # Legacy method kept for interface compatibility if needed, 
        # but now handled by Clerk.
        pass

    async def _generate_and_stream_response(self, model: str, system_prompt: str, user_prompt: str, enable_search: bool) -> None:
        """
        Unified helper with Stream Splitting logic to extracting follow-ups from a single pass.
        """
        full_answer = ""
        separator = settings.FOLLOWUP_SEPARATOR
        separator_len = len(separator)
        
        # Inject Follow-up Instruction
        # We inject it into BOTH system and user prompt to ensure adherence
        system_prompt += settings.PROMPT_FOLLOWUP_INSTRUCTION
        user_prompt_with_instruction = f"{user_prompt}\n\n{settings.PROMPT_FOLLOWUP_INSTRUCTION}"
        
        buffer = ""
        found_separator = False
        followup_buffer = ""
        
        # DEBUG: Track token count
        token_count = 0
        
        async for chunk in self._stream_call_gemini(model, system_prompt, user_prompt_with_instruction, enable_search=enable_search):
            # logger.info(f"[StreamDebug] Chunk received: {len(chunk)} chars")
            buffer += chunk
            
            if found_separator:
                followup_buffer += chunk
                continue
                
            # Check for separator in buffer
            if separator in buffer:
                found_separator = True
                # Split
                parts = buffer.split(separator)
                safe_text = parts[0]
                followup_text = parts[1]
                
                # Yield the safe text (final answer part)
                if safe_text:
                    full_answer += safe_text
                    token_count += 1
                    if token_count == 1:
                        logger.info("[StreamDebug] FIRST TOKEN YIELDED from Splitter")
                    yield f"token: {json.dumps(safe_text)}\n"
                
                logger.info("[StreamDebug] Separator found! Switching to followup collection.")
                
                # Start collecting followups
                followup_buffer = followup_text
                # Clear buffer (not needed, but good practice)
                buffer = "" 
                continue
            
            # Streaming logic with safety buffer
            # We must keep enough chars in buffer to cover partial separator
            # E.g. separator is "+++FOLLOW_UP+++" (15 chars)
            # If buffer is "Safe text... +++FO", we can yield "Safe text... "
            
            if len(buffer) > separator_len:
                # Yield the SAFE part, keep the TAIL
                safe_chunk = buffer[:-separator_len]
                buffer = buffer[-separator_len:] # Keep tail
                
                full_answer += safe_chunk
                token_count += 1
                if token_count == 1:
                     logger.info("[StreamDebug] FIRST TOKEN YIELDED from Buffer")
                yield f"token: {json.dumps(safe_chunk)}\n"
        
        # End of Stream
        logger.info(f"[StreamDebug] Stream ended. Tokens yielded: {token_count}. Separator found: {found_separator}")

        yield f"log: ✓ Response generated.\n"
        
        # If separator was never found (model ignored instruction), flush buffer as text
        if not found_separator and buffer:
             logger.warning("[StreamDebug] Separator NOT found. Flushing remaining buffer.")
             full_answer += buffer
             yield f"token: {json.dumps(buffer)}\n"
             
        # Process Follow-ups
        if followup_buffer:
             logger.info(f"Parsing integrated follow-up questions... Buffer len: {len(followup_buffer)}")
             yield f"log: Parsing integrated follow-up questions...\n"
             questions = [line.strip() for line in followup_buffer.split('\n') if line.strip() and '?' in line]
             # Basic cleanup
             questions = [q.lstrip("1234567890.-•* ") for q in questions][:3]
             
             if questions:
                 logger.info(f"Generated follow-ups: {questions}")
                 yield f"followup: {json.dumps(questions)}\n"
             else:
                 logger.warning("Follow-up buffer had content but no questions parsed.")
                 
        # Final Data Event
        yield f"data: {json.dumps({'answer': full_answer})}\n"


    async def deliberate(self, query: str, context: str = "") -> Dict[str, Any]:
        """
        Legacy/Sync Entry Point.
        Consumes the streaming generator to produce a single final response.
        Ignores 'context' argument in favor of internal Clerk+Retrieval flow.
        """
        final_answer = ""
        opinions = []
        
        # We start a new conversation context for this one-off request
        async for event in self.deliberate_stream(query, chat_history=[], enable_web_search=False):
            clean_event = event.strip()
            
            if clean_event.startswith("data:"):
                import json
                try:
                    data = json.loads(clean_event[5:])
                    if "answer" in data:
                        final_answer += data["answer"] # Although usually atomic in stream
                except: pass
            
            elif clean_event.startswith("opinion:"):
                import json
                try:
                    opinions.append(json.loads(clean_event[8:]))
                except: pass

        return {
            "query": query,
            "answer": final_answer,
            "council_opinions": opinions
        }

council_service = CouncilService()
