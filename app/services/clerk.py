import httpx
import json
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.config import settings
from app.logger import logger

class IntentType(str, Enum):
    SEARCH_STATUTES = "search_statutes"
    SEARCH_CASES = "search_cases"
    SEARCH_BOTH = "search_both"

class ClerkResponse(BaseModel):
    rewritten_query: str
    is_legal: bool
    direct_answer: Optional[str] = None
    search_intents: List[IntentType] = []

class ClerkService:
    """
    The Clerk (Router Agent)
    Responsibilities:
    1. CQR (Contextual Query Rewriting)
    2. Classification (Legal vs Non-Legal)
    3. Routing (Statutes vs Cases vs Both)
    """

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        self.model = settings.MODEL_CLERK
        logger.info(f"ClerkService initialized with model: {self.model}")

    async def _call_gemini_flash(self, system_prompt: str, user_prompt: str, enable_web_search: bool = False) -> str:
        """Lightweight call to Gemini Flash"""
        url = f"{self.base_url}/{self.model}:generateContent?key={self.api_key}"
        
        # Dynamic Temperature: Higher for creative/search-based answers, Lower for strict routing
        temperature = settings.TEMPERATURE_CLERK_SEARCH if enable_web_search else settings.TEMPERATURE_CLERK
        
        
        config = {
            "temperature": temperature
        }
        
        # Check against model name for tools (Gemma often doesn't support the same tools API or we want to keep it simple)
        is_gemma = "gemma" in self.model.lower()

        payload = {
            "contents": [{"parts": [{"text": f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"}]}],
            "generationConfig": config
        }
        
        # Add Web Search Tool if enabled -- BUT ONLY FOR GEMINI MODELS
        if enable_web_search and not is_gemma:
            payload["tools"] = [{"googleSearch": {}}]

        async with httpx.AsyncClient(timeout=15.0) as client: # Increased timeout for search
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"[Clerk] Gemini Error ({response.status_code}): {response.text}")
                return "{}" # Fail safe
            
            data = response.json()
            try:
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"]
                
                # Cleanup Markdown if present (Gemma often wraps in ```json)
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                     raw_text = raw_text.split("```")[1].strip()
                
                return raw_text
            except (KeyError, IndexError):
                logger.error(f"[Clerk] Malformed response: {data}")
                return "{}"

    async def classify_and_route(self, query: str, history: List[Dict], enable_web_search: bool = False, mode: str = "research") -> ClerkResponse:
        """
        Main entry point for the Clerk.
        Processing Steps:
        1. Contextualize the query based on history.
        2. Classify intent.
        3. Return structured routing decision.
        """
        
        # Prepare History Context
        history_text = "NO PREVIOUS HISTORY"
        if history:
            # We trust the history passed to us (already filtered by chat endpoint)
            history_text = "\n".join([f"{msg.get('role', 'unknown').upper()}: {msg.get('content', '')}" for msg in history])

        system_prompt = settings.PROMPT_CLERK
        
        user_prompt = f"""
        HISTORY:
        {history_text}
        
        CURRENT QUERY:
        {query}
        
        USER SETTING:
        Mode: {mode.upper()}
        (If Mode is RESEARCH, you should bias towards 'search_both' to ensure comprehensive coverage unless the query is extremely simple or irrelevant to case law.)
        """
        
        try:
            logger.info(f"[Clerk] ========== Processing Query ==========")
            logger.info(f"[Clerk] Query: {query}")
            
            # Log as Turns (approx pairs / 2)
            # History includes user+assistant pairs.
            turns_count = len(history) // 2
            logger.info(f"[Clerk] Context Window: {len(history)} messages (~{turns_count} turns)")
            logger.info(f"[Clerk] Web Search Enabled: {enable_web_search}")
            
            if history:
                logger.info(f"[Clerk] History Context Preview:")
                # Show last few messages for debug
                for i, msg in enumerate(history[-4:]):
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')[:80]
                    logger.info(f"  [{len(history)-3+i}] {role.upper()}: {content}...")
            else:
                logger.info(f"[Clerk] No previous history")
            
            logger.info(f"[Clerk] Sending to Gemini Flash API...")
            logger.info(f"[Clerk] System Prompt Length: {len(system_prompt)} chars")
            logger.info(f"[Clerk] User Prompt Length: {len(user_prompt)} chars")
            raw_response = await self._call_gemini_flash(system_prompt, user_prompt, enable_web_search)
            data = json.loads(raw_response)
            
            # Validation / Fallback
            is_legal = data.get("is_legal", False)
            rewritten = data.get("rewritten_query", query)
            direct_answer = data.get("direct_answer")
            intents_str = data.get("search_intents", [])
            
            # Map strings to Enum
            valid_intents = []
            if is_legal:
                for idx in intents_str:
                    if idx in ["search_statutes", "search_cases", "search_both"]:
                        valid_intents.append(IntentType(idx))
                
                # Default to BOTH if legal but no intent found (Safety)
                if not valid_intents:
                    valid_intents = [IntentType.SEARCH_BOTH]
            
            logger.info(f"[Clerk] Decision: Legal={is_legal}, Intent={valid_intents}, Rewritten='{rewritten}'")
            
            return ClerkResponse(
                rewritten_query=rewritten,
                is_legal=is_legal,
                direct_answer=direct_answer,
                search_intents=valid_intents
            )
            
        except Exception as e:
            logger.error(f"[Clerk] Critical Failure: {e}")
            logger.error(f"[Clerk] Failed Raw Response: {raw_response}")
            # Fail safe: Treat as Legal Scan Both to be safe, or just return original
            return ClerkResponse(
                rewritten_query=query,
                is_legal=True, # Safety default
                search_intents=[IntentType.SEARCH_BOTH]
            )

clerk_service = ClerkService()
