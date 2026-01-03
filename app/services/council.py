import asyncio
import httpx
from typing import List, Dict, Any
from app.config import settings
from app.logger import logger

class CouncilService:
    """
    AI Council Service using Google Gemini API directly.
    """

    # Gemini Models (Using variants for diversity)
    MODEL_CONSTITUTIONAL = "gemini-2.0-flash"
    MODEL_STATUTORY = "gemini-2.0-flash"
    MODEL_CASE_LAW = "gemini-2.5-flash"
    MODEL_DEVIL = "gemini-2.0-flash"
    MODEL_CHAIRMAN = "gemini-2.5-flash" 

    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Council service will fail.")

        logger.info("CouncilService initialized with Google Gemini API")

    async def _call_gemini(self, model: str, system_prompt: str, user_query: str, enable_search: bool = False) -> str:
        """Helper to call Gemini REST API"""
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": f"SYSTEM INSTRUCTION: {system_prompt}\n\n{user_query}"}]
            }],
             "generationConfig": {
                "temperature": 0.7
            }
        }
        
        if enable_search:
            payload["tools"] = [{"googleSearch": {}}]
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code != 200:
                logger.error(f"Gemini API Error ({response.status_code}): {response.text}")
                response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]

    async def _get_member_opinion(self, role: str, model: str, system_prompt: str, user_query: str, context: str, enable_search: bool = False) -> Dict[str, str]:
        """Async worker to get a single council member's opinion"""
        # No rate limiting (semaphore removed)
        try:
            logger.info(f"[{role}] Deliberating using {model} (Web Search: {enable_search})...")
            
            if enable_search:
                full_prompt = f"""
                CONTEXT FROM DATABASE:
                {context}

                USER QUERY:
                {user_query}
                
                INSTRUCTIONS:
                1. Check the provided context first.
                2. If the context is missing relevant details (e.g., specific recent events, case laws), YOU MUST USE THE GOOGLE SEARCH TOOL to find the answer.
                3. Do not say "I cannot provide analysis". Search the web and provide a summary based on external facts.
                4. SPECIAL INSTRUCTION: If the user query is about a current event, news, or general topic NOT strictly related to established Indian Law statutes (e.g. sports, politics, gossip), start your response with:
                   "[[NON-LEGAL]]
                   
                   [detailed answer based on search results]"
                """
            else:
                full_prompt = f"""
                CONTEXT FROM INDIAN LAWS:
                {context}

                USER QUERY:
                {user_query}
                
                Based strictly on the context above, provide your analysis.
                """
            
            # Use the search tool if enabled
            opinion = await self._call_gemini(model, system_prompt, full_prompt, enable_search=enable_search)

            logger.info(f"[{role}] Opinion submitted.")
            return {
                "role": role,
                "model": model,
                "opinion": opinion
            }

        except Exception as e:
            logger.warning(f"[{role}] Absented due to error: {str(e)}")
            return None

    async def _get_chairman_ruling(self, query: str, context: str, opinions: List[Dict[str, str]]) -> str:
        """The Chairman synthesizes all opinions into a final answer"""
        valid_opinions = [op for op in opinions if op is not None]
        
        if not valid_opinions:
            return "The AI Council could not convene due to technical errors."
            
        logger.info(f"[Chairman] Reviewing {len(valid_opinions)} council opinions...")

        # CHECK FOR "SPECIAL POWER" (Short-circuit if non-legal)
        for op in valid_opinions:
            if op["role"] == "Case Law Researcher" and "[[NON-LEGAL]]" in op["opinion"]:
                logger.info(f"[Chairman] Detected non-legal query via Case Law Researcher. Raw opinion length: {len(op['opinion'])}")
                clean_opinion = op["opinion"].replace("[[NON-LEGAL]]", "").strip()
                if not clean_opinion:
                     logger.warning("[Chairman] Cleaned opinion is empty! Falling back to raw opinion.")
                     clean_opinion = op["opinion"] # Fallback if regex/replace killed it
                
                return f"**Special Direct Ruling (Non-Legal Inquiry):**\n\n{clean_opinion}"

        opinions_text = "\n\n".join([
            f"=== OPINION FROM {op['role']} ({op['model']}) ===\n{op['opinion']}"
            for op in valid_opinions
        ])

        system_prompt = """You are the Chief Justice and Chairman of the AI Legal Council. 
        Your goal is to provide the most accurate, balanced, and legally sound answer to the user's query.
        
        1. Review the opinions submitted by your council members.
        2. Resolve any conflicts between them.
        3. Formulate a final, authoritative response citing specific sections from the context.
        4. Focus on INDIAN LAW.
        """

        user_prompt = f"""
        ORIGINAL QUERY: {query}
        
        RETRIEVED CONTEXT:
        {context}
        
        COUNCIL DELIBERATIONS:
        {opinions_text}
        
        Provide your Final Ruling below:
        """

        try:
            return await self._call_gemini(self.MODEL_CHAIRMAN, system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"[Chairman] Failed to rule: {str(e)}")
            return "The Chairman is currently unavailable."

    async def deliberate(self, query: str, context: str) -> Dict[str, Any]:
        """Main entry point"""
        
        tasks = [
            self._get_member_opinion(
                role="Constitutional Expert", 
                model=self.MODEL_CONSTITUTIONAL,
                system_prompt="You are a Constitutional Law expert. Analyze the query based on the Constitution of India. Prioritize Fundamental Rights and constitutional validity.",
                user_query=query, 
                context=context
            ),
            self._get_member_opinion(
                role="Statutory Analyst", 
                model=self.MODEL_STATUTORY,
                system_prompt="You are a Black-letter law expert. Focus strictly on the text, definitions, and penalties in the provided Acts (IPC, CrPC, BNS). Be literal and precise.",
                user_query=query, 
                context=context
            ),
            self._get_member_opinion(
                role="Case Law Researcher", 
                model=self.MODEL_CASE_LAW,
                system_prompt="You are a Case Law specialist. Look for relevant precedents and court rulings on the web if not in context. Identify landmark judgments.",
                user_query=query, 
                context=context,
                enable_search=True  # ENABLE WEB SEARCH HERE
            ),
            self._get_member_opinion(
                role="Devil's Advocate", 
                model=self.MODEL_DEVIL,
                system_prompt="You are the Devil's Advocate. Your job is to find loopholes, defenses, exceptions, or alternative interpretations.",
                user_query=query, 
                context=context
            )
        ]

        logger.info("Council session started (Gemini). Members deliberating...")
        opinions = await asyncio.gather(*tasks)
        
        valid_opinions = [op for op in opinions if op is not None]
        final_answer = await self._get_chairman_ruling(query, context, valid_opinions)
        
        return {
            "query": query,
            "answer": final_answer,
            "council_opinions": valid_opinions
        }

council_service = CouncilService()
