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
    MODEL_CHAIRMAN = "gemini-2.5-pro" 

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
                4. SPECIAL INSTRUCTION: If the user query is about a current event, news, or general topic NOT strictly related to established Indian Law statutes (e.g. sports, politics, gossip), you MUST:
                   - Use the Google Search tool to find the answer.
                   - Start your response with "[[NON-LEGAL]]".
                   - IMMEDIATELY followed by a detailed answer to the user's query based on your search.
                   
                   Example Format:
                   [[NON-LEGAL]]
                   According to recent news, the event ...
                   
                   WARNING: Do NOT verify if it is legal or not in the text. Just answer the question. Do NOT output the tag alone.
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
                if len(clean_opinion) < 5:
                     logger.warning("[Chairman] Cleaned opinion is empty! Ignoring non-legal short-circuit.")
                     continue
                
                return f"**Special Direct Ruling (Non-Legal Inquiry):**\n\n{clean_opinion}"

        opinions_text = "\n\n".join([
            f"=== OPINION FROM {op['role']} ({op['model']}) ===\n{op['opinion']}"
            for op in valid_opinions
        ])

        system_prompt = """You are the Chief Justice and Chairman of the AI Legal Council. 
        Your goal is to provide the most accurate, relevant, balanced, concise and legally sound answer to the user's query.

        CRITICAL INSTRUCTIONS - READ CAREFULLY:
        1. **ABSOLUTELY NO INTRODUCTIONS**: Never say "As the Chief Justice", "I have reviewed...", "The council has deliberated...", or "Here is the ruling".
        2. **START IMMEDIATELY WITH THE ANSWER**: Begin your response directly with the legal analysis or answer.
        3. **FORMATTING**: Use Markdown headings (#, ##) to structure the response. separating sections with distinct whitespace.
        4. **TONE**: authoritative, objective, and direct.
        5. **CONTEXT**: Focus strictly on INDIAN LAW.
        
        Example of how NOT to start:
        "As the Chairman, I have..." (WRONG)
        "Based on the opinions..." (WRONG)
        
        Example of how to start:
        "**The Law on [Topic]**..." (RIGHT)
        "Under Section X of the IPC..." (RIGHT)
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


    async def deliberate_stream(self, query: str, context: str):
        """Streaming entry point for SSE"""
        
        # 1. Define member tasks
        member_prompts = [
            ("Constitutional Expert", self.MODEL_CONSTITUTIONAL, 
             "You are a Constitutional Law expert. Analyze the query based on the Constitution of India. Prioritize Fundamental Rights and constitutional validity.", False),
            ("Statutory Analyst", self.MODEL_STATUTORY,
             "You are a Black-letter law expert. Focus strictly on the text, definitions, and penalties in the provided Acts (IPC, CrPC, BNS). Be literal and precise.", False),
            ("Case Law Researcher", self.MODEL_CASE_LAW,
             "You are a Case Law specialist. Look for relevant precedents and court rulings on the web if not in context. Identify landmark judgments.", True),
            ("Devil's Advocate", self.MODEL_DEVIL,
             "You are the Devil's Advocate. Your job is to find loopholes, defenses, exceptions, or alternative interpretations.", False)
        ]

        tasks = []
        for role, model, sys_prompt, search in member_prompts:
            tasks.append(
                asyncio.create_task(
                    self._get_member_opinion(role, model, sys_prompt, query, context, enable_search=search)
                )
            )
            # Yield initial log for each member
            yield f"log: {role} has started review...\n"

        logger.info("[Stream] Council members dispatched.")

        # 2. Wait for members to finish as they complete
        valid_opinions = []
        
        for completed_task in asyncio.as_completed(tasks):
            result = await completed_task
            if result:
                valid_opinions.append(result)
                # Yield opinion event
                import json
                yield f"opinion: {json.dumps(result)}\n"
                yield f"log: {result['role']} has submitted their opinion.\n"
        
        # 3. Chairman Ruling
        if not valid_opinions:
            yield f"data: {json.dumps({'error': 'The AI Council could not convene due to technical errors.'})}\n"
            return

        yield "log: Chairman is reviewing all opinions...\n"
        
        # We need to capture the Chairman's output. 
        # Since _get_chairman_ruling is atomic, we can't stream the generation token-by-token 
        # unless we rewrite it to use streamGenerateContent. 
        # For now, we'll keep it atomic but yield it as the final chunk.
        
        # Reuse existing logic but we need to handle the case where it returns a string
        # We manually construct the messages again or just call the helper 
        
        final_answer = await self._get_chairman_ruling(query, context, valid_opinions)
        
        # Yield result
        # We can stream the text in chunks if we want to simulate typing, but for now just send it.
        # Format for SSE data: 
        import json
        yield f"data: {json.dumps({'answer': final_answer})}\n"
        yield "log: Session closed.\n"

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

        logger.info("Council session started. Members deliberating...")
        opinions = await asyncio.gather(*tasks)
        
        valid_opinions = [op for op in opinions if op is not None]
        final_answer = await self._get_chairman_ruling(query, context, valid_opinions)
        
        return {
            "query": query,
            "answer": final_answer,
            "council_opinions": valid_opinions
        }

council_service = CouncilService()
