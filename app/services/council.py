import asyncio
from typing import List, Dict, Any
from openai import AsyncOpenAI
from app.config import settings
from app.logger import logger

class CouncilService:
    """
    AI Council Service using OpenRouter to orchestrate multiple diverse LLMs.
    Implements the architecture defined in AI_COUNCIL_ARCHITECTURE.md
    """

    # Free Tier Models 
    MODEL_CONSTITUTIONAL = "meta-llama/llama-3.3-70b-instruct:free"  # High reasoning for complex rights
    MODEL_STATUTORY = "mistralai/mistral-small-3.1-24b-instruct:free"  # Precise and logical
    MODEL_CASE_LAW = "nousresearch/hermes-3-llama-3.1-405b:free"  # Massive knowledge base for precedents
    MODEL_DEVIL = "deepseek/deepseek-r1-0528:free"  # Strong reasoning for alternative views
    MODEL_CHAIRMAN = "google/gemini-2.0-flash-exp:free"  # 1M context window for synthesis

    def __init__(self):
        self.api_key = settings.OPENROUTER_API_KEY
        self.base_url = settings.OPENROUTER_BASE_URL
        
        # Rate Limit Protection: Limit concurrency to 2 parallel requests
        self.semaphore = asyncio.Semaphore(2)
        
        if not self.api_key:
            logger.warning("OPENROUTER_API_KEY not found. Council service will fail if used.")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.info("CouncilService initialized with OpenRouter (Max Concurrency: 2)")

    async def _get_member_opinion(self, role: str, model: str, system_prompt: str, user_query: str, context: str) -> Dict[str, str]:
        """Async worker to get a single council member's opinion"""
        async with self.semaphore:
            # Stagger requests slightly to avoid hitting per-second limits
            await asyncio.sleep(0.5) 
            
            try:
                logger.info(f"[{role}] Deliberating using {model}...")
                
                full_prompt = f"""
                CONTEXT FROM INDIAN LAWS:
                {context}

                USER QUERY:
                {user_query}
                
                Based strictly on the context above, provide your analysis.
                """

                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_prompt}
                    ],
                    temperature=0.7,
                    extra_headers={
                        "HTTP-Referer": "https://ai-lawyer-council.com", 
                        "X-Title": "AI Lawyer Council"
                    }
                )
                
                opinion = response.choices[0].message.content
                logger.info(f"[{role}] Opinion submitted.")
                return {
                    "role": role,
                    "model": model,
                    "opinion": opinion
                }

            except Exception as e:
                logger.warning(f"[{role}] Absented due to error: {str(e)}")
                # Return None to indicate absence
                return None

    async def _get_chairman_ruling(self, query: str, context: str, opinions: List[Dict[str, str]]) -> str:
        """The Chairman synthesizes all opinions into a final answer"""
        valid_opinions = [op for op in opinions if op is not None]
        
        if not valid_opinions:
            logger.error("[Chairman] No council members present!")
            return "The AI Council could not convene due to high traffic (Rate Limits). Please try again later."
            
        logger.info(f"[Chairman] Reviewing {len(valid_opinions)} council opinions...")

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
            response = await self.client.chat.completions.create(
                model=self.MODEL_CHAIRMAN,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,
                 extra_headers={
                    "HTTP-Referer": "https://ai-lawyer-council.com",
                    "X-Title": "AI Lawyer Council"
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[Chairman] Failed to rule: {str(e)}")
            return "The Chairman is currently unavailable due to high traffic."

    async def deliberate(self, query: str, context: str) -> Dict[str, Any]:
        """Main entry point: Orchestrate the full council session"""
        
        # 1. Define the Council Members and their System Prompts
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
                system_prompt="You are a Case Law specialist. Look for precedents and interpret how courts typically view these scenarios. If no specific case is in context, apply general judicial logic.",
                user_query=query, 
                context=context
            ),
            self._get_member_opinion(
                role="Devil's Advocate", 
                model=self.MODEL_DEVIL,
                system_prompt="You are the Devil's Advocate. Your job is to find loopholes, defenses, exceptions, or alternative interpretations that the others might miss. Be critical and skeptical.",
                user_query=query, 
                context=context
            )
        ]

        # 2. Parallel Execution (Async but Semaphore limited)
        logger.info("Council session started. Members deliberating...")
        opinions = await asyncio.gather(*tasks)
        
        # Filter out None values (absent members)
        valid_opinions = [op for op in opinions if op is not None]

        # 3. Chairman's Ruling
        final_answer = await self._get_chairman_ruling(query, context, valid_opinions)
        
        return {
            "query": query,
            "answer": final_answer,
            "council_opinions": valid_opinions
        }

council_service = CouncilService()
