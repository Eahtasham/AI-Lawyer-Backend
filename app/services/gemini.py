import google.generativeai as genai
from app.config import settings
from app.logger import logger

class GeminiService:
    def __init__(self):
        logger.info(" Initializing Gemini service...")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        logger.info(" Gemini service initialized")
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer using Gemini"""
        try:
            logger.info(f" Generating answer with Gemini...")
            logger.debug(f"Context length: {len(context)} chars")
            
            prompt = f"""You are a legal assistant AI trained on Indian Laws.

Use ONLY the following retrieved legal document text to answer the user query.
If the answer is not present in the context, say: "Not found in retrieved documents."

CONTEXT:
{context}

QUERY:
{query}

Provide a clear, concise answer with references (chapter, section, etc.)."""

            response = self.model.generate_content(prompt)
            
            logger.info(" Answer generated successfully")
            return response.text
            
        except Exception as e:
            logger.error(f" Gemini generation failed: {str(e)}", exc_info=True)
            raise

gemini_service = GeminiService()