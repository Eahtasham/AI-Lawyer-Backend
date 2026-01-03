import httpx
from app.config import settings
from app.logger import logger

class GeminiService:
    """Gemini service using REST API (bypasses gRPC issues)"""
    
    def __init__(self):
        logger.info(" Initializing Gemini service (REST API mode)...")
        self.api_key = settings.GEMINI_API_KEY
        self.model = settings.GEMINI_MODEL
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
        logger.info(" Gemini service ready")
    
    def generate(self, query: str, context: str) -> str:
        """Generate answer using Gemini REST API"""
        try:
            logger.info(f" Generating answer with Gemini (REST)...")
            logger.debug(f"Context length: {len(context)} chars")
            
            prompt = f"""You are a legal assistant AI trained on Indian Laws.

Use ONLY the following retrieved legal document text to answer the user query.
If the answer is not present in the context, say: "Not found in retrieved documents."

CONTEXT:
{context}

QUERY:
{query}

Provide a clear, concise answer with references (chapter, section, etc.)."""

            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }]
            }
            
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()
                
            result = response.json()
            answer = result["candidates"][0]["content"]["parts"][0]["text"]
            
            logger.info(" Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f" Gemini generation failed: {str(e)}", exc_info=True)
            raise

gemini_service = GeminiService()