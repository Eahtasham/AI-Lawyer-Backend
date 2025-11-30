from qdrant_client import QdrantClient
from app.config import settings
from app.logger import logger
import google.generativeai as genai

class QdrantService:
    def __init__(self):
        logger.info(" Initializing Qdrant service...")
        self.client = QdrantClient(
            url=settings.QDRANT_URL, 
            api_key=settings.QDRANT_API_KEY
        )
        genai.configure(api_key=settings.GEMINI_API_KEY)
        logger.info(" Qdrant service initialized")
    
    def search(self, query: str, top_k: int = 5):
        """Search Qdrant using embedded query"""
        try:
            logger.info(f" Searching for: {query[:50]}...")
            
            # Embed query
            logger.debug(" Generating embedding...")
            embedding = genai.embed_content(
                model="models/gemini-embedding-001",
                content=query,
                task_type="retrieval_query",
                output_dimensionality=2048
            )
            
            # Search Qdrant
            logger.debug(f" Querying Qdrant (top_k={top_k})...")
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=embedding["embedding"],
                limit=top_k
            )
            
            # Format results
            chunks = []
            for i, point in enumerate(results.points, 1):
                chunks.append({
                    "rank": i,
                    "score": point.score,
                    "text": point.payload.get("text", ""),
                    "metadata": {
                        "law": point.payload.get("law", ""),
                        "chapter_number": point.payload.get("chapter_number", ""),
                        "chapter_title": point.payload.get("chapter_title", ""),
                        "section_number": point.payload.get("section_number", ""),
                        "section_title": point.payload.get("section_title", "")
                    }
                })
            
            logger.info(f" Found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f" Qdrant search failed: {str(e)}", exc_info=True)
            raise

qdrant_service = QdrantService()