import httpx
from qdrant_client import QdrantClient
from app.config import settings
from app.logger import logger

class QdrantService:
    """Qdrant service with REST-based embedding and Dual-Collection Support"""
    
    def __init__(self):
        logger.info(" Initializing Qdrant service (lazy mode)...")
        self._client = None
        self.api_key = settings.GEMINI_API_KEY
        self.embed_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        
        # Collection names from config
        self.collection_statutes = settings.QDRANT_COLLECTION_STATUTES
        self.collection_cases = settings.QDRANT_COLLECTION_CASES
        
        logger.info(f" Qdrant Configured with: Statutes='{self.collection_statutes}', Cases='{self.collection_cases}'")
    
    @property
    def client(self):
        """Lazy initialization of Qdrant client"""
        if self._client is None:
            logger.info(" Connecting to Qdrant...")
            try:
                self._client = QdrantClient(
                    url=settings.QDRANT_URL, 
                    api_key=settings.QDRANT_API_KEY,
                    timeout=10
                )
                logger.info(" Qdrant client connected")
            except Exception as e:
                logger.error(f" Failed to connect to Qdrant: {e}")
                raise
        return self._client
    
    def _get_embedding(self, text: str) -> list:
        """Get embedding using Gemini REST API"""
        url = f"{self.embed_url}?key={self.api_key}"
        
        payload = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": 2048
        }
        
        # logger.debug(f" Requesting embedding for: {text[:20]}...")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        
        result = response.json()
        embedding = result["embedding"]["values"]
        
        if len(embedding) != 2048:
            logger.warning(f" Expected 2048 dimensions, got {len(embedding)}!")
        
        return embedding
    
    def _normalize_metadata(self, payload: dict) -> dict:
        """
        Normalize metadata from diverse Qdrant collections.
        """
        # 1. Check for Case Law Structure (Field 'petitioner' or 'court' usually indicates case)
        if 'petitioner' in payload or 'respondent' in payload or 'judgment' in payload.get('summary', {}):
            # It's a Case Judgment
            # Get Summary
            summary_obj = payload.get('summary', {})
            executive_summary = summary_obj.get('executive_summary', '') if isinstance(summary_obj, dict) else ""
            
            # Construct text content if missing or if we want summary
            # User prefers "relevant part... proper summary"
            text_content = payload.get('text', '')
            if not text_content and executive_summary:
                 text_content = f"**Summary**: {executive_summary}"

            return {
                'source_type': 'case_law',
                'case_name': payload.get('title', 'Untitled Judgment'), # Frontend expects case_name
                'petitioner': payload.get('petitioner', ''),
                'respondent': payload.get('respondent', ''),
                'case_number': payload.get('case_number', ''),
                'case_type': payload.get('case_type', ''),
                'court': payload.get('court', 'Supreme Court'),
                'date': payload.get('date', 'Unknown Date'),
                'year': payload.get('year', ''),
                'url': payload.get('url', ''),
                'citation': ", ".join([str(x) for x in payload.get('citation_refs', []) if x is not None]) if isinstance(payload.get('citation_refs'), list) else str(payload.get('citation_refs', '') or ''),
                'bench': ", ".join([str(x) for x in payload.get('bench', []) if x is not None]) if isinstance(payload.get('bench'), list) else str(payload.get('bench', '') or ''),
                'text': text_content, # Normalized text field
                # Keep raw
                'full_metadata': payload
            }
            
        # 2. Check for Statute/Act Structure
        else:
            # It's a Statute
            # Cleanup array fields
            sect_num = payload.get('section_numbers', [])
            sect_title = payload.get('section_titles', [])
            
            if isinstance(sect_num, list): sect_num = ", ".join(map(str, sect_num))
            if isinstance(sect_title, list): sect_title = ", ".join(map(str, sect_title))
            
            return {
                'source_type': 'statute',
                'law': payload.get('act_title', payload.get('law', 'Unknown Act')), # Frontend expects 'law'
                'act_id': payload.get('act_id', ''),
                'section_number': sect_num or str(payload.get('section_number', '')),
                'section_title': sect_title or payload.get('section_title', ''),
                'chapter_title': payload.get('chapter_name', payload.get('chapter_title', '')),
                'year': payload.get('year', ''),
                'enactment_date': payload.get('enactment_date', ''),
                'url': payload.get('url', None),
                'text': payload.get('text', ''), # User wants relevant part, usually 'text' in statute payload IS the section text
                # Keep raw
                'full_metadata': payload
            }

    def search(self, query: str, collection_name: str, top_k: int = 5):
        """Generic search method"""
        try:
            logger.info(f" [Qdrant] Searching '{collection_name}' for: {query[:40]}...")
            
            # Embed
            embedding = self._get_embedding(query)
            
            # Query
            results = self.client.query_points(
                collection_name=collection_name,
                query=embedding,
                limit=top_k
            )
            
            chunks = []
            for i, point in enumerate(results.points, 1):
                payload = point.payload
                metadata = self._normalize_metadata(payload)
                
                # IMPORTANT: For cases, the text might be in 'summary.executive_summary' or 'text' field
                # The user payload shows 'summary' key having structured summaries.
                # But Qdrant points usually have a 'text' field for RAG. 
                # If 'text' is missing, fallback to formatted summary.
                
                content_text = payload.get("text", "")
                if not content_text and metadata['source_type'] == 'case_law':
                     # Synthesize text from summary if main text missing
                     summ = payload.get('summary', {})
                     if isinstance(summ, dict):
                         content_text = f"HEADNOTE: {summ.get('executive_summary', '')}\n\nFACTS: {summ.get('facts', '')}\n\nHELD: {summ.get('judgment', '')}"
                
                chunks.append({
                    "rank": i,
                    "score": point.score,
                    "text": content_text,
                    "metadata": metadata
                })
            
            logger.info(f" [Qdrant] Found {len(chunks)} results in {collection_name}")
            return chunks
            
        except Exception as e:
            logger.error(f" [Qdrant] Search failed in {collection_name}: {str(e)}")
            # Don't crash the whole chain, return empty
            return []

    def search_statutes(self, query: str, top_k: int = None):
        """Helper for Statutes Collection"""
        if top_k is None:
            top_k = settings.RAG_TOP_K
        return self.search(query, self.collection_statutes, top_k)

    def search_cases(self, query: str, top_k: int = None):
        """Helper for Case Law Collection"""
        if top_k is None:
            top_k = settings.RAG_TOP_K
        return self.search(query, self.collection_cases, top_k)

qdrant_service = QdrantService()