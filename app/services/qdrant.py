import httpx
from qdrant_client import QdrantClient
from app.config import settings
from app.logger import logger

class QdrantService:
    """Qdrant service with REST-based embedding (bypasses gRPC issues)"""
    
    def __init__(self):
        logger.info(" Initializing Qdrant service (lazy mode)...")
        self._client = None
        self.api_key = settings.GEMINI_API_KEY
        self.embed_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
        logger.info(" Qdrant service ready (will connect on first use)")
    
    @property
    def client(self):
        """Lazy initialization of Qdrant client"""
        if self._client is None:
            logger.info(" Connecting to Qdrant...")
            self._client = QdrantClient(
                url=settings.QDRANT_URL, 
                api_key=settings.QDRANT_API_KEY,
                timeout=10
            )
            logger.info(" Qdrant client connected")
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
        
        logger.debug(f" Requesting embedding with payload: {payload}")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
        
        result = response.json()
        embedding = result["embedding"]["values"]
        
        logger.info(f" Embedding dimension: {len(embedding)}")
        
        if len(embedding) != 2048:
            logger.warning(f" Expected 2048 dimensions, got {len(embedding)}!")
        
        return embedding
    
    def _normalize_metadata(self, payload: dict) -> dict:
        """
        Normalize metadata from both old and new Qdrant point structures.
        
        Old structure (from central_acts):
            - law, chapter_number, chapter_title, section_number, section_title
            
        New structure (from indian_kanoon):
            - doc_id, year, act_title, act_id, enactment_date
            - section_numbers (array), section_titles (array)
            - part_id, part_name, chapter_name
            - chunk_index, token_count, source, source_file
        """
        # Check for new structure indicators
        is_new_structure = 'act_title' in payload or 'doc_id' in payload or 'section_numbers' in payload
        
        if is_new_structure:
            # Handle new structure - normalize to a unified format
            section_numbers = payload.get('section_numbers', [])
            section_titles = payload.get('section_titles', [])
            
            # Convert arrays to strings for display
            section_number = ', '.join(str(s) for s in section_numbers) if section_numbers else ''
            section_title = ', '.join(str(s) for s in section_titles) if section_titles else ''
            
            return {
                # Primary fields (unified format)
                'law': payload.get('act_title', ''),
                'chapter_number': payload.get('part_id', ''),
                'chapter_title': payload.get('chapter_name', '') or payload.get('part_name', ''),
                'section_number': section_number,
                'section_title': section_title,
                # Extended new fields
                'doc_id': payload.get('doc_id', ''),
                'year': payload.get('year', ''),
                'act_id': payload.get('act_id', ''),
                'enactment_date': payload.get('enactment_date', ''),
                'source': payload.get('source', ''),
                'source_file': payload.get('source_file', ''),
                'chunk_index': payload.get('chunk_index', ''),
                'token_count': payload.get('token_count', ''),
            }
        else:
            # Handle old structure - pass through with defaults
            return {
                'law': payload.get('law', ''),
                'chapter_number': payload.get('chapter_number', ''),
                'chapter_title': payload.get('chapter_title', ''),
                'section_number': payload.get('section_number', ''),
                'section_title': payload.get('section_title', ''),
                # Extended fields (empty for old structure)
                'doc_id': '',
                'year': '',
                'act_id': '',
                'enactment_date': '',
                'source': '',
                'source_file': '',
                'chunk_index': '',
                'token_count': '',
            }
    
    def search(self, query: str, top_k: int = 5):
        """Search Qdrant using embedded query"""
        try:
            logger.info(f" Searching for: {query[:50]}...")
            
            # Embed query using REST API
            logger.debug(" Generating embedding (REST)...")
            embedding = self._get_embedding(query)
            
            # Search Qdrant
            logger.debug(f" Querying Qdrant (top_k={top_k})...")
            results = self.client.query_points(
                collection_name=settings.QDRANT_COLLECTION,
                query=embedding,
                limit=top_k
            )
            
            # Format results - Handle both old and new metadata structures
            chunks = []
            for i, point in enumerate(results.points, 1):
                payload = point.payload
                
                # Normalize metadata from either old or new structure
                metadata = self._normalize_metadata(payload)
                
                chunks.append({
                    "rank": i,
                    "score": point.score,
                    "text": payload.get("text", ""),
                    "metadata": metadata
                })
            
            logger.info(f" Found {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f" Qdrant search failed: {str(e)}", exc_info=True)
            raise

qdrant_service = QdrantService()