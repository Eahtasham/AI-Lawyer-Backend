from app.services.qdrant import qdrant_service
from app.services.gemini import gemini_service
from app.models.schemas import ChatResponse, ChunkResult

class RAGService:
    def process_query(self, query: str, top_k: int = 5) -> ChatResponse:
        """Main RAG pipeline"""
        
        # Retrieve chunks
        chunks = qdrant_service.search(query, top_k)
        
        if not chunks:
            return ChatResponse(
                query=query,
                answer="No relevant documents found.",
                chunks=[],
                model_used="none"
            )
        
        # Build context
        context = "\n\n".join([
            f"[Chunk {c['rank']}]\n{c['text']}\nMetadata: {c['metadata']}"
            for c in chunks
        ])
        
        # Generate answer
        answer = gemini_service.generate(query, context)
        
        return ChatResponse(
            query=query,
            answer=answer,
            chunks=[ChunkResult(**c) for c in chunks],
            llm_model="gemini-3-pro"
        )

rag_service = RAGService()
