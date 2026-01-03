from app.services.qdrant import qdrant_service
from app.services.council import council_service
from app.models.schemas import ChatResponse, ChunkResult

class RAGService:
    async def process_query(self, query: str, top_k: int = 5) -> ChatResponse:
        """Main RAG pipeline (Async with AI Council)"""
        
        # Retrieve chunks (Sync for now, but Qdrant service is fast)
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
        
        # Generate answer using AI Council (Async)
        council_result = await council_service.deliberate(query, context)
        answer = council_result["answer"]
        
        return ChatResponse(
            query=query,
            answer=answer,
            chunks=[ChunkResult(**c) for c in chunks],
            llm_model="AI_Council (Gemini/Llama Mix)",
            council_opinions=council_result.get("council_opinions", [])
        )

rag_service = RAGService()
