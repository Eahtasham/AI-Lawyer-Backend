from app.services.council import council_service
from app.models.schemas import ChatResponse, ChunkResult

class RAGService:
    async def process_query(self, query: str, top_k: int = 5) -> ChatResponse:
        """Main RAG pipeline (Delegates to Council Agent)"""
        
        # New Flow: Clerk -> Retrieval -> Council -> Chairman
        # Handled internally by council_service
        result = await council_service.deliberate(query)
        
        return ChatResponse(
            query=query,
            answer=result.get("answer", "No answer generated."),
            chunks=[], # Chunks are handled in streaming only for now, or could be extracted if deliberste returned them
            llm_model="AI_Council v3.0",
            council_opinions=result.get("council_opinions", [])
        )

rag_service = RAGService()
