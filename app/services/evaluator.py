import json
import logging
import httpx
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.db import db_service

logger = logging.getLogger(__name__)


class EvaluatorService:
    def __init__(self):
        # Use model from configuration (default to same as chairman if not specified)
        self.eval_model = getattr(settings, 'MODEL_EVALUATOR', settings.MODEL_CHAIRMAN)
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def evaluate_rag_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        generated_answer: str,
        session_id: str,
        message_id: str,
        model_used: str,
        latency_ms: int = 0,
    ) -> Dict[str, float]:
        """
        Evaluates a RAG response based on faithfulness, relevance, context precision, and chunk coverage.
        """
        context = "\n\n".join(
            [
                chunk.get("page_content", chunk.get("text", ""))
                for chunk in retrieved_chunks
            ]
        )

        chunk_coverage = 1.0 if len(retrieved_chunks) > 0 else 0.0

        prompt = f"""
        You are an expert legal evaluator focusing on Indian Law.
        Evaluate the following response based on two criteria:
        1. Faithfulness (0.0 to 1.0): Is the generated answer completely derived from the provided context? (1.0 = yes, entirely, 0.0 = completely hallucinated outside context).
        2. Relevance (0.0 to 1.0): Does the generated answer directly and accurately address the user's query? (1.0 = completely relevant, 0.0 = completely irrelevant).

        User Query: {query}
        Context Provided:
        {context}

        Generated Answer:
        {generated_answer}

        Return ONLY a JSON object with keys "faithfulness" and "relevance".
        """

        url = f"{self.base_url}/{self.eval_model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1},
        }

        try:
            async with httpx.AsyncClient(timeout=40.0) as client:
                response = await client.post(url, json=payload)
                if response.status_code != 200:
                    logger.error(f"Gemini API Error during evaluation: {response.text}")
                    return {
                        "faithfulness": 0.0,
                        "relevance": 0.0,
                        "context_precision": 0.0,
                        "chunk_coverage": 0.0,
                    }

                data = response.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                result = json.loads(
                    text.strip().replace("```json", "").replace("```", "")
                )

                faithfulness = float(result.get("faithfulness", 0.0))
                relevance = float(result.get("relevance", 0.0))
                context_precision = relevance * 0.9  # Appx heuristic for now

                # Store to DB asynchronously
                self._store_metrics(
                    session_id=session_id,
                    message_id=message_id,
                    query=query,
                    context_precision=context_precision,
                    faithfulness=faithfulness,
                    relevance=relevance,
                    chunk_coverage=chunk_coverage,
                    model_used=model_used,
                    latency_ms=latency_ms,
                )

                return {
                    "faithfulness": faithfulness,
                    "relevance": relevance,
                    "context_precision": context_precision,
                    "chunk_coverage": chunk_coverage,
                }
        except Exception as e:
            logger.error(f"Error during RAG evaluation: {e}")
            return {
                "faithfulness": 0.0,
                "relevance": 0.0,
                "context_precision": 0.0,
                "chunk_coverage": 0.0,
            }

    def _store_metrics(self, **kwargs):
        try:
            db_service.supabase.table("rag_metrics").insert(kwargs).execute()
        except Exception as e:
            logger.error(f"Error saving RAG metrics to DB: {e}")


evaluator_service = EvaluatorService()
