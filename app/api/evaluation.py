from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
from app.services.db import db_service
import subprocess
import asyncio
import os

router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


@router.get("/metrics")
async def get_rag_metrics():
    try:
        response = (
            db_service.supabase.table("rag_metrics")
            .select("*")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
        return {"data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmark")
async def get_benchmarks():
    try:
        response = (
            db_service.supabase.table("benchmark_runs")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return {"data": response.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/run-benchmark")
async def run_benchmark():
    """Run benchmark tests directly and stream results"""
    async def event_generator():
        try:
            from app.services.evaluator import evaluator_service
            from app.services.council import council_service
            from app.config import settings
            import time
            import json
            import os
            
            # Load benchmark dataset or create fallback
            dataset_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "eval_data", "indian_legal_bench.json"
            )
            
            if not os.path.exists(dataset_path):
                # Use fallback dataset
                dataset = [
                    {"query": "What is Section 302 of the Indian Penal Code?"},
                    {"query": "Explain the Right to Privacy in India"},
                    {"query": "What are the provisions for bail in BNS?"},
                ]
                yield f"data: Using fallback dataset (3 test queries)\n\n"
            else:
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)
                yield f"data: Loaded {len(dataset)} test queries\n\n"
            
            yield f"data: Starting benchmark evaluation...\n\n"
            
            results = []
            for i, test_case in enumerate(dataset):
                query = test_case["query"]
                yield f"data: Running test {i+1}/{len(dataset)}: {query[:50]}...\n\n"
                
                try:
                    start_time = time.time()
                    
                    # Get response from council
                    response_parts = []
                    chunks = []
                    
                    async for event in council_service.deliberate_stream(
                        query=query,
                        chat_history=[],
                        enable_web_search=False,
                        conv_id=f"benchmark_{i}",
                        mode="research",
                    ):
                        if event.startswith("data:"):
                            try:
                                data = json.loads(event[5:].strip())
                                if "answer" in data:
                                    response_parts.append(data["answer"])
                            except:
                                pass
                        elif event.startswith("chunks:"):
                            try:
                                chunks = json.loads(event[7:].strip())
                            except:
                                pass
                    
                    answer = "".join(response_parts)
                    latency_ms = int((time.time() - start_time) * 1000)
                    
                    # Evaluate the response
                    metrics = await evaluator_service.evaluate_rag_response(
                        query=query,
                        retrieved_chunks=chunks,
                        generated_answer=answer,
                        session_id=f"benchmark_{i}",
                        message_id=f"msg_{i}",
                        model_used="samvidhaan-council",
                        latency_ms=latency_ms
                    )
                    
                    results.append({
                        "relevance": metrics.get("relevance", 0),
                        "faithfulness": metrics.get("faithfulness", 0),
                        "context_precision": metrics.get("context_precision", 0),
                        "latency_ms": latency_ms
                    })
                    
                    yield f"data: Test {i+1} completed - Relevance: {metrics.get('relevance', 0):.2f}, Faithfulness: {metrics.get('faithfulness', 0):.2f}, Latency: {latency_ms}ms\n\n"
                    
                except Exception as e:
                    yield f"data: Test {i+1} failed: {str(e)}\n\n"
            
            # Calculate and store aggregate results
            if results:
                avg_relevance = sum(r["relevance"] for r in results) / len(results)
                avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
                avg_precision = sum(r["context_precision"] for r in results) / len(results)
                avg_latency = sum(r["latency_ms"] for r in results) / len(results)
                
                # Store to database
                try:
                    db_service.supabase.table("benchmark_runs").insert({
                        "model_name": "samvidhaan-council",
                        "avg_relevance": avg_relevance,
                        "avg_faithfulness": avg_faithfulness,
                        "avg_context_precision": avg_precision,
                        "avg_latency_ms": int(avg_latency),
                        "total_queries": len(results)
                    }).execute()
                    yield f"data: Results stored to database\n\n"
                except Exception as e:
                    yield f"data: Failed to store results: {str(e)}\n\n"
                
                yield f"data: === BENCHMARK COMPLETE ===\n\n"
                yield f"data: Average Relevance: {avg_relevance:.2f}\n\n"
                yield f"data: Average Faithfulness: {avg_faithfulness:.2f}\n\n"
                yield f"data: Average Precision: {avg_precision:.2f}\n\n"
                yield f"data: Average Latency: {int(avg_latency)}ms\n\n"
            
            yield f"data: [DONE]\n\n"
            
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
