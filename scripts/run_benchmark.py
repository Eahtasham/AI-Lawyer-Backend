import asyncio
import json
import time
import os
import sys
import httpx
from pathlib import Path

# Add backend dir to path for imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.evaluator import evaluator_service
from app.services.council import council_service
from app.services.db import db_service


async def _mock_stream_generator_to_string(gen):
    full_str = ""
    chunks = []
    async for event in gen:
        if event.startswith("data:"):
            try:
                data = json.loads(event[5:].strip())
                if "answer" in data:
                    full_str += data["answer"]
            except:
                pass
        elif event.startswith("chunks:"):
            try:
                chunks = json.loads(event[7:].strip())
            except:
                pass
    return full_str, chunks


async def evaluate_samvidhaan_council(dataset: list) -> dict:
    print("Evaluating Samvidhaan Council...")
    metrics = []

    for item in dataset:
        query = item["query"]
        start_time = time.time()

        # Bypass user DB logic by providing None as conv_id
        gen = council_service.deliberate_stream(
            query=query,
            chat_history=[],
            enable_web_search=False,
            conv_id=None,
            mode="research",
        )

        answer, chunks = await _mock_stream_generator_to_string(gen)
        latency = int((time.time() - start_time) * 1000)

        # Evaluate
        eval_result = await evaluator_service.evaluate_rag_response(
            query=query,
            retrieved_chunks=chunks,
            generated_answer=answer,
            session_id="benchmark",
            message_id="benchmark",
            model_used="samvidhaan_council",
            latency_ms=latency,
        )

        eval_result["latency"] = latency
        metrics.append(eval_result)
        time.sleep(2)  # rate limit prevention

    return _aggregate_metrics(metrics)


async def evaluate_gemini_baseline(
    dataset: list, model_name="gemini-3.5-flash"
) -> dict:
    print(f"Evaluating {model_name} Baseline...")
    metrics = []

    base_url = "https://generativelanguage.googleapis.com/v1beta/models"
    url = f"{base_url}/{model_name}:generateContent?key={settings.GEMINI_API_KEY}"

    async with httpx.AsyncClient(timeout=40.0) as client:
        for item in dataset:
            query = item["query"]
            start_time = time.time()

            prompt = (
                f"You are a legal assistant for Indian Law. Answer this query: {query}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1},
            }

            response = await client.post(url, json=payload)
            if response.status_code == 200:
                data = response.json()
                answer = data["candidates"][0]["content"]["parts"][0]["text"]
            else:
                answer = "Error generating answer from Baseline LLM."

            latency = int((time.time() - start_time) * 1000)

            # Evaluate context-free baseline
            eval_result = await evaluator_service.evaluate_rag_response(
                query=query,
                retrieved_chunks=[],
                generated_answer=answer,
                session_id="benchmark",
                message_id="benchmark",
                model_used=model_name,
                latency_ms=latency,
            )

            eval_result["latency"] = latency
            metrics.append(eval_result)
            time.sleep(2)

    return _aggregate_metrics(metrics)


def _aggregate_metrics(metrics: list) -> dict:
    if not metrics:
        return {}
    return {
        "relevance": sum(m.get("relevance", 0) for m in metrics) / len(metrics),
        "faithfulness": sum(m.get("faithfulness", 0) for m in metrics) / len(metrics),
        "context_precision": sum(m.get("context_precision", 0) for m in metrics)
        / len(metrics),
        "latency": sum(m.get("latency", 0) for m in metrics) / len(metrics),
    }


async def run_all_benchmarks():
    dataset_path = os.path.join(
        Path(__file__).resolve().parent.parent, "eval_data", "indian_legal_bench.json"
    )
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    models_to_test = [
        ("Samvidhaan Council", evaluate_samvidhaan_council),
        ("gemini-3.5-flash", evaluate_gemini_baseline),
    ]

    for name, eval_func in models_to_test:
        results = await eval_func(dataset)

        # Save to Supabase
        db_service.supabase.table("benchmark_runs").insert(
            {
                "model_name": name,
                "dataset_name": "Indian Legal Bench V1 - Small",
                "avg_relevance": results.get("relevance", 0),
                "avg_faithfulness": results.get("faithfulness", 0),
                "avg_context_precision": results.get("context_precision", 0),
                "avg_latency_ms": int(results.get("latency", 0)),
                "total_queries": len(dataset),
            }
        ).execute()

        print(f"Finished {name}: {results}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Load dotenv from the backend root folder
    env_path = os.path.join(Path(__file__).resolve().parent.parent, ".env")
    load_dotenv(env_path)
    asyncio.run(run_all_benchmarks())
