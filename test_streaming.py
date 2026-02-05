import httpx
import asyncio
import json

async def test_stream():
    url = "http://localhost:8000/api/stream"
    params = {
        "query": "What are the punishments for theft under IPC?",
        "context_window": 5,
        "mode": "fast" # Use fast mode for quick test
    }
    
    print(f"Connecting to {url}...")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", url, params=params) as response:
                print(f"Status: {response.status_code}")
                async for line in response.aiter_lines():
                    if line.startswith("token:"):
                        print(f"[TOKEN] {line[6:]}")
                    elif line.startswith("followup:"):
                        print(f"[FOLLOWUP] {line[9:]}")
                    elif line.startswith("data:"):
                        print(f"[DATA] {line[5:]}")
                    elif line.startswith("log:"):
                        # print(f"[LOG] {line[4:]}")
                        pass
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_stream())
