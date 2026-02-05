
import asyncio
import os
import sys

# Add backend path to sys.path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Load .env manually
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), 'backend', '.env'))
except ImportError:
    pass

from app.services.council import council_service
from app.config import settings

async def test_modes():
    print("\n=== Testing FAST Mode (Single Pass) ===")
    try:
        async for chunk in council_service.deliberate_stream(
            query="What is theft under IPC?", 
            mode="fast", 
            enable_web_search=False
        ):
            if "log:" in chunk: print(chunk.strip())
            # Don't print tokens to keep output clean, rely on followup check
            if "followup:" in chunk: print(f"\n[SUCCESS] FOLLOWUP EVENT: {chunk.strip()}")
            if settings.FOLLOWUP_SEPARATOR in chunk:
                 print("\n[FAILURE] SEPARATOR LEAKED INTO STREAM!")
    except Exception as e: print(f"Fast Mode Error: {e}")

    print("\n\n=== Testing BALANCED Mode (Single Pass) ===")
    try:
        async for chunk in council_service.deliberate_stream(
            query="What is theft under IPC?", 
            mode="balanced", 
            enable_web_search=False
        ):
            if "log:" in chunk: print(chunk.strip())
            if "followup:" in chunk: print(f"\n[SUCCESS] FOLLOWUP EVENT: {chunk.strip()}")
    except Exception as e: print(f"Balanced Mode Error: {e}")

if __name__ == "__main__":
    if not settings.GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY not set.")
    else:
        asyncio.run(test_modes())
