import asyncio
import sys
import os

# Ensure app is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.clerk import clerk_service
from app.logger import logger

async def test_clerk():
    print("Testing Clerk Service...")
    
    # Test 1: Generic Query
    q1 = "Hi, who are you?"
    print(f"\nQuery 1: {q1}")
    res1 = await clerk_service.classify_and_route(q1, [])
    print(f"Result 1: Legal={res1.is_legal}, Answer='{res1.direct_answer}'")
    
    # Test 2: Legal Statute
    q2 = "What is the punishment for murder under BNS?"
    print(f"\nQuery 2: {q2}")
    res2 = await clerk_service.classify_and_route(q2, [])
    print(f"Result 2: Legal={res2.is_legal}, Intent={res2.search_intents}")
    
    # Test 3: Case Law
    q3 = "Summarize the Puttaswamy judgment."
    print(f"\nQuery 3: {q3}")
    res3 = await clerk_service.classify_and_route(q3, [])
    print(f"Result 3: Legal={res3.is_legal}, Intent={res3.search_intents}")

if __name__ == "__main__":
    asyncio.run(test_clerk())
