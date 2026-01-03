import asyncio
from app.services.council import council_service
from app.logger import logger

# Configure logging to console for the test
import logging
logging.basicConfig(level=logging.INFO)

async def test_council():
    print("\n" + "="*50)
    print("AI COUNCIL TEST RUNNER")
    print("="*50 + "\n")
    
    query = "What is the punishment for theft under Indian Law?"
    context = """
    [Chunk 1]
    IPC Section 378. Theft.—Whoever, intending to take dishonestly any movable property out of the possession of any person without that person’s consent, moves that property in order to such taking, is said to commit theft.
    
    [Chunk 2]
    IPC Section 379. Punishment for theft.—Whoever commits theft shall be punished with imprisonment of either description for a term which may extend to three years, or with fine, or with both.
    """
    
    print(f"QUERY: {query}\n")
    print("Calling CouncilService.deliberate()...\n")
    
    try:
        result = await council_service.deliberate(query, context)
        
        print("\n" + "="*20 + " COUNCIL OPINIONS " + "="*20)
        for opinion in result['council_opinions']:
            print(f"\n[{opinion['role']}] ({opinion['model']}):")
            print(f"{opinion['opinion'][:200]}...") # truncate for display
            
        print("\n" + "="*20 + " CHAIRMAN'S RULING " + "="*20)
        print(result['answer'])
        print("\n" + "="*50)
        print("TEST PASSED")
        
    except Exception as e:
        print(f"\nTEST FAILED: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_council())
