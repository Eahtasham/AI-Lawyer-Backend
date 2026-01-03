import httpx
import sys
import asyncio
from cli_tool.logger import logger

# Configuration
API_URL = "http://localhost:8000/api/chat"

async def send_chat_message(message: str):
    """Sends a chat message to the backend and returns the response."""
    payload = {
        "query": message,
        "top_k": 5
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(API_URL, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as exc:
        logger.error(f"An error occurred while requesting {exc.request.url!r}.")
        print(f"\n[ERROR] Connection error: {exc}")
        return None
    except httpx.HTTPStatusError as exc:
        logger.error(f"Error response {exc.response.status_code} while requesting {exc.request.url!r}.")
        print(f"\n[ERROR] Server returned error: {exc.response.status_code}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n[ERROR] Unexpected error: {e}")
        return None

def print_response(data):
    """Pretty prints the chat response."""
    if not data:
        return

    answer = data.get("answer", "No answer provided.")
    chunks = data.get("chunks", [])
    council = data.get("council_opinions", [])
    
    if council:
        print("\n" + "="*20 + " COUNCIL OPINIONS " + "="*20)
        for opinion in council:
            role = opinion.get("role", "Unknown")
            model = opinion.get("model", "Unknown")
            text = opinion.get("opinion", "")
            print(f"\n[{role}] ({model}):")
            print("-" * 30)
            print(text.strip())
            print("-" * 30)
    
    print("\n" + "="*50)
    print(f"CHAIRMAN'S RULING: {answer}")
    print("="*50)
    
    if chunks:
        print("\nSources (Indian Kanoon):")
        for i, chunk in enumerate(chunks, 1):
            text = chunk.get("text", "").strip()
            # print first 100 chars of source
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"  {i}. {preview}")
            if chunk.get("metadata"):
                 print(f"     (Metadata: {chunk['metadata']})")
    print("\n" + "-"*50 + "\n")

async def chat_loop():
    """Main interactive chat loop."""
    print("Welcome to the AI Lawyer CLI Chat!")
    print("Type 'exit', 'quit', or 'q' to end the session.")
    print(f"Connecting to: {API_URL}\n")
    
    logger.info("CLI Chat session started.")

    while True:
        try:
            user_input = input("You: ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        logger.info(f"User query: {user_input}")
        
        print("Thinking...", end = "\r")
        response_data = await send_chat_message(user_input)
        
        if response_data:
            print(" " * 20, end="\r") # Clear "Thinking..."
            print_response(response_data)
            logger.info("Response received and displayed.")
        else:
             print(" " * 20, end="\r") # Clear "Thinking..."

    logger.info("CLI Chat session ended.")

def main():
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")
        logger.info("CLI Chat session interrupted by user.")

if __name__ == "__main__":
    main()
