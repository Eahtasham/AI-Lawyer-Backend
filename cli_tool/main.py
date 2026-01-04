import asyncio
import sys
import logging
from app.services.rag import rag_service
from app.models.schemas import ChatResponse
from cli_tool.logger import logger

# Configure logging to be less verbose for the chat interface
logging.getLogger("httpx").setLevel(logging.WARNING)

def print_separator(char="=", length=50):
    print(char * length)

def print_response(data: ChatResponse):
    """Pretty prints the chat response matches test_council.py style."""
    if not data:
        return

    answer = data.answer
    chunks = data.chunks
    council = data.council_opinions
    
    # 1. Council Opinions
    if council:
        print("\n" + "="*20 + " COUNCIL OPINIONS " + "="*20)
        for opinion in council:
            role = opinion.get("role", "Unknown")
            model = opinion.get("model", "Unknown")
            text = opinion.get("opinion", "")
            
            # Check for Special Power (Direct Ruling) tag in opinions (though usually Chairman handles it)
            # But here we just print what the council said.
            print(f"\n[{role}] ({model}):")
            print("-" * 30)
            print(text.strip())
            print("-" * 30)
    
    # 2. Chairman's Ruling
    print("\n" + "="*20 + " CHAIRMAN'S RULING " + "="*20)
    print(f"{answer}")
    print("="*50)
    
    # 3. Sources
    if chunks:
        print("\n" + "-"*20 + " Sources (Indian Kanoon) " + "-"*20)
        for i, chunk in enumerate(chunks, 1):
            text = chunk.text.strip()
            # print first 150 chars of source
            preview = text[:150] + "..." if len(text) > 150 else text
            print(f"  {i}. {preview}")
            if chunk.metadata:
                 print(f"     (Metadata: {chunk.metadata})")
    print("\n")

async def chat_loop():
    """Main interactive chat loop."""
    print_separator()
    print("AI LAWYER COUNCIL - CLI INTERFACE")
    print("Direct Connection to Gemini 2.0 Backend")
    print_separator()
    print("Type 'exit', 'quit', or 'q' to end the session.\n")
    
    while True:
        try:
            user_input = input("YOU: ").strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        if user_input.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        print("\nCalling CouncilService.deliberate()...")
        print("Thinking...", end = "\r")
        
        try:
            # DIRECT SERVICE CALL (No HTTP)
            response = await rag_service.process_query(user_input)
            
            print(" " * 20, end="\r") # Clear "Thinking..."
            print_response(response)
            
        except Exception as e:
            print(f"\n[ERROR] Processing failed: {e}")
            logger.error(f"Error processing query: {e}")

    print("Session ended.")

def main():
    try:
        asyncio.run(chat_loop())
    except KeyboardInterrupt:
        print("\nGoodbye!")

if __name__ == "__main__":
    main()
