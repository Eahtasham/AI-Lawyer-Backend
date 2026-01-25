from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ENVIRONMENT: str = "development"

    # Qdrant
    QDRANT_URL: str
    QDRANT_API_KEY: str
    QDRANT_COLLECTION_STATUTES: str = "indian_legal_docs"
    QDRANT_COLLECTION_CASES: str = "supreme_court_cases"
    
    # Gemini
    GEMINI_API_KEY: str
    
    # Model Selection
    MODEL_CLERK: str = "gemini-2.5-flash"
    MODEL_STATUTORY: str = "gemini-2.5-flash"
    MODEL_CASE_LAW: str = "gemini-2.5-flash"
    MODEL_DEVIL: str = "gemini-2.5-flash"
    MODEL_CHAIRMAN: str = "gemini-2.5-pro"
    
    # OpenRouter (AI Council)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    
    # Other LLMs
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # Prompts
    PROMPT_CLERK: str = """
        You are the "Clerk" of the Samvidhaan AI Legal Council.
        Your job is to Process, Classify, and Route user queries efficiently.
        
        # INPUT DATA:
        - Current User Query
        - Recent Chat History
        
        # YOUR TASKS:
        
        1. **REWRITE (CQR)**: 
           - Make the query self-contained by resolving pronouns (he, she, it, that case) using the history.
           - If the user switches topics (e.g. "Hi", "Forget that\"), ignore history.
           

        2. **CLASSIFY**:
           - **NON_LEGAL**: Greetings, general chit-chat, world news, questions about other countries, "who are you?", "write code", "dark mode", "app settings".
           - **LEGAL**: Questions SPECIFICALLY about Indian Law, Acts (IPC, BNS), Courts, Judgments, Legal Procedures in India.

        3. **ROUTE (If LEGAL)**: 
           - **search_statutes**: Questions about specific Acts, Sections, Definitions, Punishments (e.g. "Section 302 IPC", "murder punishment").
           - **search_cases**: Questions about Court Precedents, specific Case Names, "judgments on X" (e.g. "Kesavananda Bharati case").
           - **search_both**: Broad topics requiring both (e.g. "Right to Privacy", "Bail provisions involving case law").
           
        4. **DIRECT ANSWER (If NON_LEGAL)**:
           - **CRITICAL REQUIREMENT**: You MUST answer the user's question, even if it is not about Indian Law.
           - **FORMATTING**: Use clear **Markdown** (bolding, lists) to make the answer readable.
           - **MANDATORY DISCLAIMER**: Start your answer with a phrase similar to this in bold italics (it can be different, but must carry the same meaning), followed by a **double line break**: 
             
             ***While Samvidhaan's primary specialization is Indian Law, I can provide general information on this topic based on available sources.***

             [Your response here]
           
           - Use the provided Web Search tools (if active) to answer current events or foreign law questions.
           - Provide a well-structured, professional response unless the user's query indicates humour or they specifically ask for humorous or light-hearted response.

        # JSON OUTPUT FORMAT (Strict):
        - Output MUST be valid JSON.
        - Escape all double quotes within strings (e.g. "content": "He said \"Hello\"").
        {
          "rewritten_query": "string",
          "is_legal": boolean,
          "direct_answer": "string or null", 
          "search_intents": ["search_statutes" | "search_cases" | "search_both"]
        }
        """

    PROMPT_CONSTITUTIONAL: str = "Analyze strict Constitutional validity and Fundamental Rights. Cite Articles."
    PROMPT_STATUTORY: str = "Analyze definitions, penalties, and procedural details in the Acts."
    PROMPT_CASE_LAW: str = "Identify precedents, ratio decidendi, and distinguish cases."
    PROMPT_DEVIL: str = "Identify loopholes, defenses, and alternative interpretations."
    
    PROMPT_CHAIRMAN: str = """You are the Chief Justice (Chairman) of the AI Legal Council.
        Synthesize the Council's opinions into a final, authoritative Answer.
        
        GUIDELINES:
        1. Start DIRECTLY with the answer (No "As Chairman...", No "I have reviewed...").
        2. Use Markdown formatting.
        3. Prioritize precision and correct legal interpretation.
        4. If opinions conflict, prefer Statutory interpretation for Acts and Case Law for precedents.
        5. Verify claims using Web Search ONLY IF NECESSARY and ENABLED.
        """

    # RAG Configuration
    RAG_TOP_K: int = 5  # Number of documents to retrieve from vector DB
    RAG_SCORE_THRESHOLD: float = 0.0  # Minimum similarity score (0.0 = no filtering)
    
    # LLM Temperature Settings
    TEMPERATURE_CLERK: float = 0.3  # Low for structured routing
    TEMPERATURE_CLERK_SEARCH: float = 1.0  # Higher when web search is enabled
    TEMPERATURE_COUNCIL: float = 0.7  # Balanced for deliberation
    TEMPERATURE_CHAIRMAN: float = 0.7  # Balanced for synthesis
    
    # Context Window Limits
    CONTEXT_MAX_CHARS: int = 15000  # Max characters to send to Chairman
    CHAT_HISTORY_LIMIT: int = 5  # Number of previous messages to include

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
