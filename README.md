# Samvidhaan Backend ("The AI Council")

The **Samvidhaan Backend** is a sophisticated legal reasoning system designed to provide accurate, balanced, and citation-backed answers to queries about Indian Law.

Unlike traditional chatbots that use a single LLM, this project implements a **Multi-Agent "AI Council" Architecture**. It simulates a panel of legal experts deliberating on a case before a Chief Justice (The Chairman) delivers the final verdict.

## 🏛️ The AI Council Architecture

The system dispatches every user query to four specialized AI agents in parallel:

1.  **📜 Constitutional Expert** (`gemini-2.0-flash`): Evaluates queries against the Constitution of India, focusing on Fundamental Rights and validity.
2.  **⚖️ Statutory Analyst** (`gemini-2.0-flash`): A "Black-letter law" literalist that focuses strictly on Acts (IPC, CrPC, BNS) and defined penalties.
3.  **📚 Case Law Researcher** (`gemini-2.5-flash`): Equipped with **Google Search**, this agent finds relevant Supreme Court precedents and recent judgments.
4.  **😈 Devil's Advocate** (`gemini-2.0-flash`): Actively searches for loopholes, defenses, exceptions, and alternative interpretations.

**👑 The Chairman:** A meta-learner agent (`gemini-2.5-flash`) that synthesizes these diverse (and often conflicting) opinions into a single, authoritative ruling.

## 🚀 Key Features

*   **RAG Pipeline**: Ingests legal texts (IndianKanoon), performs semantic chunking, and retrieves context from **Qdrant**.
*   **Hybrid Search**: Combines dense vector embeddings with metadata filtering (e.g., specific Acts or Years).
*   **Parallel Deliberation**: Uses Python `asyncio` to run all council members simultaneously for low latency.
*   **3-Tier Modes**: Offers "Fast", "Balanced", and "Research" modes to balance speed vs. depth.
*   **Live Streaming**: Exposes a Server-Sent Events (SSE) endpoint to stream deliberation logs in real-time to the frontend.
*   **Citation-Backed**: Answers are grounded in actual statutes and case laws, minimizing hallucinations.

## 🛠️ Tech Stack

*   **Framework**: FastAPI
*   **Language**: Python 3.10+
*   **LLMs**: Google Gemini 2.0 Flash / 2.5 Flash
*   **Vector Database**: Qdrant (Managed/Cloud)
*   **Tools**: Google Search (Built-in via Gemini API)

## 📦 Setup Instructions

1.  **Clone the repository**
    ```bash
    git clone https://github.com/durjoydutta/samvidhaan.git
    cd samvidhaan/backend
    ```

2.  **Create and activate virtual environment**
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables**
    Create a `.env` file in the root directory:
    ```ini
    ENVIRONMENT=development
    
    # Google Gemini
    GEMINI_API_KEY=your_gemini_key_here
    
    # Qdrant Vector DB
    QDRANT_URL=your_qdrant_url
    QDRANT_API_KEY=your_qdrant_key
    QDRANT_COLLECTION=annotated_laws
    ```

5.  **Run the Server**
    ```bash
    uvicorn app.main:app --reload
    ```

## 🔌 API Usage

### 1. Streaming (Live Deliberation)
**GET** `/api/stream?query=...&mode=...&web_search=...`

**Parameters:**
- `query`: The legal question.
- `mode`: `fast` | `balanced` | `research` (Default: `research`).
- `web_search`: `true` | `false` (Enables Google Search for case law).

This endpoint uses **Server-Sent Events (SSE)** to stream the Council's thought process.

**Event Types:**
- `log`: System status updates (e.g., "Constitutional Expert started...")
- `chunks`: The retrieved legal context from Qdrant.
- `opinion`: JSON object containing an agent's specific deliberation.
- `data`: The final synthesized answer from the Chairman.

### 2. Standard Chat (JSON)
**POST** `/api/chat`

Returns the final response in a single JSON object (blocking).

**Payload:**
```json
{
  "query": "What is the punishment for theft?",
  "top_k": 5
}
```

**Response:**
```json
{
  "query": "...",
  "answer": "...",
  "chunks": [...],
  "council_opinions": [...]
}
```

## 📚 Documentation
For a deep dive into the architecture, please refer to the Project Manual:
- [AI Council Manual (LaTeX source)](./ai_council_poc.tex)
