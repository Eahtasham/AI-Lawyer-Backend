# PROMPT: Build an AI Council Architecture for Indian Law

**Objective:** Transform the existing single-model RAG pipeline into a robust **AI Council** system. This system will employ multiple "Council Members" (AI Agents with distinct legal personas) to analyze the retrieved context and user query independently, followed by a "Chairman" AI that synthesizes their insights into a final, authoritative response. This is inspired by Andrej Karpathy's "LLM Council" but tailored for strict accuracy in Indian Law.

## 1. Architecture Overview

### A. Context Ingestion (Existing)
*   **Source:** `QdrantService` (already implemented in `app/services/qdrant.py`).
*   **Action:** Retrieve relevant chunks ($k=5$ or more) based on the user's query.
*   **Data:** Indian Kanoon documents, Central Acts, etc.

### B. The AI Council (New)
Develop a `Council` class that orchestrates multiple `CouncilMember` instances.

**The Council Members:**
Create specific personas to view the problem from different legal angles. All members utilize the *same* retrieved context but process it differently.

1.  **Constitutional Expert:** Focuses on fundamental rights, constitutionality, and high-level legal principles.
    *   *System Prompt Focus:* "You are a Constitutional Law expert. Analyze the query based on the Constitution of India..."
2.  **Statutory Analyst:** Focuses on the strict interpretation of the text (IPC, CrPC, etc.) and specific section wording.
    *   *System Prompt Focus:* "You are a Black-letter law expert. Focus strictly on the definitions and penalties in the provided Acts..."
3.  **Case Law Researcher:** Focuses on precedents and judicial interpretations (if available in context) or general judicial logic.
    *   *System Prompt Focus:* "You are a Case Law specialist. Interpret how courts typically view these scenarios..."
4.  **Devil's Advocate (Defense/Prosecution):** Argues the counter-point or looks for loopholes.
    *   *System Prompt Focus:* "You are a critical thinker. Identify exceptions, defenses, or alternative interpretations..."

### C. The Chairman (New)
A final "Meta-Learner" step.
*   **Input:** Original Query + Retrieved Context + All Council Member Responses.
*   **Role:** Synthesize the disparate views. Resolve conflicts. Highlight consensus. Provide the final actionable advice.
*   **System Prompt:** "You are the Chief Justice/Chairman. Review the opinions of the council. Synthesize the most accurate, balanced, and legally sound response..."

## 2. Implementation Steps

1.  **Extend `GeminiService`:**
    *   Modify `app/services/gemini.py` or create a new `CouncilService`.
    *   Allow passing a custom `system_instruction` per request (currently hardcoded).
    *   Implement `generate_async` to run council members in parallel (using `asyncio.gather`).

2.  **Define Personas:**
    *   Create a `CONFG` or `CONSTANTS` file defining the specific system prompts for each Indian Law persona.

3.  **Peer Review Layer (Optional but Recommended):**
    *   *Phase 1:* Members generate initial drafts.
    *   *Phase 2:* (Advanced) Members see anonymized drafts of others and critique them.
    *   *Phase 3:* Chairman Synthesis.
    *   *MVP Start:* Skip Phase 2 for now, go straight from Phase 1 -> Phase 3 for speed, add Phase 2 later.

4.  **Integration:**
    *   Update `RAGService` in `app/services/rag.py` to call the `Council` instead of the single `gemini_service`.

## 3. Output Format
The final response should be structured:
*   **Chairman's Ruling:** The direct answer.
*   **Council Deliberations:** (Collapsible/Optional) Summary of what each expert argued (e.g., "The Constitutional Expert raised a privacy concern...").
*   **References:** Citations from the Qdrant context.

## 4. Key Constraints
*   **Indian Law Context:** Ensure all personas prioritize Indian statutes (IPC, BNS, Constitution).
*   **Context Grounding:** All members MUST blindly adhere to the `Qdrant` retrieved context if it conflicts with their internal training, to avoid hallucinations.
