# Proof of Concept: AI Council Architecture for Indian Legal Query Resolution

## 1. Executive Summary
This document outlines the Proof of Concept (PoC) for upgrading the existing **Samvidhaan Backend** from a standard RAG (Retrieval-Augmented Generation) system to a multi-agent **"AI Council"** architecture. By emulating a panel of legal experts rather than a single assistant, we aim to significantly reduce hallucinations and improve the nuance, accuracy, and legal grounding of answers, specifically tailored for Indian Law.

## 2. Problem Statement
Single-model LLM responses, even with RAG, suffer from:
*   **"Bias of One":** Relying on a single model's training bias.
*   **Lack of Diverse Perspective:** Real legal advice requires viewing a problem from constitutional, statutory, and case-law perspectives simultaneously.
*   **Hallucination Risks:** Single models may prioritize fluency over strict adherence to the retrieving Indian Kanoon documents.

## 3. The Proposed Solution: "AI Council"
Inspired by ensemble learning and recent "LLM Council" research (e.g., Karpathy's architecture), this system dispatches every user query to multiple specialized AI "personas" (Agents) before synthesizing a final answer.

### Core Architecture
1.  **Ingestion Layer (Existing):**
    *   **Vector Database:** Qdrant (Stores Indian Kanoon Acts & Sections).
    *   **Retrieval:** Fetches Top-K relevant legal text chunks based on semantic similarity.

2.  **The Council (New Multi-Agent Layer):**
    Instead of sending context to just one LLM, we send it to four distinct agents in parallel:
    *   **Agent A: The Constitutional Expert.** Focuses purely on fundamental rights and validity under the Constitution of India.
    *   **Agent B: The Statutory Analyst.** Focuses strictly on black-letter law (IPC, CrPC, BNS) definitions and penalties.
    *   **Agent C: The Case Law Researcher.** Interpretive logic based on judicial precedents.
    *   **Agent D: The Devil's Advocate.** Actively searches for loopholes, defenses, and alternative interpretations to ensure the final advice is robust.

3.  **The Chairman (Synthesis Layer):**
    *   **Role:** Acts as the Senior Partner or Chief Justice.
    *   **Input:** User Query + Retrieved Context + Opinions from Agents A, B, C, & D.
    *   **Output:** A single, authoritative, and balanced legal response that resolves conflicts between the council members.

## 4. Technical Implementation Plan
*   **Stack:** Python, FastAPI (Existing), Qdrant (Existing), Google Gemini (via REST API).
*   **Concurrency:** Use `asyncio` to run Council Member generation in parallel (reducing latency).
*   **Prompt Engineering:** specialized system prompts for each persona (as defined in `PROMPT.md`).

## 5. Expected Benefits
*   **Higher Accuracy:** Cross-verification by multiple agents reduces errors.
*   **Comprehensive Coverage:** Ensures no angle (statutory vs. constitutional) is missed.
*   **Explainability:** The final output can optionally show the "deliberations" of the council, building user trust.

## 6. Roadmap
1.  **Phase 1 (MVP):** Implement the Council & Chairman logic in `app/services/council.py`. Connect to existing `QdrantService`.
2.  **Phase 2 (Evaluation):** Compare "Council" answers vs. "Single Model" answers on a set of benchmark Indian Law queries.
3.  **Phase 3 (Optimization):** Introduce a "Peer Review" round where agents critique each other before the Chairman decides.

---
*Prepared by Development Team for Supervisor Review*
