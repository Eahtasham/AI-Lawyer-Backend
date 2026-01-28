# Samvidhaan: The Council Architecture v3.0

> **⚠️ STATUS: PROOF OF CONCEPT (POC)**
> This documentation outlines the proposed logic for the Samvidhaan legal engine. Architecture, model selections, and agent roles are strictly for internal discussion and subject to refinement during development.

## ⚖️ Executive Summary
Samvidhaan is a multi-agent legal reasoning engine designed to provide authoritative responses to Indian legal queries. By separating "Routing" (The Clerk) from "Deliberation" (The Council), the system achieves high precision, reduced hallucinations, and cost-efficiency.

---

## 🏗️ The Core Components

### 1. The "Clerk" (Router Agent)
* **Model:** `Gemini 2.5 Flash` (Optimized for Low Latency)
* **Role:** The Gatekeeper
* **Key Responsibilities:**
    * **CQR (Contextual Query Rewriting):** Resolves pronouns and context from chat history to make queries self-contained.
    * **Classification:** Determines if a query is **LEGAL** or **NON-LEGAL**.
    * **Routing:** Identifies which specific collections (`Statutes`, `Case Law`, or both) need to be queried.

### 2. The Council (Deliberation Engine)
A panel of specialized agents that process retrieved context before passing opinions to a central authority.
* **Chairman:** `Gemini 2.5 Pro` (Optimized for High Reasoning).
* **Specialized Members:**
    * **Constitutional Expert:** Focuses on Articles and Fundamental Rights.
    * **Statutory Analyst:** Deep-dives into specific Acts and Sections.
    * **Case Law Researcher:** Analyzes precedents and Supreme Court judgments.
    * **Devil’s Advocate:** Challenges the prevailing logic to minimize bias.

### 3. Knowledge Bases (Vector DB)
Managed via **Qdrant** with dal-collection indexing:
* `indian_legal_docs`: Constitution, Central Acts, and Statutes.
* `supreme_court_cases`: Summarized Judgments (~1950–Present).

---

## 🔄 System Architecture & Data Flow



1. **Input:** User submits a query and toggles **Web Search [ON/OFF]**.
2. **Routing:** The Clerk classifies the intent. 
   - *Non-Legal:* Instant response (Bypass Council).
   - *Legal:* Route to Qdrant retrieval.
3. **Retrieval:** Relevant context is fetched from the appropriate collections.
4. **Deliberation:** Expert agents review the context and generate specialized opinions.
5. **Synthesis:** The Chairman AI reviews all opinions and drafts the final authoritative response.

---

## 🛠️ Web Search Logic (Strict)
To maintain legal integrity, web search is governed by a strict user toggle:
* **Enabled:** All agents (Clerk, Council, Chairman) may access Google Search for recent rulings or supplementary facts.
* **Disabled:** Agents are strictly offline, relying exclusively on the retrieved RAG context and internal training.

---

## 🚀 POC Objectives & Benefits
* **Precision:** Specialized focus (e.g., Statutory vs. Case Law) significantly reduces "hallucinated" legal advice.
* **Speed:** Generic queries bypass heavy compute pipelines, ensuring a responsive UI.
* **Cost Efficiency:** Heavy-duty models (`Pro`) are reserved only for the final synthesis, while `Flash` handles high-volume routing.
* **User Agency:** Explicit control over data sources via the Web Search toggle.

---

## 📝 Contribution & Feedback
This is a living document. Please open an **Issue** or **Discussion** thread if you have suggestions regarding:
1. Agent specialization logic.
2. Vector DB chunking strategies for Indian Statutes.
3. Improving the Clerk's classification accuracy.